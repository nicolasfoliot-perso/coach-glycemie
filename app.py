"""Coach Glycémie — journal alimentaire privé assisté par une IA cloud.

Les données sont conservées dans le projet Supabase privé de l'utilisateur.
SQLite reste disponible comme sauvegarde locale de migration. Les mots de passe
ne sont jamais enregistrés par l'application.
"""

from __future__ import annotations

import json
import math
import re
import sqlite3
import tomllib
import unicodedata
from contextlib import contextmanager
from dataclasses import dataclass
from datetime import date, datetime, time, timedelta
from pathlib import Path
from time import sleep
from typing import Any, Iterator
from zoneinfo import ZoneInfo

import altair as alt
import httpx
import pandas as pd
import streamlit as st
from google import genai
from google.genai import types
from groq import Groq

from cloud_store import insert_row, select_rows, update_rows, upsert_row
from supabase_sync import (
    SupabaseSession,
    SupabaseSyncError,
    recovery_session_from_token,
    refresh_session,
    request_password_reset,
    session_from_storage_payload,
    session_storage_payload,
    sign_in,
    update_password,
)


APP_TITLE = "Coach Glycémie"
PASSWORD_RESET_REDIRECT_URL = "https://coach-glycemie-nicolas.streamlit.app/"
BROWSER_SESSION_STORAGE_KEY = "coach_glycemie_supabase_session_v1"
DB_PATH = Path(__file__).with_name("coach_glycemie.db")
SECRETS_DIR = Path(__file__).parent / ".streamlit"
SECRETS_PATH = SECRETS_DIR / "secrets.toml"
GEMINI_SECRET_NAME = "GEMINI_API_KEY"
GROQ_SECRET_NAME = "GROQ_API_KEY"
DEEPSEEK_SECRET_NAME = "DEEPSEEK_API_KEY"

# Ces modèles cloud multilingues acceptent les sorties JSON Schema strictes et
# ne consomment aucune mémoire dédiée au modèle sur le Mac.
DEEPSEEK_MODEL_NAME = "deepseek-flash"
GEMINI_MODEL_NAME = "gemini-3.1-flash-lite"
GROQ_MODEL_NAME = "qwen/qwen3.8-27b"
DEFAULT_COUNTRY = "Argentine"
DEFAULT_EATING_WINDOW_START = "10:30"
DEFAULT_EATING_WINDOW_HOURS = 10
EATING_WINDOW_HOUR_OPTIONS = [8, 10, 12]
REPORT_CONTEXT_VERSION = 3
LOCAL_TIMEZONE = ZoneInfo("America/Argentina/Buenos_Aires")
MEAL_TYPES = ["Petit déjeuner", "Déjeuner", "Goûter", "Dîner", "Souper", "Autre"]
PAIN_TYPES = [
    "Brûlure entre le sternum et le nombril",
    "Douleur aiguë sous le nombril",
    "Nœud / crampe abdominale",
    "Ballonnement / pression",
    "Autre douleur abdominale",
]
BRISTOL_TYPES = {
    1: "Petites billes dures, difficiles à évacuer",
    2: "En forme de saucisse mais grumeleuse",
    3: "En forme de saucisse avec des fissures",
    4: "Lisse, souple, en forme de saucisse ou de serpent",
    5: "Morceaux mous aux bords nets",
    6: "Morceaux duveteux, pâteux, aux bords irréguliers",
    7: "Entièrement liquide, sans morceau solide",
}
DEFAULT_CARBS_GOAL_G = 150.0
DEFAULT_PROTEIN_GOAL_G = 100.0
DEFAULT_FAT_GOAL_G = 70.0
DEFAULT_CALORIE_GOAL_KCAL = 2000.0
DEFAULT_MEALS_PER_DAY = 2
DEFAULT_HEIGHT_CM = 170.0
DEFAULT_AGE_YEARS = 56
DEFAULT_SEX = "Homme"
DEFAULT_ACTIVITY_LEVEL = "Modérée"
DEFAULT_TARGET_WEIGHT_KG = 76.0
WEIGHT_LOSS_DAILY_DEFICIT_KCAL = 350.0
WEIGHT_STABILIZATION_TRANSITION_KG = 1.0
BALANCED_CARB_SHARE_RANGE = (40, 50)
SEX_OPTIONS = ["Homme", "Femme"]
ACTIVITY_FACTORS = {
    "Sédentaire": 1.20,
    "Légère": 1.35,
    "Modérée": 1.45,
    "Élevée": 1.65,
}
ACTIVITY_TYPES = ["Natation", "Marche", "Vélo"]
ACTIVITY_INTENSITIES = ["Douce", "Modérée", "Soutenue"]
OBJECTIVE_OPTIONS = [
    "Stabiliser ma glycémie",
    "Perdre du poids progressivement",
    "Maintenir mon poids",
    "Prendre du poids progressivement",
    "Améliorer progressivement mes repas",
    "Mieux comprendre mes habitudes",
    "Gérer mon poids durablement",
]
OPEN_FOOD_FACTS_SEARCH_URL = "https://world.openfoodfacts.org/cgi/search.pl"
OPEN_FOOD_FACTS_STRUCTURED_SEARCH_URL = "https://world.openfoodfacts.org/api/v2/search"
OPEN_FOOD_FACTS_PRODUCT_URL = "https://world.openfoodfacts.org/api/v2/product/{code}.json"
OPEN_FOOD_FACTS_USER_AGENT = "CoachGlycemie/1.2 (application locale personnelle)"
OPEN_FOOD_FACTS_RETRYABLE_STATUS = {429, 502, 503, 504}


@dataclass(frozen=True)
class AISettings:
    """Clés disponibles et ordre de recours aux fournisseurs cloud."""

    deepseek_api_key: str = ""
    gemini_api_key: str = ""
    groq_api_key: str = ""

    @property
    def has_any_key(self) -> bool:
        return bool(
            self.deepseek_api_key or self.gemini_api_key or self.groq_api_key
        )


class AIProvidersUnavailable(RuntimeError):
    """Regroupe les erreurs lorsque tous les fournisseurs essayés échouent."""

    def __init__(self, failures: list[tuple[str, Exception]]) -> None:
        self.failures = failures
        super().__init__("Tous les fournisseurs IA configurés sont indisponibles.")

MEAL_SYSTEM_PROMPT = """
Tu es un coach nutritionnel prudent, bienveillant et factuel, spécialisé dans
l'impact probable des repas sur la glycémie et l'insuline.

PHILOSOPHIE ALIMENTAIRE DE L'APPLICATION
- L'utilisateur souhaite perdre environ 3 à 4 kg progressivement, puis maintenir
  son poids. Il ne souhaite ni régime keto ni restriction forte des glucides.
- Ne pénalise jamais un repas uniquement parce qu'il contient des glucides.
  Évalue plutôt leur quantité, leur qualité, les fibres, la portion et leur
  association avec protéines, légumes ou lipides.
- Ne conseille pas de supprimer globalement fruits entiers, légumineuses,
  avoine, céréales complètes, pommes de terre ou autres féculents. Si un
  ajustement est utile, propose d'abord une portion réaliste, une source moins
  raffinée, davantage de fibres ou une meilleure association au repas.
- Une approche très pauvre en glucides ou cétogène ne doit jamais être proposée
  spontanément.

MISSION
- Analyser uniquement le repas décrit par l'utilisateur.
- Estimer qualitativement son impact glycémique probable en tenant compte des
  glucides rapides, de la charge glucidique, des fibres, protéines, lipides,
  portions et du degré de transformation.
- Donner une note entière de 0 à 10 en additionnant cinq critères, chacun noté
  de 0 à 2 : qualité/quantité des glucides, présence de fibres ou légumes,
  présence de protéines, qualité des aliments et équilibre raisonnable de la
  portion. Un repas peut donc être favorable sans être parfait. Ne pénalise pas
  automatiquement un fruit entier, un produit laitier nature ou une portion
  modérée de féculents complets.
- Associer obligatoirement la couleur selon la note :
  * vert pour une note de 7 à 10 ;
  * orange pour une note de 4 à 6 ;
  * rouge pour une note de 0 à 3.
- Écrire un commentaire court, concret, encourageant, en français, avec au
  maximum trois astuces réalistes. Il doit faire au maximum 450 caractères et
  toujours se terminer par une phrase complète. Ne jamais culpabiliser.
- Ne pas diagnostiquer de maladie, ne pas prescrire de traitement et signaler
  clairement toute incertitude due à une portion ou préparation non précisée.
- Pour un produit de marque, utiliser en priorité les valeurs fournies dans la
  bibliothèque alimentaire personnelle. Utiliser ensuite les valeurs vérifiées
  Open Food Facts. Ne jamais inventer une composition exacte ni confondre deux
  variantes d'une même marque.
- Tenir compte du fait que l'utilisateur réside en Argentine : noms de produits,
  portions et habitudes locales peuvent différer de ceux d'autres pays.
- Extraire du texte la date, l'heure et le type du repas. Interpréter les
  expressions relatives (aujourd'hui, hier, ce matin, anoche, etc.) à partir
  de la date/heure locale de référence fournie par l'application.
- Ne jamais inventer une date ou une heure. Si l'une manque ou reste vague,
  laisser le champ correspondant vide, demander une précision courte et mettre
  "requires_timing_clarification" à true. "Ce matin" sans heure est vague ;
  "à midi" vaut 12:00 et "à minuit" vaut 00:00.
- Détecter le type à partir des mots employés (petit déjeuner/desayuno,
  déjeuner/almuerzo, goûter/merienda, dîner/cena, souper). En l'absence de libellé
  explicite, l'inférer à partir de l'heure exacte.
- La position dans la fenêtre alimentaire choisie est calculée ensuite par
  l'application. Ne pas présenter le jeûne intermittent comme un traitement.
- Estimer les grammes de glucides, protéines et lipides pour l'ensemble du
  repas. Utiliser les quantités décrites et les données produit vérifiées. Si
  les portions sont imprécises, donner une estimation centrale prudente et
  régler "confiance_macros" sur "faible" ou "moyenne".

FORMAT DE SORTIE IMPERATIF
Retourne uniquement un objet JSON valide, sans Markdown, sans bloc de code et
sans texte avant ou après. Utilise exactement ce schéma :
{
  "note": 7,
  "couleur": "vert",
  "commentaire": "Commentaire bienveillant et conseils pratiques.",
  "glucides_g": 35.0,
  "proteines_g": 24.0,
  "lipides_g": 18.0,
  "confiance_macros": "moyenne",
  "repas_date": "2026-09-16",
  "repas_heure": "12:30",
  "type_repas_detecte": "Déjeuner",
  "confiance_horaire": "élevée",
  "requires_timing_clarification": false,
  "timing_clarification_question": ""
}
La note doit être un entier entre 0 et 10. La couleur doit être exactement
"vert", "orange" ou "rouge". Le commentaire doit être une chaîne non vide.
Les trois quantités sont des nombres positifs ou nuls pour le repas complet.
"confiance_macros" vaut exactement "faible", "moyenne" ou "élevée".
"repas_date" est au format YYYY-MM-DD et "repas_heure" au format HH:MM, ou
une chaîne vide si l'information est ambiguë. "type_repas_detecte" est l'une
des catégories autorisées. "confiance_horaire" vaut "faible", "moyenne" ou
"élevée". La question de clarification est vide si aucune précision n'est
nécessaire.
""".strip()

REPORT_SYSTEM_PROMPT = """
Tu es un coach nutritionnel prudent et encourageant. À partir d'un journal de
repas déjà noté et, si disponibles, de quelques mesures de poids, tu résumes la
tendance probable de stabilité glycémique. Tu ne poses jamais de diagnostic et
tu ne prescris aucun traitement. Tu fondes tes conclusions uniquement sur les
données fournies, tu mentionnes les limites si l'historique est court, et tu
proposes des changements progressifs et réalistes.
Le projet alimentaire est durable et non cétogène : perdre environ 3 à 4 kg,
puis stabiliser le poids, tout en conservant des glucides de qualité. Ne propose
jamais spontanément une alimentation keto, très pauvre en glucides ou une
suppression générale des féculents et des fruits. Privilégie la qualité, les
portions, les fibres, l'équilibre du repas et la tendance énergétique sur
plusieurs semaines.
Si des sensations de faim ou douleurs abdominales sont fournies, tu peux relever
des associations temporelles répétées avec certains repas, mais tu ne présentes
jamais une association comme une cause et tu ne proposes aucun diagnostic. Tu
recommandes une évaluation médicale si les douleurs se répètent, s'aggravent ou
inquiètent l'utilisateur.
Ne mentionne jamais une ancienne douleur isolée qui n'a pas été signalée de
nouveau. Les données reçues sous "symptomes_actifs_ou_recurrents" ont déjà été
filtrées : une douleur n'y figure que si elle date de moins de 72 heures ou si
le même type s'est répété au moins deux fois sur 14 jours. L'absence de nouveau
signalement ne permet pas d'affirmer une guérison ; elle signifie seulement que
ce sujet ne doit pas encombrer le bilan actuel.
Si un journal des selles est fourni, ne propose un ajustement alimentaire que
si au moins trois observations récentes montrent une tendance répétée. Ne tire
aucune conclusion d'une selle isolée, ne diagnostique aucune maladie et conseille
une évaluation médicale en cas de sang, selles noires, douleur importante ou
changement persistant inquiétant.
Si des activités physiques sont fournies, utilise-les pour contextualiser la
faim, l'énergie et la tendance du poids. Leur dépense calorique est une estimation
et le niveau d'activité habituel peut déjà en intégrer une partie : ne les compte
pas automatiquement deux fois et ne recommande pas de compenser chaque calorie.
Les aliments personnels enregistrés peuvent préciser les calculs des produits
récurrents, mais leur étiquette peut évoluer.

Retourne uniquement un objet JSON valide, sans Markdown ni texte additionnel,
avec exactement ce schéma :
{
  "resume": "Synthèse factuelle de la tendance en 2 à 4 phrases.",
  "tendance": "amelioration",
  "conseils": [
    "Premier conseil personnalisé et concret.",
    "Deuxième conseil personnalisé et concret.",
    "Troisième conseil personnalisé et concret."
  ],
  "encouragement": "Une phrase finale motivante et non culpabilisante."
}
"tendance" doit être exactement "amelioration", "stable" ou "a_surveille".
"conseils" doit contenir exactement trois chaînes non vides.
""".strip()

CHAT_SYSTEM_PROMPT = """
Tu es l'assistant nutritionnel conversationnel d'un journal alimentaire
personnel. Tu adoptes la méthode de travail d'un nutritionniste-diététicien
expérimenté : raisonnement structuré, pédagogie, prudence et conseils alimentaires
concrets. Tu réponds en français, avec un ton clair, bienveillant et non
culpabilisant. Tu ne prétends jamais être un humain ou un professionnel diplômé.

APPROCHE NUTRITIONNELLE
- Analyse selon la question la structure des repas, les portions, la densité
  énergétique, les protéines, glucides, lipides, fibres, variété alimentaire,
  hydratation, horaires, satiété, activité et évolution du poids.
- Commence par les éléments les plus importants. Propose ensuite un à trois
  ajustements réalistes, hiérarchisés et expliqués, plutôt qu'une longue liste
  d'interdits.
- Lorsque les quantités sont suffisamment renseignées, utilise les valeurs du
  journal pour illustrer la réponse. Sinon, demande une précision ciblée ou
  formule une option avec une portion usuelle explicitement présentée comme une
  hypothèse.
- Pour une perte de poids progressive, juge la tendance sur plusieurs jours ou
  semaines. Ne recommande pas de restriction compensatoire à partir d'une seule
  journée, surtout si le journal est incomplet.
- Le projet de l'utilisateur est une perte progressive d'environ 3 à 4 kg,
  suivie d'une stabilisation. Il veut conserver une alimentation équilibrée avec
  des glucides et ne souhaite pas de régime keto ou très pauvre en glucides.
  Ne présente jamais les glucides comme mauvais par principe et ne propose pas
  spontanément de supprimer fruits entiers, légumineuses, avoine, céréales
  complètes ou féculents. Travaille d'abord sur leur qualité, leur portion, les
  fibres et leur association avec le reste du repas.
- Intègre la faim, l'énergie, le plaisir, la tolérance digestive et la faisabilité
  dans les recommandations. Le jeûne intermittent est une préférence
  d'organisation, pas un traitement ni un objectif à défendre à tout prix.
- Pour les produits et habitudes locales, privilégie des exemples disponibles en
  Argentine, sans inventer la composition exacte d'une marque non vérifiée.
- Tiens compte des activités sportives consignées, mais rappelle que leur dépense
  calorique reste estimative. Le niveau d'activité habituel du profil peut déjà
  inclure une partie de ces séances : ne double-compte pas automatiquement cette
  dépense et ne conseille pas de « manger en retour » toutes les calories.

RÈGLES DE FOND
- Réponds d'abord à la question posée, puis explique brièvement ce que les
  données du journal permettent réellement de conclure.
- Distingue toujours les aliments réellement consignés, les estimations de
  portions et les jours incomplets. Ne transforme jamais une absence de saisie
  en absence d'alimentation.
- Pour les calories, macronutriments et projections de poids, parle de repères
  éducatifs et d'estimations, jamais de prescription.
- Les aliments de la bibliothèque personnelle ont été saisis par l'utilisateur.
  Tu peux utiliser leurs valeurs pour reconnaître les aliments récurrents, tout
  en signalant qu'une recette ou une étiquette peut changer.
- Ne diagnostique aucune maladie, ne prescris aucun traitement et ne recommande
  jamais de modifier ou d'arrêter un médicament.
- Ne présente jamais une association entre un repas et un symptôme comme une
  causalité. Les anciennes douleurs isolées ont déjà été retirées du contexte :
  ne les réintroduis pas et ne prétends pas qu'elles sont guéries.
- N'interprète le journal des selles que si "analyse_autorisee" vaut true.
- Si la question ou le contexte décrit un signe potentiellement urgent (douleur
  intense ou croissante, sang, selles noires, malaise, douleur thoracique,
  difficulté à respirer ou autre signe inquiétant), recommande clairement une
  évaluation médicale rapide ou les services d'urgence locaux.
- Si une donnée nécessaire manque, dis-le simplement et propose quoi consigner
  ensuite. N'invente aucune mesure, marque ou valeur nutritionnelle.
- Tiens compte du marché alimentaire argentin lorsque c'est pertinent, sans
  prétendre avoir vérifié une fiche produit absente du contexte.
- Garde la réponse facile à lire, généralement entre 2 et 6 courts paragraphes
  ou une petite liste. Les suggestions finales doivent être des questions que
  l'utilisateur peut poser pour poursuivre l'échange.

FORMAT DE SORTIE IMPÉRATIF
Retourne uniquement un objet JSON valide, sans Markdown autour de l'objet, avec
exactement ce schéma logique : une réponse, jusqu'à trois suggestions de suivi,
un indicateur de besoin d'avis médical et un message de sécurité éventuellement
vide. Le champ "reponse" peut contenir du Markdown simple.
""".strip()

SYMPTOM_SYSTEM_PROMPT = """
Tu extrais des événements de faim et de douleur abdominale décrits en français
ou en espagnol. Tu es un extracteur de données, pas un médecin.

RÈGLES
- Un même texte peut contenir plusieurs événements : crée un élément par événement.
- Interprète aujourd'hui, hier, avant-hier et les jours de la semaine à partir de
  la date/heure locale de référence fournie par l'application.
- "declencheur_rapporte" contient uniquement ce que l'utilisateur présente ou
  soupçonne lui-même comme origine (aliment, boisson, stress, jeûne, etc.). Ne
  déduis jamais une cause et ne pose jamais de diagnostic.
- Classe une douleur dans l'une des catégories autorisées. "Nœud / crampe
  abdominale" est une description, pas un diagnostic de côlon irritable.
- L'intensité doit être comprise entre 1 et 10. Si elle manque, mets 0 et demande
  une précision. Une intensité qualitative explicite peut être convertie avec
  prudence : petite/faible/légère = 3, moyenne/modérée = 5, forte/intense = 8,
  insupportable = 10.
- L'heure et la date doivent être suffisamment précises. Pour une expression vague
  comme "le matin", "après manger" ou "il y a quelques jours", mets occurred_at
  à "", requires_clarification à true et pose une question courte.
- Si le texte ne contient ni faim ni douleur, retourne "events" comme liste vide.
- N'ajoute aucun événement qui n'est pas explicitement rapporté.
- Indique aussi si le texte décrit explicitement des aliments ou boissons
  réellement consommés. "Après le déjeuner" ou "le café a peut-être causé"
  ne suffit pas : contains_meal reste false. Si un vrai repas est décrit,
  recopie dans meal_description uniquement sa description avec sa date et son
  heure, afin qu'un second analyseur nutritionnel puisse le traiter.

Retourne uniquement le JSON imposé par le schéma, sans Markdown.
""".strip()

BOWEL_SYSTEM_PROMPT = """
Tu extrais des passages aux toilettes décrits en français ou en espagnol. Tu es
un extracteur de données, pas un médecin.

RÈGLES
- Crée un événement par selle explicitement décrite.
- Interprète les dates et heures relatives à partir de la référence fournie.
- Classe la forme selon l'échelle de Bristol 1 à 7 à partir du numéro donné ou
  de la description. N'invente pas un type si la forme n'est pas assez précise.
- Si la date, l'heure ou la forme manque, mets occurred_at à "" ou
  bristol_type à 0, active requires_clarification et pose une question courte.
- La description reprend seulement les faits rapportés : forme, couleur,
  difficulté, urgence ou sensation d'évacuation, sans diagnostic.
- Si le texte ne décrit aucune selle réellement passée, retourne une liste vide.

Retourne uniquement un objet JSON valide, sans Markdown. Exemple :
{
  "events": [{
    "occurred_at": "2026-09-21T08:30:00-03:00",
    "bristol_type": 4,
    "description": "Selle lisse et souple",
    "confidence": "élevée",
    "requires_clarification": false,
    "clarification_question": ""
  }]
}
""".strip()

FOOD_LABEL_SYSTEM_PROMPT = """
Tu extrais les données d'une étiquette nutritionnelle décrite en français ou en
espagnol. Tu es un extracteur de données, pas une base de produits en ligne.

RÈGLES
- Utilise uniquement les valeurs explicitement présentes dans le texte fourni.
- N'invente jamais une valeur à partir de ta connaissance d'une marque.
- Si les valeurs sont données par portion, convertis-les mathématiquement pour
  100 g à partir du poids explicite de la portion.
- Si les calories ne sont pas indiquées mais que glucides, protéines et lipides
  le sont, laisse calories_100g à 0 : l'application les calculera ensuite.
- Si une valeur manque, mets 0 et ajoute son nom à champs_manquants.
- "donnees_explicites" vaut true seulement si le texte contient au moins une
  valeur nutritionnelle et une base exploitable (pour 100 g ou portion en g).
- Ne fais aucune recherche web et ne prétends jamais avoir vérifié le fabricant.

Retourne uniquement le JSON imposé par le schéma, sans Markdown.
""".strip()

MEAL_RESPONSE_SCHEMA = {
    "type": "object",
    "properties": {
        "note": {"type": "integer", "minimum": 0, "maximum": 10},
        "couleur": {"type": "string", "enum": ["vert", "orange", "rouge"]},
        "commentaire": {"type": "string"},
        "glucides_g": {"type": "number", "minimum": 0, "maximum": 1000},
        "proteines_g": {"type": "number", "minimum": 0, "maximum": 1000},
        "lipides_g": {"type": "number", "minimum": 0, "maximum": 1000},
        "confiance_macros": {
            "type": "string",
            "enum": ["faible", "moyenne", "élevée"],
        },
        "repas_date": {"type": "string"},
        "repas_heure": {"type": "string"},
        "type_repas_detecte": {"type": "string", "enum": MEAL_TYPES},
        "confiance_horaire": {
            "type": "string",
            "enum": ["faible", "moyenne", "élevée"],
        },
        "requires_timing_clarification": {"type": "boolean"},
        "timing_clarification_question": {"type": "string"},
    },
    "required": [
        "note", "couleur", "commentaire", "glucides_g", "proteines_g",
        "lipides_g", "confiance_macros", "repas_date", "repas_heure",
        "type_repas_detecte", "confiance_horaire",
        "requires_timing_clarification", "timing_clarification_question",
    ],
    "additionalProperties": False,
}

REPORT_RESPONSE_SCHEMA = {
    "type": "object",
    "properties": {
        "resume": {"type": "string"},
        "tendance": {
            "type": "string",
            "enum": ["amelioration", "stable", "a_surveille"],
        },
        "conseils": {"type": "array", "items": {"type": "string"}},
        "encouragement": {"type": "string"},
    },
    "required": ["resume", "tendance", "conseils", "encouragement"],
    "additionalProperties": False,
}

CHAT_RESPONSE_SCHEMA = {
    "type": "object",
    "properties": {
        "reponse": {"type": "string"},
        "suggestions": {
            "type": "array",
            "items": {"type": "string"},
            "maxItems": 3,
        },
        "avis_medical_recommande": {"type": "boolean"},
        "message_securite": {"type": "string"},
    },
    "required": [
        "reponse",
        "suggestions",
        "avis_medical_recommande",
        "message_securite",
    ],
    "additionalProperties": False,
}

SYMPTOM_RESPONSE_SCHEMA = {
    "type": "object",
    "properties": {
        "events": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "kind": {"type": "string", "enum": ["faim", "douleur"]},
                    "occurred_at": {"type": "string"},
                    "pain_type": {"type": "string", "enum": ["", *PAIN_TYPES]},
                    "intensity": {"type": "integer", "minimum": 0, "maximum": 10},
                    "description": {"type": "string"},
                    "declencheur_rapporte": {"type": "string"},
                    "confidence": {
                        "type": "string",
                        "enum": ["faible", "moyenne", "élevée"],
                    },
                    "requires_clarification": {"type": "boolean"},
                    "clarification_question": {"type": "string"},
                },
                "required": [
                    "kind", "occurred_at", "pain_type", "intensity",
                    "description", "declencheur_rapporte", "confidence",
                    "requires_clarification", "clarification_question",
                ],
                "additionalProperties": False,
            },
        },
        "contains_meal": {"type": "boolean"},
        "meal_description": {"type": "string"},
    },
    "required": ["events", "contains_meal", "meal_description"],
    "additionalProperties": False,
}

BOWEL_RESPONSE_SCHEMA = {
    "type": "object",
    "properties": {
        "events": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "occurred_at": {"type": "string"},
                    "bristol_type": {
                        "type": "integer",
                        "minimum": 0,
                        "maximum": 7,
                    },
                    "description": {"type": "string"},
                    "confidence": {
                        "type": "string",
                        "enum": ["faible", "moyenne", "élevée"],
                    },
                    "requires_clarification": {"type": "boolean"},
                    "clarification_question": {"type": "string"},
                },
                "required": [
                    "occurred_at",
                    "bristol_type",
                    "description",
                    "confidence",
                    "requires_clarification",
                    "clarification_question",
                ],
                "additionalProperties": False,
            },
        }
    },
    "required": ["events"],
    "additionalProperties": False,
}

FOOD_LABEL_RESPONSE_SCHEMA = {
    "type": "object",
    "properties": {
        "nom": {"type": "string"},
        "marque": {"type": "string"},
        "portion_reference_g": {"type": "number", "minimum": 0, "maximum": 5000},
        "calories_100g": {"type": "number", "minimum": 0, "maximum": 1000},
        "glucides_100g": {"type": "number", "minimum": 0, "maximum": 100},
        "proteines_100g": {"type": "number", "minimum": 0, "maximum": 100},
        "lipides_100g": {"type": "number", "minimum": 0, "maximum": 100},
        "fibres_100g": {"type": "number", "minimum": 0, "maximum": 100},
        "donnees_explicites": {"type": "boolean"},
        "confiance": {
            "type": "string",
            "enum": ["faible", "moyenne", "élevée"],
        },
        "champs_manquants": {
            "type": "array",
            "items": {"type": "string"},
        },
        "note": {"type": "string"},
    },
    "required": [
        "nom", "marque", "portion_reference_g", "calories_100g",
        "glucides_100g", "proteines_100g", "lipides_100g", "fibres_100g",
        "donnees_explicites", "confiance", "champs_manquants", "note",
    ],
    "additionalProperties": False,
}

COLOR_META = {
    "vert": {"label": "Profil glycémique favorable", "emoji": "🟢", "hex": "#22c55e"},
    "orange": {"label": "Repas à équilibrer", "emoji": "🟠", "hex": "#f59e0b"},
    "rouge": {"label": "Charge glycémique probablement élevée", "emoji": "🔴", "hex": "#ef4444"},
}

MACRO_STATUS_META = {
    "bas": {"label": "Sous le repère indicatif", "emoji": "🔵", "hex": "#3b82f6"},
    "vert": {"label": "Dans la zone cible", "emoji": "🟢", "hex": "#22c55e"},
    "orange": {"label": "Proche de la zone cible", "emoji": "🟠", "hex": "#f59e0b"},
    "rouge": {"label": "Loin de la zone cible", "emoji": "🔴", "hex": "#ef4444"},
}


def score_color(score: int | float) -> str:
    """Applique le repère éducatif affiché, distinct d'un seuil médical."""
    numeric_score = int(score)
    return "vert" if numeric_score >= 7 else "orange" if numeric_score >= 4 else "rouge"


DAILY_STATUS_META = {
    "vert": {"label": "Dans le cap", "emoji": "🟢"},
    "orange": {"label": "À ajuster", "emoji": "🟠"},
    "rouge": {"label": "Écart important", "emoji": "🔴"},
    "en_cours": {"label": "Journée en cours", "emoji": "🔵"},
}


def daily_energy_status(
    calories_kcal: float,
    target_kcal: float,
    *,
    day_is_complete: bool,
) -> tuple[float, str]:
    """Classe l'apport face à une cible déjà ajustée à l'objectif de poids."""
    ratio = calories_kcal / target_kcal if target_kcal > 0 else 0.0
    if not day_is_complete or calories_kcal <= 0 or target_kcal <= 0:
        return ratio, "en_cours"
    if 0.85 <= ratio <= 1.10:
        return ratio, "vert"
    if 0.70 <= ratio <= 1.20:
        return ratio, "orange"
    return ratio, "rouge"


def combined_daily_status(
    glycemic_status: str | None,
    energy_status: str,
) -> str:
    """Synthétise les deux axes sans laisser les calories masquer la glycémie."""
    if glycemic_status is None or energy_status == "en_cours":
        return "en_cours"
    if "rouge" in {glycemic_status, energy_status}:
        return "rouge"
    if glycemic_status == "vert" and energy_status == "vert":
        return "vert"
    return "orange"


def eating_window_has_ended_today(start_value: str, window_hours: int) -> bool:
    """Indique si la fenêtre alimentaire du jour est terminée à l'heure locale."""
    try:
        start_hour, start_minute = map(int, start_value.split(":"))
    except (AttributeError, TypeError, ValueError):
        return False
    start_total = start_hour * 60 + start_minute
    end_unwrapped = start_total + normalized_eating_window_hours(window_hours) * 60
    now = datetime.now(LOCAL_TIMEZONE)
    current_total = now.hour * 60 + now.minute
    if end_unwrapped < 24 * 60:
        return current_total >= end_unwrapped
    end_total = end_unwrapped % (24 * 60)
    return end_total <= current_total < start_total


st.set_page_config(
    page_title=APP_TITLE,
    page_icon="🥗",
    layout="wide",
    initial_sidebar_state="collapsed",
)


_RECOVERY_HASH_BRIDGE = st.components.v2.component(
    "coach_password_recovery_hash_bridge",
    html="""
    <span aria-hidden="true" style="display:none"></span>
    """,
    js="""
    export default function (component) {
      const { setTriggerValue } = component
      let hash = ""
      try {
        hash = window.top.location.hash || window.parent.location.hash || window.location.hash
      } catch (_) {
        hash = window.location.hash
      }
      if (!hash || !hash.includes("type=recovery")) return

      const params = new URLSearchParams(hash.replace(/^#/, ""))
      const accessToken = params.get("access_token") || ""
      if (!accessToken) return

      setTriggerValue("payload", {
        access_token: accessToken,
        refresh_token: params.get("refresh_token") || "",
        expires_at: Number(params.get("expires_at") || 0),
        type: params.get("type") || "",
      })

      try {
        const cleanUrl = window.top.location.pathname + window.top.location.search
        window.top.history.replaceState(null, "", cleanUrl)
      } catch (_) {
        window.history.replaceState(null, "", window.location.pathname + window.location.search)
      }
    }
    """,
)


_SESSION_STORAGE_BRIDGE = st.components.v2.component(
    "coach_persistent_session_storage_bridge",
    html="""
    <span aria-hidden="true" style="display:none"></span>
    """,
    js="""
    export default function (component) {
      const { data, setTriggerValue } = component
      const action = data?.action || ""
      const nonce = data?.nonce || ""
      const storageKey = data?.storage_key || ""
      if (!action || !storageKey) return

      if (action === "load") {
        let session = null
        try {
          const raw = window.localStorage.getItem(storageKey)
          session = raw ? JSON.parse(raw) : null
        } catch (_) {
          window.localStorage.removeItem(storageKey)
        }
        setTriggerValue("loaded", {
          nonce,
          found: Boolean(session),
          session,
        })
        return
      }

      if (action === "save") {
        try {
          window.localStorage.setItem(
            storageKey,
            JSON.stringify(data?.session || {})
          )
          setTriggerValue("saved", { nonce, ok: true })
        } catch (_) {
          setTriggerValue("saved", { nonce, ok: false })
        }
        return
      }

      if (action === "clear") {
        window.localStorage.removeItem(storageKey)
        setTriggerValue("cleared", { nonce, ok: true })
      }
    }
    """,
)


def configured_secret(name: str) -> str:
    """Lit un secret public de configuration sans exposer sa valeur."""
    try:
        return str(st.secrets.get(name, "")).strip()
    except (FileNotFoundError, KeyError):
        return ""


def active_cloud_session() -> SupabaseSession | None:
    session = st.session_state.get("supabase_session")
    return session if isinstance(session, SupabaseSession) else None


def queue_browser_session_action(
    action: str,
    session: SupabaseSession | None = None,
) -> None:
    """Programme une écriture ou suppression de session dans ce navigateur."""
    payload: dict[str, Any] = {
        "action": action,
        "nonce": f"{action}-{datetime.now().timestamp()}",
    }
    if action == "save" and session is not None:
        payload["session"] = session_storage_payload(session)
    st.session_state["browser_session_action"] = payload


def process_persisted_cloud_session(
    project_url: str,
    publishable_key: str,
) -> None:
    """Restaure ou met à jour la session choisie sur cet appareil personnel."""
    pending = st.session_state.get("browser_session_action")
    if isinstance(pending, dict):
        action = str(pending.get("action") or "")
        nonce = str(pending.get("nonce") or "")
        result = _SESSION_STORAGE_BRIDGE(
            key="persistent_cloud_session_bridge",
            data={
                "action": action,
                "nonce": nonce,
                "storage_key": BROWSER_SESSION_STORAGE_KEY,
                "session": pending.get("session") or {},
            },
            on_loaded_change=lambda: None,
            on_saved_change=lambda: None,
            on_cleared_change=lambda: None,
        )
        acknowledgement = getattr(
            result,
            {"save": "saved", "clear": "cleared"}.get(action, "loaded"),
            None,
        )
        if isinstance(acknowledgement, dict) and str(
            acknowledgement.get("nonce") or ""
        ) == nonce:
            st.session_state.pop("browser_session_action", None)
            if action == "save" and not acknowledgement.get("ok", False):
                st.session_state["persistent_session_error"] = (
                    "Safari n'a pas pu mémoriser la session sur cet appareil."
                )
        return

    if st.session_state.get("persistent_session_checked"):
        return

    restore_nonce = "restore-v1"
    result = _SESSION_STORAGE_BRIDGE(
        key="persistent_cloud_session_bridge",
        data={
            "action": "load",
            "nonce": restore_nonce,
            "storage_key": BROWSER_SESSION_STORAGE_KEY,
        },
        on_loaded_change=lambda: None,
        on_saved_change=lambda: None,
        on_cleared_change=lambda: None,
    )
    loaded = getattr(result, "loaded", None)
    if not isinstance(loaded, dict) or loaded.get("nonce") != restore_nonce:
        return

    st.session_state["persistent_session_checked"] = True
    if not loaded.get("found") or not isinstance(loaded.get("session"), dict):
        return

    try:
        remembered_session = session_from_storage_payload(
            project_url,
            publishable_key,
            loaded["session"],
        )
        remembered_session = refresh_session(remembered_session)
    except (TypeError, ValueError, SupabaseSyncError):
        queue_browser_session_action("clear")
        st.session_state["persistent_session_error"] = (
            "La session mémorisée a expiré. Reconnectez-vous une fois."
        )
    else:
        st.session_state["supabase_session"] = remembered_session
        st.session_state["cloud_login_email"] = remembered_session.email
        st.session_state["remember_cloud_session"] = True
        queue_browser_session_action("save", remembered_session)
    st.rerun()


def capture_password_recovery_session(
    project_url: str,
    publishable_key: str,
) -> None:
    """Recupere une seule fois le jeton place dans le fragment du lien Supabase."""
    result = _RECOVERY_HASH_BRIDGE(
        key="password_recovery_hash_bridge",
        on_payload_change=lambda: None,
    )
    payload = getattr(result, "payload", None)
    if not isinstance(payload, dict) or payload.get("type") != "recovery":
        return
    try:
        recovery_session = recovery_session_from_token(
            project_url,
            publishable_key,
            str(payload.get("access_token") or ""),
            str(payload.get("refresh_token") or ""),
            int(payload.get("expires_at") or 0),
        )
    except (TypeError, ValueError, SupabaseSyncError) as error:
        st.session_state["password_recovery_error"] = str(error)
        st.session_state["cloud_auth_view"] = "reset"
    else:
        st.session_state["password_recovery_session"] = recovery_session
        st.session_state["password_reset_email"] = recovery_session.email
        st.session_state["cloud_auth_view"] = "reset_link"
    st.rerun()


def require_cloud_login() -> SupabaseSession:
    """Bloque le journal tant que l'identité Supabase n'est pas établie."""
    project_url = configured_secret("SUPABASE_URL")
    publishable_key = configured_secret("SUPABASE_PUBLISHABLE_KEY")
    if not project_url or not publishable_key:
        st.error(
            "La connexion au cloud privé n'est pas encore configurée. "
            "Lancez d'abord « Configurer le cloud » sur le Mac."
        )
        st.stop()

    capture_password_recovery_session(project_url, publishable_key)
    process_persisted_cloud_session(project_url, publishable_key)

    session = active_cloud_session()
    if session is not None and session.expires_at:
        if session.expires_at <= int(datetime.now().timestamp()) + 120:
            try:
                session = refresh_session(session)
                st.session_state["supabase_session"] = session
                if st.session_state.get("remember_cloud_session"):
                    queue_browser_session_action("save", session)
                    st.rerun()
            except SupabaseSyncError:
                st.session_state.pop("supabase_session", None)
                session = None

    if session is not None:
        return session

    st.markdown(
        """
        <section class="hero">
          <h1>🥗 Coach Glycémie</h1>
          <p>Vos données sont maintenant protégées dans votre cloud privé.</p>
        </section>
        """,
        unsafe_allow_html=True,
    )
    st.session_state.setdefault("cloud_auth_view", "login")
    with st.container(border=True):
        if st.session_state["cloud_auth_view"] == "reset_link":
            st.subheader("Choisir mon nouveau mot de passe")
            st.caption(
                "Votre lien sécurisé a été vérifié. Choisissez maintenant un "
                "nouveau mot de passe."
            )
            with st.form("password_reset_confirm_form", clear_on_submit=False):
                new_password = st.text_input(
                    "Nouveau mot de passe",
                    type="password",
                    autocomplete="new-password",
                )
                confirm_password = st.text_input(
                    "Confirmer le nouveau mot de passe",
                    type="password",
                    autocomplete="new-password",
                )
                reset_confirmed = st.form_submit_button(
                    "Changer mon mot de passe",
                    type="primary",
                    icon=":material/lock_reset:",
                    width="stretch",
                )
            if reset_confirmed:
                recovery_session = st.session_state.get("password_recovery_session")
                if not isinstance(recovery_session, SupabaseSession):
                    st.error(
                        "Le lien a expiré. Revenez à la connexion et demandez-en un nouveau."
                    )
                elif len(new_password) < 8:
                    st.warning(
                        "Choisissez un mot de passe d'au moins huit caractères."
                    )
                elif new_password != confirm_password:
                    st.warning("Les deux mots de passe ne correspondent pas.")
                else:
                    try:
                        update_password(recovery_session, new_password)
                        st.session_state["supabase_session"] = recovery_session
                        st.session_state["cloud_login_email"] = recovery_session.email
                        st.session_state.pop("password_recovery_session", None)
                        st.session_state.pop("password_recovery_error", None)
                        st.session_state["remember_cloud_session"] = False
                        queue_browser_session_action("clear")
                        st.session_state["cloud_auth_view"] = "login"
                        st.rerun()
                    except SupabaseSyncError as error:
                        st.error(
                            "Le lien est invalide ou expiré. Demandez-en un nouveau. "
                            f"Détail : {error}"
                        )
            if st.button(
                "Demander un nouveau lien",
                icon=":material/refresh:",
                width="stretch",
                key="request_another_password_link",
            ):
                st.session_state.pop("password_recovery_session", None)
                st.session_state["cloud_auth_view"] = "reset"
                st.rerun()
        elif st.session_state["cloud_auth_view"] == "reset":
            st.subheader("Réinitialiser mon mot de passe")
            st.caption(
                "Recevez un lien sécurisé par e-mail. En l'ouvrant, vous "
                "reviendrez ici pour choisir un nouveau mot de passe."
            )
            if reset_error := st.session_state.pop("password_recovery_error", ""):
                st.error(
                    "Le lien est invalide ou expiré. Demandez-en un nouveau. "
                    f"Détail : {reset_error}"
                )
            with st.form("password_reset_request_form", clear_on_submit=False):
                reset_email = st.text_input(
                    "Adresse e-mail du Coach",
                    value=str(
                        st.session_state.get(
                            "password_reset_email",
                            st.session_state.get("cloud_login_email", ""),
                        )
                    ),
                    autocomplete="email",
                    key="password_reset_email_input",
                ).strip()
                reset_requested = st.form_submit_button(
                    "Recevoir le lien sécurisé",
                    type="primary",
                    icon=":material/mail:",
                    width="stretch",
                )
            if reset_requested:
                if not reset_email:
                    st.warning("Indiquez votre adresse e-mail.")
                else:
                    try:
                        request_password_reset(
                            project_url,
                            publishable_key,
                            reset_email,
                            PASSWORD_RESET_REDIRECT_URL,
                        )
                        st.session_state["password_reset_email"] = reset_email
                        st.session_state["password_reset_requested"] = True
                        st.success(
                            "Si cette adresse correspond au Coach, un e-mail "
                            "contenant le lien vient d'être envoyé. Ouvrez-le "
                            "sur cet appareil pour revenir ici."
                        )
                    except (ValueError, SupabaseSyncError) as error:
                        st.error(f"Envoi impossible : {error}")

            if st.button(
                "Retour à la connexion",
                icon=":material/arrow_back:",
                width="stretch",
                key="back_to_cloud_login",
            ):
                st.session_state["cloud_auth_view"] = "login"
                st.rerun()
        else:
            st.subheader("Ouvrir mon journal")
            st.caption(
                "Utilisez l'adresse et le mot de passe créés pour le Coach. "
                "Le mot de passe n'est pas conservé."
            )
            if persistent_error := st.session_state.pop(
                "persistent_session_error", ""
            ):
                st.info(persistent_error)
            with st.form("supabase_login_form", clear_on_submit=False):
                email = st.text_input(
                    "Adresse e-mail",
                    value=str(st.session_state.get("cloud_login_email", "")),
                    autocomplete="email",
                ).strip()
                password = st.text_input(
                    "Mot de passe",
                    type="password",
                    autocomplete="current-password",
                )
                remember_session = st.checkbox(
                    "Rester connecté sur cet appareil personnel",
                    value=False,
                    help=(
                        "Le mot de passe n'est jamais enregistré. Seule une session "
                        "Supabase renouvelable est conservée dans ce navigateur."
                    ),
                )
                submitted = st.form_submit_button(
                    "Me connecter",
                    type="primary",
                    icon=":material/login:",
                    width="stretch",
                )
            if submitted:
                if not email or not password:
                    st.warning("Indiquez votre adresse e-mail et votre mot de passe.")
                else:
                    try:
                        signed_in_session = sign_in(
                            project_url,
                            publishable_key,
                            email,
                            password,
                        )
                        st.session_state["supabase_session"] = signed_in_session
                        st.session_state["cloud_login_email"] = email
                        st.session_state["remember_cloud_session"] = remember_session
                        queue_browser_session_action(
                            "save" if remember_session else "clear",
                            signed_in_session if remember_session else None,
                        )
                        st.rerun()
                    except (ValueError, SupabaseSyncError) as error:
                        st.error(f"Connexion impossible : {error}")
            if st.button(
                "Mot de passe oublié ?",
                icon=":material/lock_reset:",
                width="stretch",
                key="open_password_reset",
            ):
                st.session_state["cloud_auth_view"] = "reset"
                st.rerun()
    st.stop()


@contextmanager
def database() -> Iterator[sqlite3.Connection]:
    """Ouvre une connexion SQLite courte et la ferme systematiquement."""
    connection = sqlite3.connect(DB_PATH, timeout=10)
    connection.row_factory = sqlite3.Row
    connection.execute("PRAGMA foreign_keys = ON")
    connection.execute("PRAGMA journal_mode = WAL")
    try:
        yield connection
        connection.commit()
    finally:
        connection.close()


def init_database() -> None:
    with database() as connection:
        connection.executescript(
            """
            CREATE TABLE IF NOT EXISTS meals (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                created_at TEXT NOT NULL,
                description TEXT NOT NULL,
                score INTEGER NOT NULL CHECK(score BETWEEN 0 AND 10),
                color TEXT NOT NULL CHECK(color IN ('vert', 'orange', 'rouge')),
                comment TEXT NOT NULL,
                model_name TEXT NOT NULL
            );

            CREATE TABLE IF NOT EXISTS weights (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                measured_on TEXT NOT NULL,
                weight_kg REAL NOT NULL CHECK(weight_kg > 0),
                note TEXT NOT NULL DEFAULT ''
            );

            CREATE TABLE IF NOT EXISTS symptom_events (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                occurred_at TEXT NOT NULL,
                kind TEXT NOT NULL CHECK(kind IN ('faim', 'douleur')),
                pain_type TEXT NOT NULL DEFAULT '',
                intensity INTEGER NOT NULL CHECK(intensity BETWEEN 1 AND 10),
                note TEXT NOT NULL DEFAULT '',
                meal_id INTEGER,
                delay_minutes INTEGER,
                created_at TEXT NOT NULL,
                FOREIGN KEY(meal_id) REFERENCES meals(id) ON DELETE SET NULL
            );

            CREATE TABLE IF NOT EXISTS bowel_events (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                occurred_at TEXT NOT NULL,
                bristol_type INTEGER NOT NULL CHECK(bristol_type BETWEEN 1 AND 7),
                description TEXT NOT NULL DEFAULT '',
                source_text TEXT NOT NULL DEFAULT '',
                extraction_confidence TEXT NOT NULL DEFAULT 'manuelle',
                created_at TEXT NOT NULL
            );

            CREATE TABLE IF NOT EXISTS activity_events (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                occurred_on TEXT NOT NULL,
                activity_type TEXT NOT NULL
                    CHECK(activity_type IN ('Natation', 'Marche', 'Vélo')),
                intensity TEXT NOT NULL DEFAULT 'Modérée',
                duration_minutes REAL NOT NULL DEFAULT 0,
                steps INTEGER NOT NULL DEFAULT 0,
                distance_km REAL NOT NULL DEFAULT 0,
                calories_kcal REAL NOT NULL CHECK(calories_kcal >= 0),
                estimation_method TEXT NOT NULL DEFAULT '',
                note TEXT NOT NULL DEFAULT '',
                created_at TEXT NOT NULL
            );

            CREATE TABLE IF NOT EXISTS favorite_foods (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                name TEXT NOT NULL,
                brand TEXT NOT NULL DEFAULT '',
                aliases TEXT NOT NULL DEFAULT '',
                serving_size_g REAL NOT NULL DEFAULT 0,
                calories_100g REAL NOT NULL DEFAULT 0,
                carbs_100g REAL NOT NULL DEFAULT 0,
                protein_100g REAL NOT NULL DEFAULT 0,
                fat_100g REAL NOT NULL DEFAULT 0,
                fiber_100g REAL NOT NULL DEFAULT 0,
                notes TEXT NOT NULL DEFAULT '',
                country TEXT NOT NULL DEFAULT 'Argentine',
                data_source TEXT NOT NULL DEFAULT 'saisie_manuelle',
                barcode TEXT NOT NULL DEFAULT '',
                source_url TEXT NOT NULL DEFAULT '',
                updated_at TEXT NOT NULL,
                UNIQUE(name, brand)
            );

            CREATE TABLE IF NOT EXISTS profile (
                id INTEGER PRIMARY KEY CHECK(id = 1),
                display_name TEXT NOT NULL DEFAULT '',
                objective TEXT NOT NULL DEFAULT 'Stabiliser ma glycémie',
                country TEXT NOT NULL DEFAULT 'Argentine',
                eating_window_start TEXT NOT NULL DEFAULT '10:30',
                eating_window_hours INTEGER NOT NULL DEFAULT 10,
                height_cm REAL NOT NULL DEFAULT 170,
                age_years INTEGER NOT NULL DEFAULT 56,
                sex TEXT NOT NULL DEFAULT 'Homme',
                activity_level TEXT NOT NULL DEFAULT 'Modérée',
                target_weight_kg REAL NOT NULL DEFAULT 76,
                auto_macro_targets INTEGER NOT NULL DEFAULT 1,
                carbs_goal_g REAL NOT NULL DEFAULT 150,
                protein_goal_g REAL NOT NULL DEFAULT 100,
                fat_goal_g REAL NOT NULL DEFAULT 70,
                calorie_goal_kcal REAL NOT NULL DEFAULT 2000,
                meals_per_day INTEGER NOT NULL DEFAULT 2,
                updated_at TEXT NOT NULL
            );

            CREATE INDEX IF NOT EXISTS idx_meals_created_at
            ON meals(created_at DESC);

            CREATE INDEX IF NOT EXISTS idx_weights_measured_on
            ON weights(measured_on ASC);

            CREATE INDEX IF NOT EXISTS idx_symptom_events_occurred_at
            ON symptom_events(occurred_at DESC);

            CREATE INDEX IF NOT EXISTS idx_bowel_events_occurred_at
            ON bowel_events(occurred_at DESC);

            CREATE INDEX IF NOT EXISTS idx_activity_events_occurred_on
            ON activity_events(occurred_on DESC);

            CREATE INDEX IF NOT EXISTS idx_favorite_foods_name
            ON favorite_foods(name COLLATE NOCASE, brand COLLATE NOCASE);
            """
        )

        # Migrations non destructives pour les bases créées par une version
        # antérieure de l'application.
        meal_columns = {
            row["name"] for row in connection.execute("PRAGMA table_info(meals)")
        }
        if "market_country" not in meal_columns:
            connection.execute(
                "ALTER TABLE meals ADD COLUMN market_country TEXT NOT NULL DEFAULT 'Argentine'"
            )
        if "product_matches_json" not in meal_columns:
            connection.execute(
                "ALTER TABLE meals ADD COLUMN product_matches_json TEXT NOT NULL DEFAULT '[]'"
            )
        meal_migrations = {
            "meal_occurred_at": "TEXT NOT NULL DEFAULT ''",
            "meal_type": "TEXT NOT NULL DEFAULT 'Autre'",
            "meal_time": "TEXT NOT NULL DEFAULT ''",
            "meal_type_source": "TEXT NOT NULL DEFAULT ''",
            "eating_window_start": "TEXT NOT NULL DEFAULT '12:00'",
            "eating_window_hours": "INTEGER NOT NULL DEFAULT 8",
            "inside_eating_window": "INTEGER",
            "carbs_g": "REAL",
            "protein_g": "REAL",
            "fat_g": "REAL",
            "macro_confidence": "TEXT NOT NULL DEFAULT ''",
        }
        for column, definition in meal_migrations.items():
            if column not in meal_columns:
                connection.execute(
                    f"ALTER TABLE meals ADD COLUMN {column} {definition}"
                )

        profile_columns = {
            row["name"] for row in connection.execute("PRAGMA table_info(profile)")
        }
        if "country" not in profile_columns:
            connection.execute(
                "ALTER TABLE profile ADD COLUMN country TEXT NOT NULL DEFAULT 'Argentine'"
            )
        if "eating_window_start" not in profile_columns:
            connection.execute(
                "ALTER TABLE profile ADD COLUMN eating_window_start TEXT NOT NULL DEFAULT '12:00'"
            )
        profile_migrations = {
            "eating_window_hours": "INTEGER NOT NULL DEFAULT 10",
            "height_cm": "REAL NOT NULL DEFAULT 170",
            "age_years": "INTEGER NOT NULL DEFAULT 56",
            "sex": "TEXT NOT NULL DEFAULT 'Homme'",
            "activity_level": "TEXT NOT NULL DEFAULT 'Modérée'",
            "target_weight_kg": "REAL NOT NULL DEFAULT 76",
            "auto_macro_targets": "INTEGER NOT NULL DEFAULT 1",
            "carbs_goal_g": "REAL NOT NULL DEFAULT 150",
            "protein_goal_g": "REAL NOT NULL DEFAULT 100",
            "fat_goal_g": "REAL NOT NULL DEFAULT 70",
            "calorie_goal_kcal": "REAL NOT NULL DEFAULT 2000",
            "meals_per_day": "INTEGER NOT NULL DEFAULT 2",
        }
        for column, definition in profile_migrations.items():
            if column not in profile_columns:
                connection.execute(
                    f"ALTER TABLE profile ADD COLUMN {column} {definition}"
                )

        favorite_food_columns = {
            row["name"]
            for row in connection.execute("PRAGMA table_info(favorite_foods)")
        }
        favorite_food_migrations = {
            "data_source": "TEXT NOT NULL DEFAULT 'saisie_manuelle'",
            "barcode": "TEXT NOT NULL DEFAULT ''",
            "source_url": "TEXT NOT NULL DEFAULT ''",
        }
        for column, definition in favorite_food_migrations.items():
            if column not in favorite_food_columns:
                connection.execute(
                    f"ALTER TABLE favorite_foods ADD COLUMN {column} {definition}"
                )

        symptom_columns = {
            row["name"]
            for row in connection.execute("PRAGMA table_info(symptom_events)")
        }
        symptom_migrations = {
            "reported_trigger": "TEXT NOT NULL DEFAULT ''",
            "source_text": "TEXT NOT NULL DEFAULT ''",
            "extraction_confidence": "TEXT NOT NULL DEFAULT 'manuelle'",
        }
        for column, definition in symptom_migrations.items():
            if column not in symptom_columns:
                connection.execute(
                    f"ALTER TABLE symptom_events ADD COLUMN {column} {definition}"
                )


def save_meal(
    description: str,
    analysis: dict[str, Any],
    product_matches: list[dict[str, Any]],
    country: str,
    eating_window_start: str,
    eating_window_hours: int,
) -> None:
    meal_type_source = (
        "interprété par l’IA · confiance "
        f"{analysis['confiance_horaire']}"
    )
    session = active_cloud_session()
    values = {
        "created_at": datetime.now().astimezone().isoformat(timespec="seconds"),
        "description": description.strip(),
        "score": analysis["note"],
        "color": analysis["couleur"],
        "comment": analysis["commentaire"],
        "model_name": analysis.get("_model_name", "IA cloud"),
        "market_country": country,
        "product_matches_json": product_matches,
        "meal_type": analysis["meal_type"],
        "meal_time": analysis["meal_time"],
        "meal_type_source": meal_type_source,
        "eating_window_start": eating_window_start,
        "eating_window_hours": normalized_eating_window_hours(eating_window_hours),
        "inside_eating_window": bool(analysis["inside_eating_window"]),
        "carbs_g": analysis["glucides_g"],
        "protein_g": analysis["proteines_g"],
        "fat_g": analysis["lipides_g"],
        "macro_confidence": analysis["confiance_macros"],
        "meal_occurred_at": analysis["meal_occurred_at"] or None,
    }
    if session is not None:
        insert_row(session, "meals", values)
        return
    with database() as connection:
        connection.execute(
            """
            INSERT INTO meals(
                created_at, description, score, color, comment, model_name,
                market_country, product_matches_json, meal_type, meal_time,
                meal_type_source, eating_window_start, eating_window_hours,
                inside_eating_window, carbs_g, protein_g, fat_g,
                macro_confidence, meal_occurred_at
            )
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                datetime.now().astimezone().isoformat(timespec="seconds"),
                description.strip(),
                analysis["note"],
                analysis["couleur"],
                analysis["commentaire"],
                analysis.get("_model_name", "IA cloud"),
                country,
                json.dumps(product_matches, ensure_ascii=False),
                analysis["meal_type"],
                analysis["meal_time"],
                meal_type_source,
                eating_window_start,
                normalized_eating_window_hours(eating_window_hours),
                int(analysis["inside_eating_window"]),
                analysis["glucides_g"],
                analysis["proteines_g"],
                analysis["lipides_g"],
                analysis["confiance_macros"],
                analysis["meal_occurred_at"],
            ),
        )


def load_meals() -> list[dict[str, Any]]:
    session = active_cloud_session()
    if session is not None:
        rows: list[Any] = select_rows(
            session,
            "meals",
            columns=(
                "id,created_at,description,score,color,comment,model_name,"
                "market_country,product_matches_json,meal_type,meal_time,"
                "meal_type_source,eating_window_start,eating_window_hours,"
                "inside_eating_window,carbs_g,protein_g,fat_g,macro_confidence,"
                "meal_occurred_at"
            ),
        )
        rows.sort(
            key=lambda row: (
                str(row.get("meal_occurred_at") or row.get("created_at") or ""),
                int(row.get("id") or 0),
            ),
            reverse=True,
        )
    else:
        with database() as connection:
            rows = connection.execute(
            """
            SELECT id, created_at, description, score, color, comment, model_name,
                   market_country, product_matches_json
                   , meal_type, meal_time, meal_type_source,
                   eating_window_start, eating_window_hours, inside_eating_window
                   , carbs_g, protein_g, fat_g, macro_confidence,
                   meal_occurred_at
            FROM meals
            ORDER BY datetime(COALESCE(NULLIF(meal_occurred_at, ''), created_at)) DESC,
                     id DESC
            """
            ).fetchall()
    meals = []
    for row in rows:
        meal = dict(row)
        # La couleur est recalculée à partir de la note afin que les anciens
        # repas suivent aussi le barème actuellement expliqué dans l'interface.
        meal["color"] = score_color(meal["score"])
        raw_matches = meal.pop("product_matches_json", [])
        if isinstance(raw_matches, list):
            meal["product_matches"] = raw_matches
        elif isinstance(raw_matches, dict):
            meal["product_matches"] = [raw_matches]
        else:
            try:
                meal["product_matches"] = json.loads(raw_matches)
            except (TypeError, json.JSONDecodeError):
                meal["product_matches"] = []
        meals.append(meal)
    return meals


def meal_occurred_at(meal: dict[str, Any]) -> datetime | None:
    """Retourne l'heure interprétée, avec repli pour les anciens repas."""
    interpreted_value = str(meal.get("meal_occurred_at") or "").strip()
    if interpreted_value:
        try:
            interpreted_at = datetime.fromisoformat(interpreted_value)
            if interpreted_at.tzinfo is None:
                interpreted_at = interpreted_at.replace(tzinfo=LOCAL_TIMEZONE)
            return interpreted_at.astimezone(LOCAL_TIMEZONE)
        except ValueError:
            pass
    try:
        recorded_at = datetime.fromisoformat(str(meal["created_at"]))
    except (KeyError, TypeError, ValueError):
        return None
    if recorded_at.tzinfo is None:
        recorded_at = recorded_at.replace(tzinfo=LOCAL_TIMEZONE)
    else:
        recorded_at = recorded_at.astimezone(LOCAL_TIMEZONE)
    meal_time_value = str(meal.get("meal_time") or "").strip()
    if meal_time_value:
        try:
            clock = time.fromisoformat(meal_time_value)
            recorded_at = recorded_at.replace(
                hour=clock.hour, minute=clock.minute, second=0, microsecond=0
            )
        except ValueError:
            pass
    return recorded_at


def find_previous_meal(
    occurred_at: datetime, meals: list[dict[str, Any]], max_hours: int = 24
) -> tuple[int | None, int | None]:
    """Associe un événement au repas précédent le plus proche, sur 24 heures."""
    best_id: int | None = None
    best_delay: int | None = None
    for meal in meals:
        meal_at = meal_occurred_at(meal)
        if meal_at is None:
            continue
        delay_minutes = int((occurred_at - meal_at).total_seconds() // 60)
        if 0 <= delay_minutes <= max_hours * 60 and (
            best_delay is None or delay_minutes < best_delay
        ):
            best_id = int(meal["id"])
            best_delay = delay_minutes
    return best_id, best_delay


def save_symptom_event(
    kind: str,
    occurred_on: date,
    occurred_time: time,
    intensity: int,
    note: str,
    meals: list[dict[str, Any]],
    pain_type: str = "",
    reported_trigger: str = "",
    source_text: str = "",
    extraction_confidence: str = "manuelle",
) -> tuple[int | None, int | None]:
    occurred_at = datetime.combine(occurred_on, occurred_time).astimezone()
    meal_id, delay_minutes = find_previous_meal(occurred_at, meals)
    session = active_cloud_session()
    if session is not None:
        insert_row(
            session,
            "symptom_events",
            {
                "occurred_at": occurred_at.isoformat(timespec="minutes"),
                "kind": kind,
                "pain_type": pain_type if kind == "douleur" else "",
                "intensity": int(intensity),
                "note": note.strip(),
                "meal_id": meal_id,
                "delay_minutes": delay_minutes,
                "created_at": datetime.now().astimezone().isoformat(timespec="seconds"),
                "reported_trigger": reported_trigger.strip(),
                "source_text": source_text.strip(),
                "extraction_confidence": extraction_confidence,
            },
        )
        return meal_id, delay_minutes
    with database() as connection:
        connection.execute(
            """
            INSERT INTO symptom_events(
                occurred_at, kind, pain_type, intensity, note, meal_id,
                delay_minutes, created_at, reported_trigger, source_text,
                extraction_confidence
            )
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                occurred_at.isoformat(timespec="minutes"),
                kind,
                pain_type if kind == "douleur" else "",
                int(intensity),
                note.strip(),
                meal_id,
                delay_minutes,
                datetime.now().astimezone().isoformat(timespec="seconds"),
                reported_trigger.strip(),
                source_text.strip(),
                extraction_confidence,
            ),
        )
    return meal_id, delay_minutes


def load_symptom_events() -> list[dict[str, Any]]:
    session = active_cloud_session()
    if session is not None:
        rows: list[Any] = select_rows(
            session,
            "symptom_events",
            columns=(
                "id,occurred_at,kind,pain_type,intensity,note,meal_id,"
                "delay_minutes,reported_trigger,source_text,extraction_confidence"
            ),
            order="occurred_at.desc,id.desc",
        )
        meals_by_id = {int(meal["id"]): meal for meal in load_meals()}
        result = []
        for row in rows:
            event = dict(row)
            linked = meals_by_id.get(int(event["meal_id"])) if event.get("meal_id") else None
            event["meal_description"] = linked.get("description") if linked else None
            event["linked_meal_type"] = linked.get("meal_type") if linked else None
            result.append(event)
        return result
    with database() as connection:
        rows = connection.execute(
            """
            SELECT symptoms.id, symptoms.occurred_at, symptoms.kind,
                   symptoms.pain_type, symptoms.intensity, symptoms.note,
                   symptoms.meal_id, symptoms.delay_minutes,
                   symptoms.reported_trigger, symptoms.source_text,
                   symptoms.extraction_confidence,
                   meals.description AS meal_description,
                   meals.meal_type AS linked_meal_type
            FROM symptom_events AS symptoms
            LEFT JOIN meals ON meals.id = symptoms.meal_id
            ORDER BY datetime(symptoms.occurred_at) DESC, symptoms.id DESC
            """
        ).fetchall()
    return [dict(row) for row in rows]


def save_bowel_event(
    occurred_on: date,
    occurred_time: time,
    bristol_type: int,
    description: str = "",
    source_text: str = "",
    extraction_confidence: str = "manuelle",
) -> None:
    """Enregistre localement un passage aux toilettes selon Bristol 1–7."""
    if bristol_type not in BRISTOL_TYPES:
        raise ValueError("Le type de Bristol doit être compris entre 1 et 7.")
    occurred_at = datetime.combine(
        occurred_on, occurred_time
    ).replace(tzinfo=LOCAL_TIMEZONE)
    session = active_cloud_session()
    if session is not None:
        insert_row(
            session,
            "bowel_events",
            {
                "occurred_at": occurred_at.isoformat(timespec="minutes"),
                "bristol_type": int(bristol_type),
                "description": description.strip(),
                "source_text": source_text.strip(),
                "extraction_confidence": extraction_confidence,
                "created_at": datetime.now(LOCAL_TIMEZONE).isoformat(timespec="seconds"),
            },
        )
        return
    with database() as connection:
        connection.execute(
            """
            INSERT INTO bowel_events(
                occurred_at, bristol_type, description, source_text,
                extraction_confidence, created_at
            )
            VALUES (?, ?, ?, ?, ?, ?)
            """,
            (
                occurred_at.isoformat(timespec="minutes"),
                int(bristol_type),
                description.strip(),
                source_text.strip(),
                extraction_confidence,
                datetime.now(LOCAL_TIMEZONE).isoformat(timespec="seconds"),
            ),
        )


def load_bowel_events() -> list[dict[str, Any]]:
    session = active_cloud_session()
    if session is not None:
        return select_rows(
            session,
            "bowel_events",
            columns=(
                "id,occurred_at,bristol_type,description,source_text,"
                "extraction_confidence,created_at"
            ),
            order="occurred_at.desc,id.desc",
        )
    with database() as connection:
        rows = connection.execute(
            """
            SELECT id, occurred_at, bristol_type, description, source_text,
                   extraction_confidence, created_at
            FROM bowel_events
            ORDER BY datetime(occurred_at) DESC, id DESC
            """
        ).fetchall()
    return [dict(row) for row in rows]


def estimate_activity_calories(
    activity_type: str,
    weight_kg: float,
    intensity: str = "Modérée",
    duration_minutes: float = 0,
    steps: int = 0,
    distance_km: float = 0,
) -> tuple[float, str]:
    """Estime prudemment une dépense sportive à partir des données disponibles."""
    weight = min(max(float(weight_kg), 20.0), 400.0)
    normalized_intensity = (
        intensity if intensity in ACTIVITY_INTENSITIES else "Modérée"
    )
    minutes = max(0.0, float(duration_minutes))
    step_count = max(0, int(steps))
    distance = max(0.0, float(distance_km))

    if activity_type == "Natation":
        if minutes <= 0:
            raise ValueError("Indiquez la durée de la séance de natation.")
        met = {"Douce": 4.8, "Modérée": 6.0, "Soutenue": 8.3}[
            normalized_intensity
        ]
        calories = met * 3.5 * weight / 200 * minutes
        method = f"Formule MET ({met:g}) · durée et poids"
    elif activity_type == "Marche":
        if step_count <= 0:
            raise ValueError("Indiquez le nombre de pas de la journée.")
        multiplier = {"Douce": 0.85, "Modérée": 1.0, "Soutenue": 1.15}[
            normalized_intensity
        ]
        calories = step_count * 0.035 * (weight / 70.0) * multiplier
        method = "Estimation par pas, poids et intensité"
    elif activity_type == "Vélo":
        if distance <= 0:
            raise ValueError("Indiquez la distance parcourue à vélo.")
        if minutes > 0:
            met = {"Douce": 4.0, "Modérée": 6.8, "Soutenue": 10.0}[
                normalized_intensity
            ]
            calories = met * 3.5 * weight / 200 * minutes
            method = f"Formule MET ({met:g}) · durée, distance et poids"
        else:
            coefficient = {"Douce": 0.22, "Modérée": 0.30, "Soutenue": 0.40}[
                normalized_intensity
            ]
            calories = distance * weight * coefficient
            method = "Estimation par distance, poids et intensité"
    else:
        raise ValueError("Type d'activité non reconnu.")

    return round(max(0.0, calories), 0), method


def save_activity_event(
    occurred_on: date,
    activity_type: str,
    weight_kg: float,
    intensity: str,
    duration_minutes: float = 0,
    steps: int = 0,
    distance_km: float = 0,
    note: str = "",
) -> float:
    calories, method = estimate_activity_calories(
        activity_type,
        weight_kg,
        intensity,
        duration_minutes,
        steps,
        distance_km,
    )
    session = active_cloud_session()
    if session is not None:
        insert_row(
            session,
            "activity_events",
            {
                "occurred_on": occurred_on.isoformat(),
                "activity_type": activity_type,
                "intensity": intensity,
                "duration_minutes": float(duration_minutes),
                "steps": int(steps),
                "distance_km": float(distance_km),
                "calories_kcal": calories,
                "estimation_method": method,
                "note": note.strip(),
                "created_at": datetime.now(LOCAL_TIMEZONE).isoformat(timespec="seconds"),
            },
        )
        return calories
    with database() as connection:
        connection.execute(
            """
            INSERT INTO activity_events(
                occurred_on, activity_type, intensity, duration_minutes, steps,
                distance_km, calories_kcal, estimation_method, note, created_at
            )
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                occurred_on.isoformat(),
                activity_type,
                intensity,
                float(duration_minutes),
                int(steps),
                float(distance_km),
                calories,
                method,
                note.strip(),
                datetime.now(LOCAL_TIMEZONE).isoformat(timespec="seconds"),
            ),
        )
    return calories


def load_activity_events() -> list[dict[str, Any]]:
    session = active_cloud_session()
    if session is not None:
        return select_rows(
            session,
            "activity_events",
            columns=(
                "id,occurred_on,activity_type,intensity,duration_minutes,steps,"
                "distance_km,calories_kcal,estimation_method,note,created_at"
            ),
            order="occurred_on.desc,id.desc",
        )
    with database() as connection:
        rows = connection.execute(
            """
            SELECT id, occurred_on, activity_type, intensity, duration_minutes,
                   steps, distance_km, calories_kcal, estimation_method, note,
                   created_at
            FROM activity_events
            ORDER BY date(occurred_on) DESC, id DESC
            """
        ).fetchall()
    return [dict(row) for row in rows]


def save_favorite_food(
    name: str,
    brand: str,
    aliases: str,
    serving_size_g: float,
    calories_100g: float,
    carbs_100g: float,
    protein_100g: float,
    fat_100g: float,
    fiber_100g: float,
    notes: str,
    country: str = DEFAULT_COUNTRY,
    food_id: int | None = None,
    data_source: str = "saisie_manuelle",
    barcode: str = "",
    source_url: str = "",
) -> int:
    """Ajoute ou met à jour une fiche alimentaire locale et réutilisable."""
    clean_name = name.strip()
    clean_brand = brand.strip()
    if not clean_name:
        raise ValueError("Indiquez le nom de l'aliment.")
    macros = [
        max(0.0, float(carbs_100g)),
        max(0.0, float(protein_100g)),
        max(0.0, float(fat_100g)),
    ]
    energy = max(0.0, float(calories_100g))
    if energy <= 0 and any(macros):
        energy = 4 * macros[0] + 4 * macros[1] + 9 * macros[2]
    now = datetime.now(LOCAL_TIMEZONE).isoformat(timespec="seconds")
    values = {
        "name": clean_name,
        "brand": clean_brand,
        "aliases": aliases.strip(),
        "serving_size_g": max(0.0, float(serving_size_g)),
        "calories_100g": round(energy, 1),
        "carbs_100g": macros[0],
        "protein_100g": macros[1],
        "fat_100g": macros[2],
        "fiber_100g": max(0.0, float(fiber_100g)),
        "notes": notes.strip(),
        "country": country.strip() or DEFAULT_COUNTRY,
        "data_source": data_source.strip() or "saisie_manuelle",
        "barcode": barcode.strip(),
        "source_url": source_url.strip(),
        "updated_at": now,
    }
    session = active_cloud_session()
    if session is not None:
        existing = None
        foods = load_favorite_foods()
        if food_id is not None:
            existing = next(
                (food for food in foods if int(food["id"]) == int(food_id)),
                None,
            )
        if existing is None:
            existing = next(
                (
                    food
                    for food in foods
                    if str(food.get("name") or "").casefold() == clean_name.casefold()
                    and str(food.get("brand") or "").casefold() == clean_brand.casefold()
                ),
                None,
            )
        if existing is not None:
            food_id = int(existing["id"])
            update_rows(
                session,
                "favorite_foods",
                values,
                filters={"id": f"eq.{food_id}"},
            )
            return food_id
        created = insert_row(session, "favorite_foods", values)
        return int(created["id"])
    with database() as connection:
        existing = None
        if food_id is not None:
            existing = connection.execute(
                "SELECT id FROM favorite_foods WHERE id = ?", (int(food_id),)
            ).fetchone()
        if existing is None:
            existing = connection.execute(
                """
                SELECT id FROM favorite_foods
                WHERE lower(name) = lower(?) AND lower(brand) = lower(?)
                """,
                (clean_name, clean_brand),
            ).fetchone()
        values = (
            clean_name,
            clean_brand,
            aliases.strip(),
            max(0.0, float(serving_size_g)),
            round(energy, 1),
            macros[0],
            macros[1],
            macros[2],
            max(0.0, float(fiber_100g)),
            notes.strip(),
            country.strip() or DEFAULT_COUNTRY,
            data_source.strip() or "saisie_manuelle",
            barcode.strip(),
            source_url.strip(),
            now,
        )
        if existing:
            food_id = int(existing["id"])
            connection.execute(
                """
                UPDATE favorite_foods
                SET name = ?, brand = ?, aliases = ?, serving_size_g = ?,
                    calories_100g = ?, carbs_100g = ?, protein_100g = ?,
                    fat_100g = ?, fiber_100g = ?, notes = ?, country = ?,
                    data_source = ?, barcode = ?, source_url = ?, updated_at = ?
                WHERE id = ?
                """,
                (*values, food_id),
            )
        else:
            cursor = connection.execute(
                """
                INSERT INTO favorite_foods(
                    name, brand, aliases, serving_size_g, calories_100g,
                    carbs_100g, protein_100g, fat_100g, fiber_100g, notes,
                    country, data_source, barcode, source_url, updated_at
                )
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                values,
            )
            food_id = int(cursor.lastrowid)
    return food_id


def load_favorite_foods() -> list[dict[str, Any]]:
    session = active_cloud_session()
    if session is not None:
        rows = select_rows(
            session,
            "favorite_foods",
            columns=(
                "id,name,brand,aliases,serving_size_g,calories_100g,carbs_100g,"
                "protein_100g,fat_100g,fiber_100g,notes,country,data_source,"
                "barcode,source_url,updated_at"
            ),
        )
        rows.sort(
            key=lambda row: (
                str(row.get("brand") or "").casefold(),
                str(row.get("name") or "").casefold(),
                int(row.get("id") or 0),
            )
        )
        return rows
    with database() as connection:
        rows = connection.execute(
            """
            SELECT id, name, brand, aliases, serving_size_g, calories_100g,
                   carbs_100g, protein_100g, fat_100g, fiber_100g, notes,
                   country, data_source, barcode, source_url, updated_at
            FROM favorite_foods
            ORDER BY lower(brand), lower(name), id
            """
        ).fetchall()
    return [dict(row) for row in rows]


def favorite_food_label(food: dict[str, Any]) -> str:
    brand = str(food.get("brand") or "").strip()
    return f"{brand} · {food['name']}" if brand else str(food["name"])


def favorite_food_as_product(food: dict[str, Any]) -> dict[str, Any]:
    """Convertit une fiche locale au format déjà utilisé par l'analyse de repas."""
    nutrition = {
        "energie_kcal_100g": round(float(food.get("calories_100g") or 0), 2),
        "glucides_100g": round(float(food.get("carbs_100g") or 0), 2),
        "fibres_100g": round(float(food.get("fiber_100g") or 0), 2),
        "proteines_100g": round(float(food.get("protein_100g") or 0), 2),
        "lipides_100g": round(float(food.get("fat_100g") or 0), 2),
    }
    return {
        "source": "bibliotheque_personnelle",
        "food_id": int(food["id"]),
        "nom": str(food["name"]),
        "marque": str(food.get("brand") or "Sans marque"),
        "quantite": "",
        "portion": (
            f"{float(food['serving_size_g']):g} g"
            if float(food.get("serving_size_g") or 0) > 0
            else ""
        ),
        "nutrition": nutrition,
        "correspondance": "personnelle vérifiée",
        "notes": str(food.get("notes") or ""),
    }


def match_favorite_foods(
    text: str,
    foods: list[dict[str, Any]],
    explicit_ids: list[int] | None = None,
) -> list[dict[str, Any]]:
    """Repère prudemment les fiches personnelles citées dans le repas."""
    haystack = normalize_search_text(text)
    selected_ids = {int(value) for value in (explicit_ids or [])}
    matches: list[dict[str, Any]] = []
    ignored = {"de", "du", "la", "le", "les", "un", "une", "type"}
    for food in foods:
        food_id = int(food["id"])
        candidates = [
            str(food.get("name") or ""),
            str(food.get("brand") or ""),
            *re.split(r"[,;|]", str(food.get("aliases") or "")),
        ]
        normalized_candidates = [
            normalize_search_text(candidate)
            for candidate in candidates
            if normalize_search_text(candidate)
        ]
        significant_tokens = {
            token
            for value in normalized_candidates
            for token in value.split()
            if len(token) >= 4 and token not in ignored
        }
        direct_match = any(
            len(candidate) >= 4 and candidate in haystack
            for candidate in normalized_candidates
        )
        token_match = bool(significant_tokens) and len(
            significant_tokens & set(haystack.split())
        ) >= min(2, len(significant_tokens))
        if food_id in selected_ids or direct_match or token_match:
            matches.append(favorite_food_as_product(food))
    return matches


def save_weight(measured_on: date, weight_kg: float, note: str) -> None:
    session = active_cloud_session()
    if session is not None:
        insert_row(
            session,
            "weights",
            {
                "measured_on": measured_on.isoformat(),
                "weight_kg": float(weight_kg),
                "note": note.strip(),
            },
        )
        return
    with database() as connection:
        connection.execute(
            "INSERT INTO weights(measured_on, weight_kg, note) VALUES (?, ?, ?)",
            (measured_on.isoformat(), float(weight_kg), note.strip()),
        )


def load_weights() -> list[dict[str, Any]]:
    session = active_cloud_session()
    if session is not None:
        return select_rows(
            session,
            "weights",
            columns="id,measured_on,weight_kg,note",
            order="measured_on.asc,id.asc",
        )
    with database() as connection:
        rows = connection.execute(
            """
            SELECT id, measured_on, weight_kg, note
            FROM weights
            ORDER BY date(measured_on) ASC, id ASC
            """
        ).fetchall()
    return [dict(row) for row in rows]


def calculate_macro_targets(
    weight_kg: float,
    height_cm: float,
    objective: str,
    age_years: int = DEFAULT_AGE_YEARS,
    sex: str = DEFAULT_SEX,
    activity_level: str = DEFAULT_ACTIVITY_LEVEL,
    target_weight_kg: float = DEFAULT_TARGET_WEIGHT_KG,
) -> dict[str, Any]:
    """Calcule des repères éducatifs personnalisés, pas une prescription."""
    weight = min(max(float(weight_kg), 20.0), 400.0)
    height = min(max(float(height_cm), 120.0), 230.0)
    age = min(max(int(age_years), 18), 100)
    normalized_sex = sex if sex in SEX_OPTIONS else DEFAULT_SEX
    normalized_activity = (
        activity_level
        if activity_level in ACTIVITY_FACTORS
        else DEFAULT_ACTIVITY_LEVEL
    )
    height_m = height / 100
    bmi = weight / (height_m**2)

    # Pour éviter que des valeurs extrêmes produisent des objectifs eux-mêmes
    # extrêmes, le poids de calcul est plafonné au poids correspondant à un
    # IMC de 30. La valeur réelle reste bien sûr conservée dans le journal.
    calculation_weight = min(weight, 30 * height_m**2)
    safety_note = ""

    # Mifflin–St Jeor estime le métabolisme de repos à partir du poids, de la
    # taille, de l'âge et du sexe. L'activité sert ensuite à approcher la
    # dépense quotidienne, qui reste une estimation à calibrer avec la tendance
    # réelle du poids sur plusieurs semaines.
    sex_adjustment = 5 if normalized_sex == "Homme" else -161
    resting_energy = (
        10 * calculation_weight + 6.25 * height - 5 * age + sex_adjustment
    )
    maintenance_energy = resting_energy * ACTIVITY_FACTORS[normalized_activity]
    energy_adjustment = 0.0

    target_weight = min(max(float(target_weight_kg), 20.0), 400.0)
    minimum_healthy_weight = 18.5 * height_m**2
    plan_phase = "Stabilisation"
    if objective == "Perdre du poids progressivement":
        if bmi < 18.5 or target_weight < minimum_healthy_weight:
            safety_note = (
                "Le déficit n'a pas été appliqué car le poids actuel ou le poids "
                "cible se situe sous un IMC de 18,5. Demandez conseil à un "
                "professionnel de santé."
            )
        elif target_weight < weight:
            distance_to_target = weight - target_weight
            if distance_to_target >= WEIGHT_STABILIZATION_TRANSITION_KG:
                energy_adjustment = -WEIGHT_LOSS_DAILY_DEFICIT_KCAL
                plan_phase = "Perte progressive"
            else:
                # Le déficit s'efface graduellement dans le dernier kilo afin
                # d'éviter un passage brutal et un dépassement du poids cible.
                transition_ratio = (
                    distance_to_target / WEIGHT_STABILIZATION_TRANSITION_KG
                )
                energy_adjustment = (
                    -WEIGHT_LOSS_DAILY_DEFICIT_KCAL * transition_ratio
                )
                plan_phase = "Transition vers la stabilisation"
    elif objective == "Prendre du poids progressivement" and target_weight > weight:
        energy_adjustment = 250.0
        plan_phase = "Prise progressive"
    elif objective != "Perdre du poids progressivement":
        plan_phase = "Repère stable"

    minimum_energy = 1500.0 if normalized_sex == "Homme" else 1200.0
    estimated_energy = min(
        max(maintenance_energy + energy_adjustment, minimum_energy), 4000
    )

    def round_to_five(value: float) -> float:
        return float(max(5, round(value / 5) * 5))

    if objective == "Perdre du poids progressivement" and energy_adjustment < 0:
        protein_g = round_to_five(calculation_weight * 1.3)
        fat_g = round_to_five(estimated_energy * 0.35 / 9)
        carbs_g = round_to_five(
            max(0, estimated_energy - protein_g * 4 - fat_g * 9) / 4
        )
    else:
        carb_share, protein_share, fat_share = 0.45, 0.20, 0.35
        carbs_g = round_to_five(estimated_energy * carb_share / 4)
        protein_g = round_to_five(estimated_energy * protein_share / 4)
        fat_g = round_to_five(estimated_energy * fat_share / 9)

    rounded_energy = int(round(estimated_energy / 50) * 50)
    carb_energy_share_pct = round(carbs_g * 4 / rounded_energy * 100)
    return {
        "carbs_g": carbs_g,
        "protein_g": protein_g,
        "fat_g": fat_g,
        "bmi": round(bmi, 1),
        "calculation_weight_kg": round(calculation_weight, 1),
        "resting_energy_kcal": int(round(resting_energy / 10) * 10),
        "maintenance_energy_kcal": int(round(maintenance_energy / 50) * 50),
        "estimated_energy_kcal": rounded_energy,
        "energy_range_low_kcal": max(
            int(round((rounded_energy - 150) / 50) * 50), int(minimum_energy)
        ),
        "energy_range_high_kcal": int(round((rounded_energy + 150) / 50) * 50),
        "daily_adjustment_kcal": int(round(energy_adjustment)),
        "plan_phase": plan_phase,
        "carb_energy_share_pct": carb_energy_share_pct,
        "balanced_carb_share_range": BALANCED_CARB_SHARE_RANGE,
        "activity_factor": ACTIVITY_FACTORS[normalized_activity],
        "target_bmi": round(target_weight / (height_m**2), 1),
        "safety_note": safety_note,
    }


def default_profile() -> dict[str, Any]:
    return {
        "display_name": "",
        "objective": "Stabiliser ma glycémie",
        "country": DEFAULT_COUNTRY,
        "eating_window_start": DEFAULT_EATING_WINDOW_START,
        "eating_window_hours": DEFAULT_EATING_WINDOW_HOURS,
        "height_cm": DEFAULT_HEIGHT_CM,
        "age_years": DEFAULT_AGE_YEARS,
        "sex": DEFAULT_SEX,
        "activity_level": DEFAULT_ACTIVITY_LEVEL,
        "target_weight_kg": DEFAULT_TARGET_WEIGHT_KG,
        "auto_macro_targets": 1,
        "carbs_goal_g": DEFAULT_CARBS_GOAL_G,
        "protein_goal_g": DEFAULT_PROTEIN_GOAL_G,
        "fat_goal_g": DEFAULT_FAT_GOAL_G,
        "calorie_goal_kcal": DEFAULT_CALORIE_GOAL_KCAL,
        "meals_per_day": DEFAULT_MEALS_PER_DAY,
    }


def load_profile() -> dict[str, Any]:
    session = active_cloud_session()
    if session is not None:
        rows = select_rows(
            session,
            "profile",
            columns=(
                "display_name,objective,country,eating_window_start,"
                "eating_window_hours,height_cm,age_years,sex,activity_level,"
                "target_weight_kg,auto_macro_targets,carbs_goal_g,"
                "protein_goal_g,fat_goal_g,calorie_goal_kcal,meals_per_day"
            ),
            limit=1,
        )
        if rows:
            return rows[0]
        return default_profile()
    with database() as connection:
        row = connection.execute(
            """
            SELECT display_name, objective, country, eating_window_start,
                   eating_window_hours, height_cm, age_years, sex,
                   activity_level, target_weight_kg,
                   auto_macro_targets, carbs_goal_g, protein_goal_g, fat_goal_g,
                   calorie_goal_kcal, meals_per_day
            FROM profile WHERE id = 1
            """
        ).fetchone()
    if row is None:
        return default_profile()
    return dict(row)


def save_profile(
    display_name: str,
    objective: str,
    country: str,
    eating_window_start: time,
    eating_window_hours: int,
    height_cm: float,
    age_years: int,
    sex: str,
    activity_level: str,
    target_weight_kg: float,
    auto_macro_targets: bool,
    carbs_goal_g: float,
    protein_goal_g: float,
    fat_goal_g: float,
    calorie_goal_kcal: float,
    meals_per_day: int,
) -> None:
    session = active_cloud_session()
    values = {
        "display_name": display_name.strip(),
        "objective": objective,
        "country": country.strip() or DEFAULT_COUNTRY,
        "eating_window_start": eating_window_start.strftime("%H:%M"),
        "eating_window_hours": normalized_eating_window_hours(eating_window_hours),
        "height_cm": float(height_cm),
        "age_years": int(age_years),
        "sex": sex if sex in SEX_OPTIONS else DEFAULT_SEX,
        "activity_level": (
            activity_level if activity_level in ACTIVITY_FACTORS else DEFAULT_ACTIVITY_LEVEL
        ),
        "target_weight_kg": float(target_weight_kg),
        "auto_macro_targets": bool(auto_macro_targets),
        "carbs_goal_g": float(carbs_goal_g),
        "protein_goal_g": float(protein_goal_g),
        "fat_goal_g": float(fat_goal_g),
        "calorie_goal_kcal": float(calorie_goal_kcal),
        "meals_per_day": int(meals_per_day),
        "updated_at": datetime.now().astimezone().isoformat(timespec="seconds"),
    }
    if session is not None:
        values["user_id"] = session.user_id
        upsert_row(
            session,
            "profile",
            values,
            conflict_column="user_id",
        )
        return
    with database() as connection:
        connection.execute(
            """
            INSERT INTO profile(
                id, display_name, objective, country, eating_window_start,
                eating_window_hours, height_cm, age_years, sex,
                activity_level, target_weight_kg,
                auto_macro_targets, carbs_goal_g, protein_goal_g, fat_goal_g,
                calorie_goal_kcal, meals_per_day, updated_at
            )
            VALUES (1, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            ON CONFLICT(id) DO UPDATE SET
                display_name = excluded.display_name,
                objective = excluded.objective,
                country = excluded.country,
                eating_window_start = excluded.eating_window_start,
                eating_window_hours = excluded.eating_window_hours,
                height_cm = excluded.height_cm,
                age_years = excluded.age_years,
                sex = excluded.sex,
                activity_level = excluded.activity_level,
                target_weight_kg = excluded.target_weight_kg,
                auto_macro_targets = excluded.auto_macro_targets,
                carbs_goal_g = excluded.carbs_goal_g,
                protein_goal_g = excluded.protein_goal_g,
                fat_goal_g = excluded.fat_goal_g,
                calorie_goal_kcal = excluded.calorie_goal_kcal,
                meals_per_day = excluded.meals_per_day,
                updated_at = excluded.updated_at
            """,
            (
                display_name.strip(),
                objective,
                country.strip() or DEFAULT_COUNTRY,
                eating_window_start.strftime("%H:%M"),
                normalized_eating_window_hours(eating_window_hours),
                float(height_cm),
                int(age_years),
                sex if sex in SEX_OPTIONS else DEFAULT_SEX,
                (
                    activity_level
                    if activity_level in ACTIVITY_FACTORS
                    else DEFAULT_ACTIVITY_LEVEL
                ),
                float(target_weight_kg),
                int(auto_macro_targets),
                float(carbs_goal_g),
                float(protein_goal_g),
                float(fat_goal_g),
                float(calorie_goal_kcal),
                int(meals_per_day),
                datetime.now().astimezone().isoformat(timespec="seconds"),
            ),
        )


def extract_json(raw_text: str) -> dict[str, Any]:
    """Décode un objet JSON, même si le modèle ajoute des balises accidentelles."""
    text = raw_text.strip()
    text = re.sub(r"^```(?:json)?\s*", "", text, flags=re.IGNORECASE)
    text = re.sub(r"\s*```$", "", text)
    try:
        value = json.loads(text)
    except json.JSONDecodeError:
        start, end = text.find("{"), text.rfind("}")
        if start < 0 or end <= start:
            raise ValueError("L'IA n'a pas renvoyé d'objet JSON exploitable.")
        try:
            value = json.loads(text[start : end + 1])
        except json.JSONDecodeError as error:
            raise ValueError("Le JSON renvoyé par l'IA est invalide.") from error
    if not isinstance(value, dict):
        raise ValueError("La réponse de l'IA doit être un objet JSON.")
    return value


def normalize_meal_analysis(payload: dict[str, Any]) -> dict[str, Any]:
    if isinstance(payload.get("note"), bool):
        raise ValueError("La note IA doit être un entier entre 0 et 10.")
    try:
        numeric_score = float(payload["note"])
    except (KeyError, TypeError, ValueError) as error:
        raise ValueError("La réponse ne contient pas de note valide.") from error

    if not numeric_score.is_integer() or not 0 <= numeric_score <= 10:
        raise ValueError("La note IA doit être un entier entre 0 et 10.")
    score = int(numeric_score)

    expected_color = score_color(score)
    color = str(payload.get("couleur", "")).strip().lower()
    aliases = {
        "green": "vert",
        "verte": "vert",
        "amber": "orange",
        "red": "rouge",
    }
    color = aliases.get(color, color)
    # La règle note/couleur reste cohérente même si le modèle se contredit.
    if color != expected_color:
        color = expected_color

    comment = str(payload.get("commentaire", "")).strip()
    if not comment:
        raise ValueError("La réponse ne contient pas de commentaire.")

    macros = {}
    for field in ("glucides_g", "proteines_g", "lipides_g"):
        value = payload.get(field)
        if isinstance(value, bool):
            raise ValueError("Une estimation de macronutriment est invalide.")
        try:
            numeric_value = float(value)
        except (TypeError, ValueError) as error:
            raise ValueError("Une estimation de macronutriment est absente.") from error
        if not 0 <= numeric_value <= 1000:
            raise ValueError("Une estimation de macronutriment est hors limites.")
        macros[field] = round(numeric_value, 1)

    confidence = str(payload.get("confiance_macros", "")).strip().lower()
    confidence = {"elevee": "élevée", "haute": "élevée"}.get(confidence, confidence)
    if confidence not in {"faible", "moyenne", "élevée"}:
        raise ValueError("Le niveau de confiance des macronutriments est invalide.")

    meal_type = str(payload.get("type_repas_detecte", "")).strip()
    if meal_type not in MEAL_TYPES:
        raise ValueError("Le type de repas interprété est invalide.")

    timing_confidence = str(payload.get("confiance_horaire", "")).strip().lower()
    timing_confidence = {
        "elevee": "élevée",
        "haute": "élevée",
    }.get(timing_confidence, timing_confidence)
    if timing_confidence not in {"faible", "moyenne", "élevée"}:
        raise ValueError("Le niveau de confiance de l'horaire est invalide.")

    meal_date_value = str(payload.get("repas_date", "")).strip()
    meal_time_value = str(payload.get("repas_heure", "")).strip()
    needs_clarification = payload.get("requires_timing_clarification") is True
    clarification_question = str(
        payload.get("timing_clarification_question", "")
    ).strip()
    interpreted_at: datetime | None = None
    try:
        parsed_date = date.fromisoformat(meal_date_value)
        parsed_time = time.fromisoformat(meal_time_value)
        interpreted_at = datetime.combine(
            parsed_date,
            parsed_time.replace(second=0, microsecond=0),
            tzinfo=LOCAL_TIMEZONE,
        )
        meal_date_value = parsed_date.isoformat()
        meal_time_value = parsed_time.strftime("%H:%M")
    except ValueError:
        needs_clarification = True

    if timing_confidence == "faible":
        needs_clarification = True
    if interpreted_at and interpreted_at > datetime.now(LOCAL_TIMEZONE) + timedelta(minutes=5):
        needs_clarification = True
        clarification_question = (
            "La date ou l'heure semble se situer dans le futur. Pouvez-vous la préciser ?"
        )
    if needs_clarification and not clarification_question:
        clarification_question = (
            "Quel jour et à quelle heure exacte avez-vous pris ce repas ?"
        )

    return {
        "note": score,
        "couleur": color,
        "commentaire": comment,
        **macros,
        "confiance_macros": confidence,
        "meal_type": meal_type,
        "meal_date": meal_date_value,
        "meal_time": meal_time_value,
        "meal_occurred_at": (
            interpreted_at.isoformat(timespec="minutes") if interpreted_at else ""
        ),
        "confiance_horaire": timing_confidence,
        "requires_timing_clarification": needs_clarification,
        "timing_clarification_question": clarification_question,
    }


def normalize_report(payload: dict[str, Any]) -> dict[str, Any]:
    summary = str(payload.get("resume", "")).strip()
    encouragement = str(payload.get("encouragement", "")).strip()
    trend = str(payload.get("tendance", "")).strip().lower()
    tips = payload.get("conseils")

    if not summary or not encouragement:
        raise ValueError("Le bilan IA est incomplet.")
    if trend not in {"amelioration", "stable", "a_surveille"}:
        raise ValueError("La tendance du bilan IA est invalide.")
    if not isinstance(tips, list) or len(tips) != 3:
        raise ValueError("Le bilan doit contenir exactement trois conseils.")
    clean_tips = [str(tip).strip() for tip in tips]
    if any(not tip for tip in clean_tips):
        raise ValueError("Un des conseils IA est vide.")
    return {
        "resume": summary,
        "tendance": trend,
        "conseils": clean_tips,
        "encouragement": encouragement,
    }


def call_deepseek(
    api_key: str,
    system_prompt: str,
    user_prompt: str,
    response_schema: dict[str, Any],
    schema_name: str,
) -> dict[str, Any]:
    """Appelle DeepSeek en JSON, avec une relance sûre si sa réponse est vide."""
    schema_instruction = (
        "\n\nRéponds uniquement avec un objet JSON valide nommé "
        f"{schema_name}, conforme exactement à ce JSON Schema :\n"
        + json.dumps(response_schema, ensure_ascii=False)
    )
    with httpx.Client(
        base_url="https://api.deepseek.com",
        headers={
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
        },
        timeout=45.0,
    ) as client:
        last_json_error: ValueError | None = None
        last_finish_reason = ""
        for attempt in range(2):
            retry_instruction = (
                "\n\nIMPORTANT : la tentative précédente n'a pas fourni de JSON. "
                "Produis maintenant immédiatement l'objet JSON complet, sans "
                "raisonnement, sans Markdown et sans aucun texte additionnel."
                if attempt
                else ""
            )
            request_body: dict[str, Any] = {
                "model": DEEPSEEK_MODEL_NAME,
                "messages": [
                    {
                        "role": "system",
                        "content": (
                            system_prompt + schema_instruction + retry_instruction
                        ),
                    },
                    {"role": "user", "content": user_prompt},
                ],
                "thinking": {"type": "disabled"},
                "reasoning_effort": "none",
                "max_tokens": 2400,
                "stream": False,
            }
            # Le mode JSON natif est prioritaire. DeepSeek documente toutefois
            # de rares réponses vides avec ce mode : la seconde tentative garde
            # le schéma strict dans le prompt mais évite ce défaut de transport.
            if attempt == 0:
                request_body["response_format"] = {"type": "json_object"}

            response = client.post("/chat/completions", json=request_body)
            response.raise_for_status()
            data = response.json()
            try:
                choice = data["choices"][0]
                message = choice["message"]
                response_text = message.get("content")
                last_finish_reason = str(choice.get("finish_reason", ""))
            except (KeyError, IndexError, TypeError) as error:
                if attempt == 0:
                    continue
                raise RuntimeError(
                    "DeepSeek a renvoyé une réponse incomplète après deux tentatives."
                ) from error

            if not isinstance(response_text, str) or not response_text.strip():
                continue
            try:
                return extract_json(response_text)
            except ValueError as error:
                last_json_error = error

    if last_json_error is not None:
        raise RuntimeError(
            "DeepSeek a renvoyé un JSON invalide après deux tentatives."
        ) from last_json_error
    if last_finish_reason == "length":
        raise RuntimeError(
            "DeepSeek a atteint sa limite de réponse avant de produire le JSON."
        )
    raise RuntimeError(
        "DeepSeek a renvoyé une réponse vide après deux tentatives."
    )


def call_groq(
    api_key: str,
    system_prompt: str,
    user_prompt: str,
    response_schema: dict[str, Any],
    schema_name: str,
) -> dict[str, Any]:
    """Appelle Groq en mode JSON Schema strict et renvoie l'objet décodé."""
    client = Groq(api_key=api_key, timeout=45.0, max_retries=2)
    try:
        response = client.chat.completions.create(
            model=GROQ_MODEL_NAME,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            response_format={
                "type": "json_schema",
                "json_schema": {
                    "name": schema_name,
                    "strict": True,
                    "schema": response_schema,
                },
            },
            reasoning_effort="none",
            temperature=0.2,
            max_tokens=1200,
        )
    finally:
        client.close()

    response_text = response.choices[0].message.content if response.choices else None
    if not response_text:
        raise RuntimeError("Groq n'a pas produit de réponse exploitable.")
    return extract_json(response_text)


def call_gemini(
    api_key: str,
    system_prompt: str,
    user_prompt: str,
    response_schema: dict[str, Any],
    schema_name: str,
) -> dict[str, Any]:
    """Appelle le SDK Google Gen AI avec une sortie JSON Schema stricte."""
    del schema_name  # Gemini reçoit directement le schéma, sans nom d'enveloppe.
    client = genai.Client(
        api_key=api_key,
        http_options=types.HttpOptions(timeout=45_000),
    )
    try:
        response = client.models.generate_content(
            model=GEMINI_MODEL_NAME,
            contents=user_prompt,
            config=types.GenerateContentConfig(
                system_instruction=system_prompt,
                temperature=0.2,
                max_output_tokens=1200,
                response_mime_type="application/json",
                response_json_schema=response_schema,
            ),
        )
    finally:
        client.close()

    response_text = response.text
    if not response_text:
        raise RuntimeError("Gemini n'a pas produit de réponse exploitable.")
    return extract_json(response_text)


def call_ai(
    settings: AISettings,
    system_prompt: str,
    user_prompt: str,
    response_schema: dict[str, Any],
    schema_name: str,
) -> tuple[dict[str, Any], str, str]:
    """Essaie DeepSeek, puis Gemini et enfin Groq en cas d'indisponibilité."""
    failures: list[tuple[str, Exception]] = []
    candidates = (
        (
            "DeepSeek",
            settings.deepseek_api_key,
            DEEPSEEK_MODEL_NAME,
            call_deepseek,
        ),
        (
            "Google Gemini",
            settings.gemini_api_key,
            GEMINI_MODEL_NAME,
            call_gemini,
        ),
        ("Groq", settings.groq_api_key, GROQ_MODEL_NAME, call_groq),
    )
    for provider, api_key, model_name, caller in candidates:
        if not api_key:
            continue
        try:
            payload = caller(
                api_key,
                system_prompt,
                user_prompt,
                response_schema,
                schema_name,
            )
            return payload, provider, model_name
        except Exception as error:
            failures.append((provider, error))

    if not failures:
        raise ValueError("Aucune clé API IA n'est configurée.")
    raise AIProvidersUnavailable(failures)


def test_deepseek_connection(api_key: str) -> None:
    """Vérifie la clé avec une petite génération JSON de bout en bout."""
    result = call_deepseek(
        api_key,
        (
            "Tu testes une connexion API. Réponds uniquement en JSON. "
            'Exemple exact : {"ok": true}'
        ),
        'Retourne exactement {"ok": true}.',
        {
            "type": "object",
            "properties": {"ok": {"type": "boolean"}},
            "required": ["ok"],
            "additionalProperties": False,
        },
        "connection_test",
    )
    if result.get("ok") is not True:
        raise RuntimeError("DeepSeek n'a pas validé le test JSON.")


def test_gemini_connection(api_key: str) -> None:
    """Vérifie la clé Gemini avec un véritable appel très court."""
    client = genai.Client(
        api_key=api_key,
        http_options=types.HttpOptions(timeout=30_000),
    )
    try:
        response = client.models.generate_content(
            model=GEMINI_MODEL_NAME,
            contents="Réponds uniquement par OK.",
            config=types.GenerateContentConfig(
                temperature=0,
                max_output_tokens=32,
            ),
        )
        if not response.text:
            raise RuntimeError("Gemini n'a pas produit de réponse exploitable.")
    finally:
        client.close()


def test_groq_connection(api_key: str) -> None:
    """Vérifie la clé Groq et un véritable appel au modèle."""
    client = Groq(api_key=api_key, timeout=30.0, max_retries=1)
    try:
        client.chat.completions.create(
            model=GROQ_MODEL_NAME,
            messages=[{"role": "user", "content": "Réponds uniquement par OK."}],
            reasoning_effort="none",
            max_tokens=8,
        )
    finally:
        client.close()


def normalize_intake_extraction(payload: dict[str, Any]) -> dict[str, Any]:
    raw_events = payload.get("events")
    if not isinstance(raw_events, list):
        raise ValueError("La réponse IA ne contient pas une liste d'événements.")

    now = datetime.now().astimezone()
    normalized_events: list[dict[str, Any]] = []
    for raw_event in raw_events:
        if not isinstance(raw_event, dict):
            continue
        kind = str(raw_event.get("kind", "")).strip().lower()
        if kind not in {"faim", "douleur"}:
            continue

        issues: list[str] = []
        try:
            intensity = int(raw_event.get("intensity", 0))
        except (TypeError, ValueError):
            intensity = 0
        if not 1 <= intensity <= 10:
            issues.append("l'intensité sur 10")

        pain_type = str(raw_event.get("pain_type", "")).strip()
        if kind == "faim":
            pain_type = ""
        elif pain_type not in PAIN_TYPES:
            issues.append("le type de douleur")

        occurred_at_text = str(raw_event.get("occurred_at", "")).strip()
        occurred_at: datetime | None = None
        if occurred_at_text:
            try:
                occurred_at = datetime.fromisoformat(
                    occurred_at_text.replace("Z", "+00:00")
                )
                if occurred_at.tzinfo is None:
                    occurred_at = occurred_at.astimezone()
                else:
                    occurred_at = occurred_at.astimezone()
                if occurred_at > now.replace(microsecond=0):
                    issues.append("une date/heure passée")
            except (TypeError, ValueError):
                issues.append("la date et l'heure exactes")
        else:
            issues.append("la date et l'heure exactes")

        confidence = str(raw_event.get("confidence", "moyenne")).strip().lower()
        confidence = {"elevee": "élevée"}.get(confidence, confidence)
        if confidence not in {"faible", "moyenne", "élevée"}:
            confidence = "faible"

        requires_clarification = (
            bool(raw_event.get("requires_clarification"))
            or bool(issues)
            or confidence == "faible"
        )
        question = str(raw_event.get("clarification_question", "")).strip()
        if requires_clarification and not question:
            if issues:
                question = (
                    "Pouvez-vous préciser "
                    + ", ".join(dict.fromkeys(issues))
                    + " ?"
                )
            else:
                question = (
                    "Pouvez-vous reformuler la date, l'heure, le type et "
                    "l'intensité pour confirmer cet événement ?"
                )

        normalized_events.append(
            {
                "kind": kind,
                "occurred_at": occurred_at.isoformat(timespec="minutes")
                if occurred_at
                else "",
                "pain_type": pain_type,
                "intensity": intensity,
                "description": str(raw_event.get("description", "")).strip(),
                "reported_trigger": str(
                    raw_event.get("declencheur_rapporte", "")
                ).strip(),
                "confidence": confidence,
                "requires_clarification": requires_clarification,
                "clarification_question": question,
            }
        )
    contains_meal = payload.get("contains_meal") is True
    meal_description = str(payload.get("meal_description", "")).strip()
    if contains_meal and not meal_description:
        contains_meal = False
    if not contains_meal:
        meal_description = ""
    return {
        "events": normalized_events,
        "contains_meal": contains_meal,
        "meal_description": meal_description,
    }


def normalize_symptom_extraction(payload: dict[str, Any]) -> list[dict[str, Any]]:
    """Compatibilité pour le formulaire dédié aux symptômes."""
    return normalize_intake_extraction(payload)["events"]


def parse_intake_text(settings: AISettings, text: str) -> dict[str, Any]:
    reference_time = datetime.now().astimezone().isoformat(timespec="minutes")
    prompt = (
        f"DATE ET HEURE LOCALES DE RÉFÉRENCE : {reference_time}\n"
        "FUSEAU : America/Argentina/Buenos_Aires\n"
        f"CATÉGORIES DE DOULEUR AUTORISÉES : {json.dumps(PAIN_TYPES, ensure_ascii=False)}\n\n"
        f"TEXTE À INTERPRÉTER :\n{text.strip()}"
    )
    payload, provider, model_name = call_ai(
        settings,
        SYMPTOM_SYSTEM_PROMPT,
        prompt,
        SYMPTOM_RESPONSE_SCHEMA,
        "symptom_extraction",
    )
    result = normalize_intake_extraction(payload)
    result["_provider"] = provider
    result["_model_name"] = model_name
    return result


def parse_symptom_text(settings: AISettings, text: str) -> dict[str, Any]:
    return parse_intake_text(settings, text)


def text_may_contain_symptom(text: str) -> bool:
    """Repère localement les signalements à envoyer à l'extracteur dédié."""
    normalized = f" {normalize_search_text(text)} "
    symptom_terms = (
        " douleur ", " douloureux ", " mal au ventre ", " mal a l estomac ",
        " inconfort ", " brulure ", " crampe ", " noeud ", " ballonnement ",
        " nausee ", " faim ", " affame ", " estomac ", " stomach ",
        " sternum ", " nombril ", " colon ",
    )
    return any(term in normalized for term in symptom_terms)


def save_extracted_symptom(
    event: dict[str, Any], source_text: str, meals: list[dict[str, Any]]
) -> tuple[int | None, int | None]:
    occurred_at = datetime.fromisoformat(event["occurred_at"]).astimezone()
    return save_symptom_event(
        event["kind"],
        occurred_at.date(),
        occurred_at.time().replace(tzinfo=None),
        int(event["intensity"]),
        event["description"],
        meals,
        event["pain_type"],
        event["reported_trigger"],
        source_text,
        event["confidence"],
    )


def save_ready_symptom_events(
    events: list[dict[str, Any]], source_text: str, meals: list[dict[str, Any]]
) -> tuple[int, list[str]]:
    """Enregistre les événements complets et isole ceux à clarifier."""
    saved_count = 0
    questions: list[str] = []
    for event in events:
        if event["requires_clarification"]:
            questions.append(event["clarification_question"])
            continue
        save_extracted_symptom(event, source_text, meals)
        saved_count += 1
    return saved_count, list(dict.fromkeys(questions))


def normalize_bowel_extraction(payload: dict[str, Any]) -> list[dict[str, Any]]:
    raw_events = payload.get("events")
    if not isinstance(raw_events, list):
        raise ValueError("La réponse IA ne contient pas de liste de selles.")

    now = datetime.now(LOCAL_TIMEZONE).replace(microsecond=0)
    normalized_events: list[dict[str, Any]] = []
    for raw_event in raw_events:
        if not isinstance(raw_event, dict):
            continue
        issues: list[str] = []
        try:
            bristol_type = int(raw_event.get("bristol_type", 0))
        except (TypeError, ValueError):
            bristol_type = 0
        if bristol_type not in BRISTOL_TYPES:
            issues.append("la forme ou le type de Bristol (1 à 7)")

        occurred_at_text = str(raw_event.get("occurred_at", "")).strip()
        occurred_at: datetime | None = None
        if occurred_at_text:
            try:
                occurred_at = datetime.fromisoformat(
                    occurred_at_text.replace("Z", "+00:00")
                )
                if occurred_at.tzinfo is None:
                    occurred_at = occurred_at.replace(tzinfo=LOCAL_TIMEZONE)
                else:
                    occurred_at = occurred_at.astimezone(LOCAL_TIMEZONE)
                if occurred_at > now:
                    issues.append("une date et une heure passées")
            except (TypeError, ValueError):
                issues.append("la date et l'heure exactes")
        else:
            issues.append("la date et l'heure exactes")

        confidence = str(raw_event.get("confidence", "moyenne")).strip().lower()
        confidence = {"elevee": "élevée"}.get(confidence, confidence)
        if confidence not in {"faible", "moyenne", "élevée"}:
            confidence = "faible"
        requires_clarification = (
            bool(raw_event.get("requires_clarification"))
            or bool(issues)
            or confidence == "faible"
        )
        question = str(raw_event.get("clarification_question", "")).strip()
        if requires_clarification and not question:
            question = "Pouvez-vous préciser " + ", ".join(
                dict.fromkeys(issues or ["le moment et la forme"])
            ) + " ?"

        normalized_events.append(
            {
                "occurred_at": (
                    occurred_at.isoformat(timespec="minutes")
                    if occurred_at
                    else ""
                ),
                "bristol_type": bristol_type,
                "description": str(raw_event.get("description", "")).strip(),
                "confidence": confidence,
                "requires_clarification": requires_clarification,
                "clarification_question": question,
            }
        )
    return normalized_events


def parse_bowel_text(settings: AISettings, text: str) -> dict[str, Any]:
    reference_time = datetime.now(LOCAL_TIMEZONE).isoformat(timespec="minutes")
    prompt = (
        f"DATE ET HEURE LOCALES DE RÉFÉRENCE : {reference_time}\n"
        "FUSEAU : America/Argentina/Buenos_Aires\n"
        "ÉCHELLE DE BRISTOL : "
        f"{json.dumps(BRISTOL_TYPES, ensure_ascii=False)}\n\n"
        f"TEXTE À INTERPRÉTER :\n{text.strip()}"
    )
    payload, provider, model_name = call_ai(
        settings,
        BOWEL_SYSTEM_PROMPT,
        prompt,
        BOWEL_RESPONSE_SCHEMA,
        "bowel_extraction",
    )
    return {
        "events": normalize_bowel_extraction(payload),
        "_provider": provider,
        "_model_name": model_name,
    }


def save_ready_bowel_events(
    events: list[dict[str, Any]], source_text: str
) -> tuple[int, list[str]]:
    saved_count = 0
    questions: list[str] = []
    for event in events:
        if event["requires_clarification"]:
            questions.append(event["clarification_question"])
            continue
        occurred_at = datetime.fromisoformat(event["occurred_at"]).astimezone(
            LOCAL_TIMEZONE
        )
        save_bowel_event(
            occurred_at.date(),
            occurred_at.time().replace(tzinfo=None),
            int(event["bristol_type"]),
            event["description"],
            source_text,
            event["confidence"],
        )
        saved_count += 1
    return saved_count, list(dict.fromkeys(questions))


def normalize_food_label(payload: dict[str, Any]) -> dict[str, Any]:
    """Valide une étiquette extraite par l'IA sans compléter les valeurs absentes."""
    limits = {
        "portion_reference_g": 5000.0,
        "calories_100g": 1000.0,
        "glucides_100g": 100.0,
        "proteines_100g": 100.0,
        "lipides_100g": 100.0,
        "fibres_100g": 100.0,
    }
    values: dict[str, float] = {}
    for field, maximum in limits.items():
        raw_value = payload.get(field, 0)
        if isinstance(raw_value, bool):
            raise ValueError("Une valeur nutritionnelle extraite est invalide.")
        try:
            numeric_value = float(raw_value)
        except (TypeError, ValueError) as error:
            raise ValueError("Une valeur nutritionnelle extraite est invalide.") from error
        if not 0 <= numeric_value <= maximum:
            raise ValueError("Une valeur nutritionnelle extraite est hors limites.")
        values[field] = round(numeric_value, 2)

    confidence = str(payload.get("confiance", "faible")).strip().lower()
    confidence = {"elevee": "élevée"}.get(confidence, confidence)
    if confidence not in {"faible", "moyenne", "élevée"}:
        confidence = "faible"
    missing = payload.get("champs_manquants", [])
    if not isinstance(missing, list):
        missing = []
    return {
        "source": "etiquette_interpretee_ia",
        "nom": str(payload.get("nom", "")).strip() or "Produit à nommer",
        "marque": str(payload.get("marque", "")).strip(),
        "portion_g": values["portion_reference_g"],
        "nutrition": {
            "energie_kcal_100g": values["calories_100g"],
            "glucides_100g": values["glucides_100g"],
            "proteines_100g": values["proteines_100g"],
            "lipides_100g": values["lipides_100g"],
            "fibres_100g": values["fibres_100g"],
        },
        "donnees_explicites": payload.get("donnees_explicites") is True,
        "correspondance": confidence,
        "champs_manquants": [str(item).strip() for item in missing if str(item).strip()],
        "note": str(payload.get("note", "")).strip(),
        "code_barres": "",
        "url_source": "",
    }


def parse_food_label(settings: AISettings, label_text: str) -> dict[str, Any]:
    prompt = (
        "MARCHÉ : Argentine\n"
        "Convertis les valeurs explicites en valeurs pour 100 g.\n\n"
        f"TEXTE DE L'ÉTIQUETTE :\n{label_text.strip()}"
    )
    payload, provider, model_name = call_ai(
        settings,
        FOOD_LABEL_SYSTEM_PROMPT,
        prompt,
        FOOD_LABEL_RESPONSE_SCHEMA,
        "food_label_extraction",
    )
    result = normalize_food_label(payload)
    result["_provider"] = provider
    result["_model_name"] = model_name
    return result


def grams_from_product_text(value: str) -> float:
    """Extrait une portion en grammes d'un libellé simple comme « 190 g »."""
    match = re.search(r"(\d+(?:[.,]\d+)?)\s*g\b", str(value), re.IGNORECASE)
    if not match:
        return 0.0
    return float(match.group(1).replace(",", "."))


def food_source_label(source: str) -> str:
    return {
        "open_food_facts": "Open Food Facts",
        "etiquette_interpretee_ia": "Étiquette interprétée par l'IA",
        "saisie_manuelle": "Saisie manuelle",
    }.get(str(source), "Saisie manuelle")


def apply_food_candidate_to_form(candidate: dict[str, Any]) -> None:
    """Préremplit la nouvelle fiche avant son rendu au prochain rerun."""
    nutrition = candidate.get("nutrition") or {}
    portion_g = float(candidate.get("portion_g") or 0)
    if portion_g <= 0:
        portion_g = grams_from_product_text(str(candidate.get("portion") or ""))
    st.session_state["favorite_food_choice"] = 0
    st.session_state["favorite_name_0"] = str(candidate.get("nom") or "")
    st.session_state["favorite_brand_0"] = str(candidate.get("marque") or "")
    st.session_state["favorite_aliases_0"] = str(candidate.get("requete") or "")
    st.session_state["favorite_calories_0"] = float(
        nutrition.get("energie_kcal_100g") or 0
    )
    st.session_state["favorite_carbs_0"] = float(
        nutrition.get("glucides_100g") or 0
    )
    st.session_state["favorite_protein_0"] = float(
        nutrition.get("proteines_100g") or 0
    )
    st.session_state["favorite_fat_0"] = float(
        nutrition.get("lipides_100g") or 0
    )
    st.session_state["favorite_fiber_0"] = float(
        nutrition.get("fibres_100g") or 0
    )
    st.session_state["favorite_serving_0"] = portion_g
    source = str(candidate.get("source") or "saisie_manuelle")
    notes = str(candidate.get("note") or "").strip()
    if source == "open_food_facts":
        notes = "Valeurs importées depuis Open Food Facts ; vérifier l'étiquette."
    st.session_state["favorite_notes_0"] = notes
    st.session_state["favorite_data_source_0"] = source
    st.session_state["favorite_barcode_0"] = str(candidate.get("code_barres") or "")
    st.session_state["favorite_source_url_0"] = str(candidate.get("url_source") or "")
    st.session_state["favorite_food_prefilled"] = True


def normalize_search_text(value: str) -> str:
    """Normalise un nom de marque pour comparer deux libellés sans accents."""
    decomposed = unicodedata.normalize("NFKD", value.casefold())
    without_accents = "".join(char for char in decomposed if not unicodedata.combining(char))
    return re.sub(r"[^a-z0-9]+", " ", without_accents).strip()


def parse_clock(value: str, default: str = DEFAULT_EATING_WINDOW_START) -> time:
    try:
        return time.fromisoformat(value)
    except (TypeError, ValueError):
        return time.fromisoformat(default)


def minutes_since_midnight(value: time) -> int:
    return value.hour * 60 + value.minute


def normalized_eating_window_hours(value: Any) -> int:
    """Ramène la durée à une valeur sûre et compréhensible dans l'interface."""
    try:
        hours = int(value)
    except (TypeError, ValueError):
        return DEFAULT_EATING_WINDOW_HOURS
    return hours if hours in EATING_WINDOW_HOUR_OPTIONS else DEFAULT_EATING_WINDOW_HOURS


def fasting_window_hours(eating_window_hours: int) -> int:
    return 24 - normalized_eating_window_hours(eating_window_hours)


def eating_window_pattern(eating_window_hours: int) -> str:
    eating_hours = normalized_eating_window_hours(eating_window_hours)
    return f"{eating_hours} h d’alimentation / {24 - eating_hours} h de jeûne"


def eating_window_details(start_value: str, window_hours: int) -> tuple[str, bool]:
    """Renvoie le libellé de la fenêtre ; le booléen indique un passage à minuit."""
    start = parse_clock(start_value)
    start_minutes = minutes_since_midnight(start)
    duration = normalized_eating_window_hours(window_hours)
    end_minutes = (start_minutes + duration * 60) % (24 * 60)
    end = time(hour=end_minutes // 60, minute=end_minutes % 60)
    return f"{start.strftime('%H:%M')}–{end.strftime('%H:%M')}", end_minutes <= start_minutes


def is_inside_eating_window(
    meal_time: time,
    start_value: str,
    window_hours: int,
) -> bool:
    start_minutes = minutes_since_midnight(parse_clock(start_value))
    meal_minutes = minutes_since_midnight(meal_time)
    elapsed = (meal_minutes - start_minutes) % (24 * 60)
    return elapsed < normalized_eating_window_hours(window_hours) * 60


def product_relevance(query: str, product: dict[str, Any]) -> float:
    if query.strip() == str(product.get("code", "")).strip():
        return 1.0
    ignored = {
        "de", "del", "la", "le", "les", "un", "une", "y", "et", "con",
        "sin", "au", "aux", "en", "el", "los", "las",
    }
    query_tokens = {
        token for token in normalize_search_text(query).split() if token not in ignored
    }
    candidate = normalize_search_text(
        f"{product.get('product_name', '')} {product.get('brands', '')}"
    )
    candidate_tokens = set(candidate.split())
    if not query_tokens or not candidate_tokens:
        return 0.0
    overlap = len(query_tokens & candidate_tokens) / len(query_tokens)
    phrase_bonus = 0.15 if normalize_search_text(query) in candidate else 0.0
    return min(1.0, overlap + phrase_bonus)


def compact_product(product: dict[str, Any], query: str) -> dict[str, Any] | None:
    score = product_relevance(query, product)
    # Une variante approximative peut fausser tous les macronutriments du repas.
    # En dessous de 0,50, on préfère donc ne fournir aucune fiche au modèle.
    if score < 0.50:
        return None

    nutriments = product.get("nutriments") or {}
    nutrition_fields = {
        "energie_kcal_100g": nutriments.get("energy-kcal_100g"),
        "glucides_100g": nutriments.get("carbohydrates_100g"),
        "sucres_100g": nutriments.get("sugars_100g"),
        "fibres_100g": nutriments.get("fiber_100g"),
        "proteines_100g": nutriments.get("proteins_100g"),
        "lipides_100g": nutriments.get("fat_100g"),
        "graisses_saturees_100g": nutriments.get("saturated-fat_100g"),
        "sel_100g": nutriments.get("salt_100g"),
    }
    nutrition = {
        key: round(float(value), 2)
        for key, value in nutrition_fields.items()
        if isinstance(value, (int, float))
    }
    code = str(product.get("code", "")).strip()
    if not code or not nutrition:
        return None
    confidence = "élevée" if score >= 0.75 else "moyenne" if score >= 0.5 else "faible"
    return {
        "source": "open_food_facts",
        "requete": query.strip(),
        "nom": str(product.get("product_name") or "Produit sans nom").strip(),
        "marque": str(product.get("brands") or "Marque non précisée").strip(),
        "quantite": str(product.get("quantity") or "").strip(),
        "portion": str(product.get("serving_size") or "").strip(),
        "nutrition": nutrition,
        "code_barres": code,
        "url_source": f"https://world.openfoodfacts.org/product/{code}",
        "correspondance": confidence,
        "score_correspondance": round(score, 2),
    }


def brand_search_candidates(query: str) -> list[str]:
    """Produit quelques tags de marque possibles à partir du libellé saisi."""
    generic_food_words = {
        "agua", "avena", "bebida", "cafe", "cereal", "chocolate", "con",
        "descremada", "entera", "galletas", "leche", "light", "natural",
        "queso", "sin", "yogur", "yogurt", "zero",
    }
    all_tokens = [token for token in normalize_search_text(query).split() if len(token) > 1]
    tokens = [
        token
        for token in all_tokens
        if len(token) > 2 and token not in generic_food_words
    ]
    candidates = ["-".join(all_tokens)] if all_tokens else []
    candidates.extend(
        "-".join(all_tokens[index : index + 2])
        for index in range(len(all_tokens) - 1)
    )
    candidates.extend(tokens)
    return list(dict.fromkeys(candidate for candidate in candidates if candidate))[:8]


def open_food_facts_json(
    client: httpx.Client, url: str, params: dict[str, Any]
) -> dict[str, Any]:
    """Lit Open Food Facts avec deux nouvelles tentatives sur panne transitoire."""
    last_response: httpx.Response | None = None
    for attempt in range(3):
        try:
            response = client.get(url, params=params)
        except httpx.TransportError:
            if attempt == 2:
                raise
            sleep(0.4 * (attempt + 1))
            continue
        last_response = response
        if response.status_code not in OPEN_FOOD_FACTS_RETRYABLE_STATUS:
            response.raise_for_status()
            payload = response.json()
            if not isinstance(payload, dict):
                raise ValueError("Open Food Facts a renvoyé un format inattendu.")
            return payload
        if attempt < 2:
            sleep(0.4 * (attempt + 1))

    status = last_response.status_code if last_response is not None else "inconnu"
    if status == 503:
        raise RuntimeError(
            "Open Food Facts est temporairement indisponible (erreur 503)."
        )
    raise RuntimeError(f"Open Food Facts ne répond pas correctement (erreur {status}).")


@st.cache_data(ttl=24 * 60 * 60, max_entries=200, show_spinner=False)
def find_argentine_product(query: str) -> list[dict[str, Any]]:
    """Cherche le meilleur produit vendu en Argentine dans Open Food Facts."""
    clean_query = query.strip()
    if not clean_query:
        return []

    fields = (
        "code,product_name,brands,quantity,serving_size,countries_tags,nutriments"
    )
    headers = {"User-Agent": OPEN_FOOD_FACTS_USER_AGENT}
    with httpx.Client(timeout=12.0, headers=headers, follow_redirects=True) as client:
        if re.fullmatch(r"\d{8,14}", clean_query):
            payload = open_food_facts_json(
                client,
                OPEN_FOOD_FACTS_PRODUCT_URL.format(code=clean_query),
                {"fields": fields},
            )
            products = [payload.get("product", {})] if payload.get("status") == 1 else []
        else:
            brand_candidates = brand_search_candidates(clean_query)
            products = []
            for candidate in brand_candidates:
                try:
                    payload = open_food_facts_json(
                        client,
                        OPEN_FOOD_FACTS_STRUCTURED_SEARCH_URL,
                        {
                            "brands_tags": candidate,
                            "countries_tags_en": "argentina",
                            "page_size": 15,
                            "sort_by": "popularity_key",
                            "fields": fields,
                        },
                    )
                    products = payload.get("products", [])
                except httpx.HTTPStatusError as error:
                    # Une variante de tag inconnue peut échouer côté serveur ;
                    # on essaie alors le candidat suivant.
                    if error.response.status_code >= 500:
                        raise
                    continue
                except ValueError:
                    continue
                if products:
                    break

            # Une recherche de marque peut renvoyer un autre produit de la même
            # gamme. Si aucun résultat n'est assez proche, la recherche plein
            # texte a davantage de chances d'identifier la bonne variante.
            if products and not any(
                compact_product(product, clean_query) for product in products
            ):
                products = []

            # Le point d'accès historique est gardé uniquement en secours car
            # lui seul sait encore faire une recherche plein texte.
            if not products:
                payload = open_food_facts_json(
                    client,
                    OPEN_FOOD_FACTS_SEARCH_URL,
                    {
                        "search_terms": clean_query,
                        "search_simple": 1,
                        "action": "process",
                        "json": 1,
                        "page_size": 10,
                        "tagtype_0": "countries",
                        "tag_contains_0": "contains",
                        "tag_0": "Argentina",
                        "fields": fields,
                    },
                )
                products = payload.get("products", [])

    ranked = sorted(products, key=lambda item: product_relevance(clean_query, item), reverse=True)
    compact = [compact_product(product, clean_query) for product in ranked]
    return [product for product in compact if product is not None][:1]


def analyze_meal(
    settings: AISettings,
    description: str,
    product_matches: list[dict[str, Any]],
    country: str,
    eating_window_start: str,
    eating_window_hours: int,
) -> dict[str, Any]:
    reference_now = datetime.now(LOCAL_TIMEZONE)
    personal_count = sum(
        product.get("source") == "bibliotheque_personnelle"
        for product in product_matches
    )
    verified_context = (
        json.dumps(product_matches, ensure_ascii=False, indent=2)
        if product_matches
        else "Aucune fiche alimentaire personnelle ou produit vérifié n'a été fourni."
    )
    prompt = (
        "Analyse ce repas. Si des quantités manquent, fais une hypothèse prudente "
        "et mentionne-la brièvement dans le commentaire. Utilise en priorité les "
        "fiches marquées bibliotheque_personnelle : leurs valeurs proviennent de "
        "l'étiquette enregistrée par l'utilisateur. Les éventuelles données Open "
        "Food Facts sont communautaires : utilise-les seulement si le nom et la "
        "variante correspondent au repas, et signale toute incertitude.\n\n"
        f"PAYS / MARCHÉ : {country}\n"
        "STRATÉGIE : alimentation équilibrée non keto, perte progressive de "
        "3 à 4 kg puis stabilisation ; conserver des glucides de qualité.\n"
        f"DATE ET HEURE LOCALES DE RÉFÉRENCE : {reference_now.isoformat(timespec='minutes')}\n"
        "FUSEAU HORAIRE : America/Argentina/Buenos_Aires\n"
        "ORGANISATION DU JEÛNE : "
        f"{eating_window_pattern(eating_window_hours)} ; fenêtre "
        f"{eating_window_details(eating_window_start, eating_window_hours)[0]}\n"
        f"REPAS : {description.strip()}\n"
        f"NOMBRE DE FICHES PERSONNELLES ASSOCIÉES : {personal_count}\n\n"
        f"FICHES ALIMENTAIRES DISPONIBLES :\n{verified_context}"
    )
    payload, provider, model_name = call_ai(
        settings,
        MEAL_SYSTEM_PROMPT,
        prompt,
        MEAL_RESPONSE_SCHEMA,
        "meal_analysis",
    )
    analysis = normalize_meal_analysis(payload)
    analysis["_provider"] = provider
    analysis["_model_name"] = model_name
    if analysis["requires_timing_clarification"]:
        analysis["inside_eating_window"] = None
    else:
        interpreted_clock = time.fromisoformat(analysis["meal_time"])
        analysis["inside_eating_window"] = is_inside_eating_window(
            interpreted_clock,
            eating_window_start,
            eating_window_hours,
        )
    return analysis


def select_coaching_symptoms(
    symptom_events: list[dict[str, Any]],
    now: datetime | None = None,
) -> list[dict[str, Any]]:
    """Garde les signaux récents ou répétés et écarte les douleurs anciennes isolées."""
    reference = now or datetime.now(LOCAL_TIMEZONE)
    if reference.tzinfo is None:
        reference = reference.replace(tzinfo=LOCAL_TIMEZONE)
    active_pain_cutoff = reference - timedelta(hours=72)
    pattern_cutoff = reference - timedelta(days=14)
    hunger_cutoff = reference - timedelta(days=7)

    parsed: list[tuple[dict[str, Any], datetime]] = []
    pain_counts: dict[str, int] = {}
    for event in symptom_events:
        try:
            occurred_at = datetime.fromisoformat(str(event["occurred_at"]))
            if occurred_at.tzinfo is None:
                occurred_at = occurred_at.replace(tzinfo=LOCAL_TIMEZONE)
            else:
                occurred_at = occurred_at.astimezone(LOCAL_TIMEZONE)
        except (KeyError, TypeError, ValueError):
            continue
        parsed.append((event, occurred_at))
        if event.get("kind") == "douleur" and occurred_at >= pattern_cutoff:
            pain_key = str(event.get("pain_type") or "Douleur abdominale")
            pain_counts[pain_key] = pain_counts.get(pain_key, 0) + 1

    selected: list[dict[str, Any]] = []
    for event, occurred_at in parsed:
        if event.get("kind") == "faim":
            if occurred_at >= hunger_cutoff:
                selected.append(event)
            continue
        pain_key = str(event.get("pain_type") or "Douleur abdominale")
        if occurred_at >= active_pain_cutoff or (
            occurred_at >= pattern_cutoff and pain_counts.get(pain_key, 0) >= 2
        ):
            selected.append(event)
    return selected[:30]


def summarize_recent_bowel_events(
    bowel_events: list[dict[str, Any]],
    now: datetime | None = None,
) -> dict[str, Any]:
    """Prépare un contexte digestif prudent sur les 14 derniers jours."""
    reference = now or datetime.now(LOCAL_TIMEZONE)
    if reference.tzinfo is None:
        reference = reference.replace(tzinfo=LOCAL_TIMEZONE)
    cutoff = reference - timedelta(days=14)
    recent: list[dict[str, Any]] = []
    for event in bowel_events:
        try:
            occurred_at = datetime.fromisoformat(str(event["occurred_at"]))
            if occurred_at.tzinfo is None:
                occurred_at = occurred_at.replace(tzinfo=LOCAL_TIMEZONE)
            else:
                occurred_at = occurred_at.astimezone(LOCAL_TIMEZONE)
        except (KeyError, TypeError, ValueError):
            continue
        if occurred_at >= cutoff:
            recent.append(
                {
                    "date_heure": occurred_at.isoformat(timespec="minutes"),
                    "type_bristol": int(event["bristol_type"]),
                    "description": event.get("description") or None,
                }
            )
    recent = recent[:30]
    return {
        "nombre_observations_14_jours": len(recent),
        "analyse_autorisee": len(recent) >= 3,
        "observations": recent if len(recent) >= 3 else [],
    }


def build_global_report(
    settings: AISettings,
    meals: list[dict[str, Any]],
    weights: list[dict[str, Any]],
    profile: dict[str, Any],
    symptom_events: list[dict[str, Any]],
    bowel_events: list[dict[str, Any]],
    activity_events: list[dict[str, Any]],
    favorite_foods: list[dict[str, Any]],
) -> dict[str, Any]:
    # Les 60 repas les plus recents suffisent pour une tendance utile et limitent
    # la quantite de donnees envoyee a l'API.
    recent_meals = [
        {
            "date": meal.get("meal_occurred_at") or meal["created_at"],
            "repas": meal["description"],
            "note": meal["score"],
            "couleur": meal["color"],
            "type_repas": meal.get("meal_type", ""),
            "heure": meal.get("meal_time", ""),
            "dans_fenetre_alimentaire": meal.get("inside_eating_window"),
            "glucides_g": meal.get("carbs_g"),
            "proteines_g": meal.get("protein_g"),
            "lipides_g": meal.get("fat_g"),
        }
        for meal in meals[:60]
    ]
    recent_weights = [
        {
            "date": item["measured_on"],
            "poids_kg": item["weight_kg"],
        }
        for item in weights[-20:]
    ]
    coaching_symptoms = select_coaching_symptoms(symptom_events)
    recent_symptoms = [
        {
            "date_heure": event["occurred_at"],
            "type": event["kind"],
            "type_douleur": event.get("pain_type") or None,
            "intensite_sur_10": event["intensity"],
            "note": event.get("note") or None,
            "declencheur_rapporte": event.get("reported_trigger") or None,
            "confiance_extraction": event.get("extraction_confidence") or None,
            "repas_precedent": event.get("meal_description") or None,
            "delai_depuis_repas_minutes": event.get("delay_minutes"),
        }
        for event in coaching_symptoms
    ]
    bowel_summary = summarize_recent_bowel_events(bowel_events)
    recent_energy_balance = build_daily_energy_balance(
        meals, activity_events, profile
    ).tail(21)
    journal = {
        "objectif": profile["objective"],
        "strategie_alimentaire": (
            "Perte progressive de 3 à 4 kg puis stabilisation, alimentation "
            "équilibrée avec glucides de qualité, sans approche keto ni "
            "restriction glucidique forte."
        ),
        "taille_cm": profile.get("height_cm"),
        "age": profile.get("age_years"),
        "sexe_de_calcul": profile.get("sex"),
        "niveau_activite": profile.get("activity_level"),
        "poids_cible_kg": profile.get("target_weight_kg"),
        "objectifs_macros_automatiques": bool(profile.get("auto_macro_targets")),
        "pays_marche": profile["country"],
        "organisation_jeune": eating_window_pattern(
            profile.get("eating_window_hours", DEFAULT_EATING_WINDOW_HOURS)
        ),
        "fenetre_alimentaire": eating_window_details(
            profile["eating_window_start"],
            profile.get("eating_window_hours", DEFAULT_EATING_WINDOW_HOURS),
        )[0],
        "objectifs_quotidiens_g": {
            "glucides": profile["carbs_goal_g"],
            "proteines": profile["protein_goal_g"],
            "lipides": profile["fat_goal_g"],
        },
        "objectif_energetique_quotidien_kcal": profile.get(
            "calorie_goal_kcal", DEFAULT_CALORIE_GOAL_KCAL
        ),
        "nombre_total_de_repas": len(meals),
        "repas_recents": recent_meals,
        "poids_recents": recent_weights,
        "symptomes_actifs_ou_recurrents": recent_symptoms,
        "regle_symptomes": (
            "Ne pas mentionner les douleurs anciennes isolées : seules les "
            "douleurs de moins de 72 heures ou répétées au moins deux fois sur "
            "14 jours figurent dans cette liste."
        ),
        "journal_des_selles": bowel_summary,
        "energie_recente": [
            {
                "date": row["Date"].date().isoformat(),
                "calories_consignees": round(float(row["Calories estimées"])),
                "depense_activite_estimee": round(float(row["Dépense activité"])),
                "apport_net_indicatif": round(float(row["Apport net indicatif"])),
                "repere_alimentaire": round(float(row["Repère alimentaire"])),
            }
            for _, row in recent_energy_balance.iterrows()
        ],
        "activites_physiques_recentes": [
            {
                "date": event.get("occurred_on"),
                "type": event.get("activity_type"),
                "intensite": event.get("intensity"),
                "duree_minutes": event.get("duration_minutes"),
                "pas": event.get("steps"),
                "distance_km": event.get("distance_km"),
                "depense_estimee_kcal": event.get("calories_kcal"),
            }
            for event in activity_events[:30]
        ],
        "regle_activite": (
            "La dépense sportive est estimative et le niveau d'activité habituel "
            "du profil peut déjà inclure ces séances. Ne pas la compter deux fois "
            "ni conseiller de compenser systématiquement les calories."
        ),
        "aliments_personnels_enregistres": [
            {
                "nom": food.get("name"),
                "marque": food.get("brand"),
                "portion_habituelle_g": food.get("serving_size_g"),
                "kcal_100g": food.get("calories_100g"),
                "glucides_100g": food.get("carbs_100g"),
                "proteines_100g": food.get("protein_100g"),
                "lipides_100g": food.get("fat_100g"),
                "fibres_100g": food.get("fiber_100g"),
            }
            for food in favorite_foods[:40]
        ],
    }
    prompt = "Génère le bilan global à partir de ces données JSON :\n" + json.dumps(
        journal, ensure_ascii=False
    )
    payload, provider, model_name = call_ai(
        settings,
        REPORT_SYSTEM_PROMPT,
        prompt,
        REPORT_RESPONSE_SCHEMA,
        "global_report",
    )
    report = normalize_report(payload)
    report["_provider"] = provider
    report["_model_name"] = model_name
    report["_context_version"] = REPORT_CONTEXT_VERSION
    return report


def normalize_chat_response(payload: dict[str, Any]) -> dict[str, Any]:
    """Valide la réponse conversationnelle avant de l'afficher."""
    answer = str(payload.get("reponse", "")).strip()
    if not answer:
        raise ValueError("La réponse du coach IA est vide.")

    raw_suggestions = payload.get("suggestions", [])
    if not isinstance(raw_suggestions, list):
        raise ValueError("Les suggestions du coach IA sont invalides.")
    suggestions = [
        str(item).strip()
        for item in raw_suggestions[:3]
        if str(item).strip()
    ]
    medical_attention = bool(payload.get("avis_medical_recommande", False))
    safety_message = str(payload.get("message_securite", "")).strip()
    if medical_attention and not safety_message:
        safety_message = (
            "Ces éléments justifient un avis médical. En cas de symptôme intense "
            "ou inquiétant, contactez rapidement un professionnel ou les "
            "services d'urgence locaux."
        )
    return {
        "content": answer,
        "suggestions": suggestions,
        "medical_attention": medical_attention,
        "safety_message": safety_message,
    }


def build_chat_journal_context(
    meals: list[dict[str, Any]],
    weights: list[dict[str, Any]],
    profile: dict[str, Any],
    symptom_events: list[dict[str, Any]],
    bowel_events: list[dict[str, Any]],
    activity_events: list[dict[str, Any]],
    favorite_foods: list[dict[str, Any]],
) -> dict[str, Any]:
    """Prépare un contexte compact, récent et prudent pour le chat du coach."""
    recent_meals = []
    for meal in meals[:20]:
        occurred_at = meal_occurred_at(meal)
        recent_meals.append(
            {
                "date_heure": (
                    occurred_at.isoformat(timespec="minutes")
                    if occurred_at
                    else meal.get("created_at")
                ),
                "type": meal.get("meal_type") or "Autre",
                "description": meal.get("description") or "",
                "note_sur_10": meal.get("score"),
                "commentaire": meal.get("comment") or "",
                "glucides_g": meal.get("carbs_g"),
                "proteines_g": meal.get("protein_g"),
                "lipides_g": meal.get("fat_g"),
                "confiance_macros": meal.get("macro_confidence") or "",
                "dans_fenetre_alimentaire": meal.get("inside_eating_window"),
            }
        )

    daily_calories = build_daily_calorie_data(meals, profile).tail(14)
    calorie_history = [
        {
            "date": row["Date"].date().isoformat(),
            "calories_estimees": round(float(row["Calories estimées"])),
            "repere_kcal": round(float(row["Objectif"])),
            "repas_avec_estimation": int(row["Repas estimés"]),
        }
        for _, row in daily_calories.iterrows()
    ]
    energy_balance = build_daily_energy_balance(
        meals, activity_events, profile
    ).tail(14)
    energy_history = [
        {
            "date": row["Date"].date().isoformat(),
            "calories_consignees": round(float(row["Calories estimées"])),
            "depense_activite_estimee": round(float(row["Dépense activité"])),
            "apport_net_indicatif": round(float(row["Apport net indicatif"])),
            "repere_alimentaire": round(float(row["Repère alimentaire"])),
        }
        for _, row in energy_balance.iterrows()
    ]

    active_symptoms = select_coaching_symptoms(symptom_events)
    symptom_context = [
        {
            "date_heure": event.get("occurred_at"),
            "type": event.get("kind"),
            "type_douleur": event.get("pain_type") or None,
            "intensite_sur_10": event.get("intensity"),
            "description": event.get("note") or None,
            "declencheur_rapporte": event.get("reported_trigger") or None,
        }
        for event in active_symptoms
    ]

    return {
        "date_locale": datetime.now(LOCAL_TIMEZONE).isoformat(timespec="minutes"),
        "profil": {
            "prenom": profile.get("display_name") or None,
            "objectif": profile.get("objective"),
            "strategie_alimentaire": (
                "Perdre 3 à 4 kg progressivement puis stabiliser ; alimentation "
                "équilibrée avec glucides de qualité ; pas de régime keto ni "
                "de restriction glucidique forte."
            ),
            "age": profile.get("age_years"),
            "sexe_de_calcul": profile.get("sex"),
            "taille_cm": profile.get("height_cm"),
            "activite": profile.get("activity_level"),
            "poids_cible_kg": profile.get("target_weight_kg"),
            "pays_marche": profile.get("country"),
            "organisation_jeune": eating_window_pattern(
                profile.get("eating_window_hours", DEFAULT_EATING_WINDOW_HOURS)
            ),
            "fenetre_alimentaire": eating_window_details(
                str(profile.get("eating_window_start") or DEFAULT_EATING_WINDOW_START),
                profile.get("eating_window_hours", DEFAULT_EATING_WINDOW_HOURS),
            )[0],
            "repere_quotidien": {
                "calories_kcal": profile.get("calorie_goal_kcal"),
                "glucides_g": profile.get("carbs_goal_g"),
                "proteines_g": profile.get("protein_goal_g"),
                "lipides_g": profile.get("fat_goal_g"),
                "repas_habituels": profile.get("meals_per_day"),
            },
        },
        "poids_recents": [
            {
                "date": item.get("measured_on"),
                "poids_kg": item.get("weight_kg"),
            }
            for item in weights[-12:]
        ],
        "repas_recents": recent_meals,
        "calories_par_jour_recents": calorie_history,
        "energie_apports_et_activite_recente": energy_history,
        "cumul_semaine_en_cours": build_current_macro_progress(
            meals, profile, "Cette semaine"
        ),
        "cumul_mois_en_cours": build_current_macro_progress(
            meals, profile, "Ce mois"
        ),
        "symptomes_actifs_ou_recurrents": symptom_context,
        "regle_symptomes": (
            "Les douleurs anciennes isolées sont exclues. Ne pas les mentionner."
        ),
        "journal_des_selles": summarize_recent_bowel_events(bowel_events),
        "activites_physiques_recentes": [
            {
                "date": event.get("occurred_on"),
                "type": event.get("activity_type"),
                "intensite": event.get("intensity"),
                "duree_minutes": event.get("duration_minutes"),
                "pas": event.get("steps"),
                "distance_km": event.get("distance_km"),
                "depense_estimee_kcal": event.get("calories_kcal"),
            }
            for event in activity_events[:21]
        ],
        "regle_activite": (
            "La dépense sportive est une estimation. Le niveau d'activité du "
            "profil peut déjà inclure ces séances : éviter le double comptage et "
            "ne pas recommander de manger automatiquement les calories dépensées."
        ),
        "aliments_personnels": [
            {
                "nom": food.get("name"),
                "marque": food.get("brand"),
                "portion_g": food.get("serving_size_g"),
                "kcal_100g": food.get("calories_100g"),
                "glucides_100g": food.get("carbs_100g"),
                "proteines_100g": food.get("protein_100g"),
                "lipides_100g": food.get("fat_100g"),
                "fibres_100g": food.get("fiber_100g"),
            }
            for food in favorite_foods[:40]
        ],
        "limite_interpretation": (
            "Les jours ou repas non consignés rendent les cumuls incomplets et "
            "ne signifient pas une consommation nulle."
        ),
    }


def answer_coach_question(
    settings: AISettings,
    question: str,
    conversation: list[dict[str, Any]],
    meals: list[dict[str, Any]],
    weights: list[dict[str, Any]],
    profile: dict[str, Any],
    symptom_events: list[dict[str, Any]],
    bowel_events: list[dict[str, Any]],
    activity_events: list[dict[str, Any]],
    favorite_foods: list[dict[str, Any]],
) -> dict[str, Any]:
    """Répond à une question avec le contexte récent du journal."""
    journal_context = build_chat_journal_context(
        meals,
        weights,
        profile,
        symptom_events,
        bowel_events,
        activity_events,
        favorite_foods,
    )
    conversation_context = [
        {
            "role": str(message.get("role") or ""),
            "contenu": str(message.get("content") or "")[:2000],
        }
        for message in conversation[-10:]
        if message.get("role") in {"user", "assistant"}
        and str(message.get("content") or "").strip()
    ]
    prompt = (
        "CONTEXTE DU JOURNAL :\n"
        + json.dumps(journal_context, ensure_ascii=False, default=str)
        + "\n\nÉCHANGES PRÉCÉDENTS :\n"
        + json.dumps(conversation_context, ensure_ascii=False)
        + "\n\nQUESTION ACTUELLE :\n"
        + question.strip()
    )
    payload, provider, model_name = call_ai(
        settings,
        CHAT_SYSTEM_PROMPT,
        prompt,
        CHAT_RESPONSE_SCHEMA,
        "coach_conversation",
    )
    response = normalize_chat_response(payload)
    response["provider"] = provider
    response["model_name"] = model_name
    return response


def provider_error_message(
    provider: str,
    error: Exception,
    settings: AISettings,
) -> str:
    """Transforme une erreur fournisseur en message lisible, sans exposer de clé."""
    message = str(error).strip() or error.__class__.__name__
    for api_key in (
        settings.deepseek_api_key,
        settings.gemini_api_key,
        settings.groq_api_key,
    ):
        if api_key:
            message = message.replace(api_key, "***")
    normalized = message.upper()
    if "402" in normalized or "PREPAYMENT CREDITS ARE DEPLETED" in normalized:
        return f"{provider} : crédit API prépayé épuisé (402)."
    if "401" in normalized or "INVALID_API_KEY" in normalized:
        return f"{provider} : clé invalide ou incomplète (401)."
    if "403" in normalized or "PERMISSION_DENIED" in normalized:
        return f"{provider} : accès refusé (403). Vérifiez la clé et le compte."
    if "404" in normalized or "NOT_FOUND" in normalized:
        return f"{provider} : modèle indisponible pour ce compte (404)."
    if "429" in normalized or "RATE_LIMIT" in normalized:
        return f"{provider} : quota temporairement atteint (429)."
    if "RÉPONSE VIDE" in normalized or "REPONSE VIDE" in normalized:
        return f"{provider} : réponse vide après deux tentatives."
    if "CONNECTION" in normalized or "TIMEOUT" in normalized:
        return f"{provider} : connexion impossible ou délai dépassé."
    return f"{provider} : {message[:500]}"


def safe_error_message(error: Exception, settings: AISettings) -> str:
    if isinstance(error, AIProvidersUnavailable):
        return " · ".join(
            provider_error_message(provider, failure, settings)
            for provider, failure in error.failures
        )
    return provider_error_message("IA", error, settings)


def load_persistent_api_key(secret_name: str) -> str:
    """Charge la clé locale sans jamais l'envoyer à l'interface en clair."""
    if not SECRETS_PATH.is_file():
        return ""

    # Streamlit charge automatiquement secrets.toml lorsque le serveur est
    # lancé depuis le dossier de l'application.
    try:
        configured_key = str(st.secrets.get(secret_name, "")).strip()
        if configured_key:
            return configured_key
    except FileNotFoundError:
        # Si l'application a été lancée depuis un autre dossier, Streamlit peut
        # ne pas trouver ce fichier. On lit alors exactement le même TOML local.
        pass

    try:
        content = tomllib.loads(SECRETS_PATH.read_text(encoding="utf-8"))
    except (OSError, tomllib.TOMLDecodeError):
        return ""
    return str(content.get(secret_name, "")).strip()


def save_persistent_api_key(secret_name: str, api_key: str) -> None:
    """Enregistre la clé dans un fichier privé, distinct du code et de SQLite."""
    clean_key = api_key.strip()
    if not clean_key:
        raise ValueError("La clé API est vide.")

    SECRETS_DIR.mkdir(parents=True, exist_ok=True, mode=0o700)
    SECRETS_DIR.chmod(0o700)
    SECRETS_PATH.touch(exist_ok=True, mode=0o600)
    SECRETS_PATH.chmod(0o600)
    try:
        existing = tomllib.loads(SECRETS_PATH.read_text(encoding="utf-8"))
    except (OSError, tomllib.TOMLDecodeError):
        existing = {}
    existing[secret_name] = clean_key
    serialized = "".join(
        f"{name} = {json.dumps(str(value))}\n"
        for name, value in sorted(existing.items())
        if re.fullmatch(r"[A-Z][A-Z0-9_]*", name)
    )
    SECRETS_PATH.write_text(
        serialized,
        encoding="utf-8",
    )
    SECRETS_PATH.chmod(0o600)


def forget_persistent_api_key(secret_name: str) -> None:
    """Retire une clé précise, sans toucher aux autres secrets locaux."""
    if not SECRETS_PATH.is_file():
        return
    try:
        existing = tomllib.loads(SECRETS_PATH.read_text(encoding="utf-8"))
    except (OSError, tomllib.TOMLDecodeError):
        return
    existing.pop(secret_name, None)
    serialized = "".join(
        f"{name} = {json.dumps(str(value))}\n"
        for name, value in sorted(existing.items())
        if re.fullmatch(r"[A-Z][A-Z0-9_]*", name)
    )
    SECRETS_PATH.write_text(serialized, encoding="utf-8")
    SECRETS_PATH.chmod(0o600)


def readable_comment(value: Any) -> tuple[str, bool]:
    """Masque uniquement le fragment final d'une ancienne réponse interrompue."""
    comment = str(value).strip()
    if len(comment) < 450 or not comment or comment[-1] in ".!?…":
        return comment, False
    sentence_end = max(comment.rfind("."), comment.rfind("!"), comment.rfind("?"))
    if sentence_end < 0:
        return comment, False
    return comment[: sentence_end + 1], True


def render_analysis_card(analysis: dict[str, Any]) -> None:
    meta = COLOR_META[analysis["couleur"]]
    score = int(analysis["note"])
    comment, was_interrupted = readable_comment(analysis["commentaire"])
    with st.container(border=True):
        score_col, comment_col = st.columns([0.2, 0.8], vertical_alignment="center")
        score_col.metric("Profil glycémique estimé", f"{score}/10")
        with comment_col:
            st.markdown(f"### {meta['emoji']} {meta['label']}")
            st.markdown(comment)
            if was_interrupted:
                st.caption(
                    "La fin de cette ancienne réponse avait été interrompue par "
                    "le fournisseur IA ; le fragment incomplet a été masqué."
                )

    if all(key in analysis for key in ("glucides_g", "proteines_g", "lipides_g")):
        with st.container(
            border=True,
            horizontal=True,
            horizontal_alignment="distribute",
        ):
            st.markdown(f"**Glucides**  \n### {analysis['glucides_g']:.1f} g")
            st.markdown(f"**Protéines**  \n### {analysis['proteines_g']:.1f} g")
            st.markdown(f"**Lipides**  \n### {analysis['lipides_g']:.1f} g")
        st.caption(
            "Confiance de l'estimation : "
            f"**{analysis.get('confiance_macros', 'non précisée')}**. "
            "Les valeurs restent approximatives si les portions ne sont pas pesées."
        )


def render_meal_history_card(meal: dict[str, Any]) -> None:
    """Affiche un repas complet sans défilement horizontal ni texte masqué."""
    meta = COLOR_META[meal["color"]]
    historical_pattern = eating_window_pattern(
        meal.get("eating_window_hours", 8)
    )
    window_label = (
        f"✅ Dans la fenêtre ({historical_pattern})"
        if meal.get("inside_eating_window") == 1
        else f"⚠️ Hors de la fenêtre ({historical_pattern})"
        if meal.get("inside_eating_window") == 0
        else "Fenêtre alimentaire non déterminée"
    )
    comment, was_interrupted = readable_comment(meal["comment"])
    with st.container(border=True):
        identity_col, score_col = st.columns(
            [0.76, 0.24], vertical_alignment="center"
        )
        with identity_col:
            st.markdown(
                f"#### {meal.get('meal_type') or 'Repas'} · "
                f"{format_meal_datetime(meal)}"
            )
            st.caption(f"{window_label} · {meal.get('macro_confidence') or 'confiance non précisée'}")
        with score_col:
            st.markdown(f"### {meta['emoji']} {meal['score']}/10")
            st.caption(meta["label"])

        st.markdown(f"**Repas**  \n{meal['description']}")
        st.markdown(f"**Analyse**  \n{comment}")
        if was_interrupted:
            st.caption(
                "La fin de cette ancienne réponse avait été interrompue ; "
                "le fragment incomplet a été masqué."
            )

        if meal.get("carbs_g") is not None:
            with st.container(horizontal=True, horizontal_alignment="distribute"):
                st.markdown(f"**Glucides** · {meal['carbs_g']:.1f} g")
                st.markdown(f"**Protéines** · {meal['protein_g']:.1f} g")
                st.markdown(f"**Lipides** · {meal['fat_g']:.1f} g")

        if meal.get("product_matches"):
            with st.expander(
                "Valeurs du produit utilisées",
                icon=":material/nutrition:",
            ):
                render_product_sources(meal["product_matches"], embedded=True)


def render_product_sources(
    products: list[dict[str, Any]], *, embedded: bool = False
) -> None:
    """Affiche la fiche ayant servi à l'analyse sans masquer son incertitude."""
    if not products:
        return
    source_panel = (
        st.container(border=True)
        if embedded
        else st.expander("Valeurs nutritionnelles utilisées", expanded=True)
    )
    with source_panel:
        if embedded:
            st.markdown("##### Valeurs nutritionnelles utilisées")
        for product in products:
            if product.get("source") == "bibliotheque_personnelle":
                source_description = "Fiche personnelle enregistrée localement"
            else:
                source_description = (
                    f"Correspondance **{product['correspondance']}** · "
                    f"[Voir la fiche Open Food Facts]({product['url_source']})"
                )
            st.markdown(
                f"**{product['nom']}** — {product['marque']}  \n"
                f"{source_description}"
            )
            nutrition_labels = {
                "energie_kcal_100g": "Énergie",
                "glucides_100g": "Glucides",
                "sucres_100g": "Sucres",
                "fibres_100g": "Fibres",
                "proteines_100g": "Protéines",
                "lipides_100g": "Lipides",
                "graisses_saturees_100g": "Graisses saturées",
                "sel_100g": "Sel",
            }
            values = []
            for key, value in product["nutrition"].items():
                unit = "kcal/100 g" if key == "energie_kcal_100g" else "g/100 g"
                values.append(f"**{nutrition_labels[key]} :** {value:g} {unit}")
            st.caption(" · ".join(values))
        st.caption(
            "Les fiches personnelles proviennent des valeurs que vous avez "
            "enregistrées. Les autres viennent d'Open Food Facts. Dans les deux "
            "cas, vérifiez l'étiquette si la recette ou le format change."
        )


def format_datetime(value: str) -> str:
    try:
        parsed = datetime.fromisoformat(value)
        return parsed.strftime("%d/%m/%Y a %H:%M")
    except ValueError:
        return value


def format_meal_datetime(meal: dict[str, Any]) -> str:
    occurred_at = meal_occurred_at(meal)
    if occurred_at is None:
        return format_datetime(str(meal.get("created_at") or ""))
    return occurred_at.strftime("%d/%m/%Y a %H:%M")


def format_delay(minutes: int | None) -> str:
    if minutes is None:
        return "—"
    hours, remaining = divmod(int(minutes), 60)
    if hours and remaining:
        return f"{hours} h {remaining:02d}"
    if hours:
        return f"{hours} h"
    return f"{remaining} min"


def macro_target_status(actual_g: float, target_g: float) -> tuple[float, str]:
    """Classe l'écart sans présenter un apport bas comme une alerte médicale."""
    ratio = actual_g / target_g if target_g > 0 else 0.0
    if ratio < 0.80:
        status = "bas"
    elif ratio <= 1.10:
        status = "vert"
    elif ratio <= 1.30:
        status = "orange"
    else:
        status = "rouge"
    return ratio, status


def estimated_meal_calories(meal: dict[str, Any]) -> float | None:
    """Estime l'énergie d'un repas à partir des trois macros enregistrées."""
    macro_columns = ("carbs_g", "protein_g", "fat_g")
    if any(meal.get(column) is None for column in macro_columns):
        return None
    return round(
        4 * float(meal["carbs_g"])
        + 4 * float(meal["protein_g"])
        + 9 * float(meal["fat_g"]),
        0,
    )


def build_daily_calorie_data(
    meals: list[dict[str, Any]], profile: dict[str, Any]
) -> pd.DataFrame:
    """Agrège les calories estimées pour chaque journée réellement renseignée."""
    daily_rows: list[dict[str, Any]] = []
    for meal in meals:
        occurred_at = meal_occurred_at(meal)
        calories = estimated_meal_calories(meal)
        if occurred_at is None or calories is None:
            continue
        daily_rows.append(
            {
                "Date": occurred_at.astimezone(LOCAL_TIMEZONE).date(),
                "Calories estimées": calories,
                "Repas estimés": 1,
            }
        )

    if not daily_rows:
        return pd.DataFrame()

    frame = pd.DataFrame(daily_rows)
    daily = frame.groupby("Date", as_index=False).agg(
        {"Calories estimées": "sum", "Repas estimés": "sum"}
    )
    goal = float(
        profile.get("calorie_goal_kcal") or DEFAULT_CALORIE_GOAL_KCAL
    )
    daily["Objectif"] = goal
    daily["Écart"] = daily["Calories estimées"] - goal
    daily["Date"] = pd.to_datetime(daily["Date"])
    return daily.sort_values("Date")


def build_daily_activity_data(
    activity_events: list[dict[str, Any]],
) -> pd.DataFrame:
    """Agrège la dépense sportive estimée par date de pratique."""
    if not activity_events:
        return pd.DataFrame()
    frame = pd.DataFrame(activity_events)
    frame["Date"] = pd.to_datetime(frame["occurred_on"], errors="coerce")
    frame["Dépense activité"] = pd.to_numeric(
        frame["calories_kcal"], errors="coerce"
    ).fillna(0.0)
    frame = frame.dropna(subset=["Date"])
    if frame.empty:
        return pd.DataFrame()
    daily = frame.groupby("Date", as_index=False).agg(
        {"Dépense activité": "sum", "id": "count"}
    )
    return daily.rename(columns={"id": "Activités"}).sort_values("Date")


def build_daily_energy_balance(
    meals: list[dict[str, Any]],
    activity_events: list[dict[str, Any]],
    profile: dict[str, Any],
) -> pd.DataFrame:
    """Réunit apports et activités sans modifier le repère alimentaire."""
    intake = build_daily_calorie_data(meals, profile)
    activity = build_daily_activity_data(activity_events)
    if intake.empty and activity.empty:
        return pd.DataFrame()
    frames = []
    if not intake.empty:
        frames.append(intake[["Date", "Calories estimées", "Repas estimés"]])
    if not activity.empty:
        frames.append(activity[["Date", "Dépense activité", "Activités"]])
    balance = frames[0]
    for frame in frames[1:]:
        balance = balance.merge(frame, on="Date", how="outer")
    for column in (
        "Calories estimées",
        "Repas estimés",
        "Dépense activité",
        "Activités",
    ):
        if column not in balance:
            balance[column] = 0.0
        balance[column] = balance[column].fillna(0.0)
    goal = float(profile.get("calorie_goal_kcal") or DEFAULT_CALORIE_GOAL_KCAL)
    balance["Repère alimentaire"] = goal
    balance["Apport net indicatif"] = (
        balance["Calories estimées"] - balance["Dépense activité"]
    )
    balance["Écart net indicatif"] = (
        balance["Apport net indicatif"] - balance["Repère alimentaire"]
    )
    return balance.sort_values("Date")


def build_weight_projection(
    meals: list[dict[str, Any]],
    weights: list[dict[str, Any]],
    profile: dict[str, Any],
    horizon_days: int = 84,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Projette prudemment le poids à partir des journées alimentaires complètes."""
    if not weights:
        return pd.DataFrame(), {"reason": "Ajoutez d'abord une mesure de poids."}

    daily = build_daily_calorie_data(meals, profile)
    if daily.empty:
        return pd.DataFrame(), {
            "reason": "Aucune journée ne comporte encore d'estimation calorique."
        }

    expected_meals = max(1, int(profile.get("meals_per_day") or 1))
    today = pd.Timestamp(date.today())
    complete_days = daily[
        (daily["Repas estimés"] >= expected_meals)
        & (daily["Date"] <= today)
        & (daily["Date"] >= today - pd.Timedelta(days=28))
    ].tail(14)
    if len(complete_days) < 3:
        return pd.DataFrame(), {
            "reason": (
                "Il faut au moins 3 journées complètes sur les 28 derniers jours "
                f"({expected_meals} repas consignés par jour) pour éviter une "
                "projection artificiellement basse."
            ),
            "complete_days": int(len(complete_days)),
        }

    latest_weight = float(weights[-1]["weight_kg"])
    targets = calculate_macro_targets(
        latest_weight,
        float(profile.get("height_cm") or DEFAULT_HEIGHT_CM),
        str(profile.get("objective") or OBJECTIVE_OPTIONS[0]),
        int(profile.get("age_years") or DEFAULT_AGE_YEARS),
        str(profile.get("sex") or DEFAULT_SEX),
        str(profile.get("activity_level") or DEFAULT_ACTIVITY_LEVEL),
        float(profile.get("target_weight_kg") or DEFAULT_TARGET_WEIGHT_KG),
    )
    average_intake = float(complete_days["Calories estimées"].mean())
    maintenance = float(targets["maintenance_energy_kcal"])
    daily_gap = average_intake - maintenance
    intake_uncertainty = min(300.0, max(150.0, average_intake * 0.10))
    adaptation_days = 180.0
    target_weight = float(
        profile.get("target_weight_kg") or DEFAULT_TARGET_WEIGHT_KG
    )
    objective = str(profile.get("objective") or OBJECTIVE_OPTIONS[0])
    stabilize_at_target = (
        objective == "Perdre du poids progressivement"
        and latest_weight > target_weight
    )

    def projected_weight(day_number: int, intake: float) -> float:
        energy_gap = intake - maintenance
        adaptive_days = adaptation_days * (
            1 - math.exp(-day_number / adaptation_days)
        )
        raw_projection = latest_weight + energy_gap * adaptive_days / 7700.0
        if stabilize_at_target:
            # Cette courbe représente le plan de l'application : le déficit est
            # levé à la cible au lieu d'être extrapolé comme s'il continuait.
            return max(raw_projection, target_weight)
        return raw_projection

    start_date = date.today()
    rows = []
    for day_number in range(max(1, int(horizon_days)) + 1):
        central = projected_weight(day_number, average_intake)
        lower = projected_weight(day_number, average_intake - intake_uncertainty)
        upper = projected_weight(day_number, average_intake + intake_uncertainty)
        rows.append(
            {
                "Date": pd.Timestamp(start_date + timedelta(days=day_number)),
                "Projection": round(central, 3),
                "Scénario bas": round(min(lower, upper), 3),
                "Scénario haut": round(max(lower, upper), 3),
            }
        )

    projection = pd.DataFrame(rows)
    metadata = {
        "complete_days": int(len(complete_days)),
        "average_intake_kcal": round(average_intake),
        "maintenance_kcal": round(maintenance),
        "daily_gap_kcal": round(daily_gap),
        "latest_weight_kg": latest_weight,
        "projected_weight_kg": float(projection.iloc[-1]["Projection"]),
        "uncertainty_kcal": round(intake_uncertainty),
        "horizon_days": int(horizon_days),
        "target_weight_kg": target_weight,
        "stabilizes_at_target": stabilize_at_target,
    }
    return projection, metadata


def build_current_macro_progress(
    meals: list[dict[str, Any]], profile: dict[str, Any], period_name: str
) -> dict[str, Any]:
    """Compare deux cumuls sur les mêmes jours calendaires jusqu'à aujourd'hui."""
    now = datetime.now(LOCAL_TIMEZONE)
    today = now.date()
    if period_name == "Aujourd'hui":
        start_date = today
        full_period_end_date = today + timedelta(days=1)
        total_days = 1
        period_label = f"Aujourd'hui · {today:%d/%m/%Y}"
    elif period_name == "Cette semaine":
        start_date = today - timedelta(days=today.weekday())
        full_period_end_date = start_date + timedelta(days=7)
        total_days = 7
        period_label = (
            f"Cumul depuis le lundi {start_date:%d/%m} "
            f"jusqu'au {today:%d/%m/%Y}"
        )
    else:
        start_date = today.replace(day=1)
        if today.month == 12:
            full_period_end_date = date(today.year + 1, 1, 1)
        else:
            full_period_end_date = date(today.year, today.month + 1, 1)
        total_days = (full_period_end_date - start_date).days
        period_label = (
            f"Cumul depuis le 01/{today:%m} jusqu'au {today:%d/%m/%Y}"
        )

    elapsed_days = min((today - start_date).days + 1, total_days)
    comparison_end_date = min(
        today + timedelta(days=1),
        full_period_end_date,
    )
    start_at = datetime.combine(start_date, time.min, tzinfo=LOCAL_TIMEZONE)
    comparison_end_at = datetime.combine(
        comparison_end_date,
        time.min,
        tzinfo=LOCAL_TIMEZONE,
    )
    totals = {"carbs_g": 0.0, "protein_g": 0.0, "fat_g": 0.0}
    meal_count = 0
    for meal in meals:
        occurred_at = meal_occurred_at(meal)
        if (
            occurred_at is None
            or not start_at <= occurred_at < comparison_end_at
        ):
            continue
        if any(meal.get(column) is None for column in totals):
            continue
        for column in totals:
            totals[column] += float(meal[column])
        meal_count += 1

    macro_config = [
        ("Glucides", "carbs_g", float(profile["carbs_goal_g"])),
        ("Protéines", "protein_g", float(profile["protein_goal_g"])),
        ("Lipides", "fat_g", float(profile["fat_goal_g"])),
    ]
    indicators = []
    for label, column, daily_target in macro_config:
        target_to_date = daily_target * elapsed_days
        actual = round(totals[column], 1)
        ratio, status = macro_target_status(actual, target_to_date)
        indicators.append(
            {
                "label": label,
                "actual_g": actual,
                "target_g": round(target_to_date, 1),
                "full_period_target_g": round(daily_target * total_days, 1),
                "ratio": ratio,
                "status": status,
            }
        )
    return {
        "period_label": period_label,
        "elapsed_days": elapsed_days,
        "total_days": total_days,
        "comparison_start_date": start_date,
        "comparison_end_date": comparison_end_date,
        "meal_count": meal_count,
        "indicators": indicators,
    }


def build_macro_gauge(indicator: dict[str, Any]) -> alt.LayerChart:
    """Construit une jauge linéaire : bas → cible → haut."""
    zones = pd.DataFrame(
        [
            {"Début": 0, "Fin": 80, "Couleur": "#3b82f6", "Ligne": "Zone"},
            {"Début": 80, "Fin": 110, "Couleur": "#22c55e", "Ligne": "Zone"},
            {"Début": 110, "Fin": 130, "Couleur": "#f59e0b", "Ligne": "Zone"},
            {"Début": 130, "Fin": 150, "Couleur": "#ef4444", "Ligne": "Zone"},
        ]
    )
    zone_chart = (
        alt.Chart(zones)
        .mark_bar(size=30)
        .encode(
            x=alt.X(
                "Début:Q",
                scale=alt.Scale(domain=[0, 150], clamp=True),
                axis=alt.Axis(
                    title=None,
                    values=[0, 60, 80, 100, 110, 130, 150],
                    labelExpr="datum.value + '%'",
                    labelFontSize=10,
                    tickSize=3,
                    grid=False,
                ),
            ),
            x2=alt.X2("Fin:Q"),
            y=alt.Y("Ligne:N", axis=None, title=None),
            color=alt.Color("Couleur:N", scale=None, legend=None),
        )
    )

    marker_position = min(max(indicator["ratio"] * 100, 0), 150)
    marker_data = pd.DataFrame(
        [
            {
                "Position": marker_position,
                "Consommé": indicator["actual_g"],
                "Objectif": indicator["target_g"],
                "Pourcentage": indicator["ratio"] * 100,
            }
        ]
    )
    current_marker = (
        alt.Chart(marker_data)
        .mark_rule(strokeWidth=4, color="#17211b")
        .encode(
            x=alt.X("Position:Q", scale=alt.Scale(domain=[0, 150], clamp=True)),
            tooltip=[
                alt.Tooltip("Consommé:Q", title="Consommé", format=".1f"),
                alt.Tooltip("Objectif:Q", title="Objectif", format=".1f"),
                alt.Tooltip("Pourcentage:Q", title="Objectif atteint", format=".0f"),
            ],
        )
    )
    current_point = (
        alt.Chart(marker_data)
        .mark_point(shape="diamond", filled=True, size=120, color="#17211b")
        .encode(
            x=alt.X("Position:Q", scale=alt.Scale(domain=[0, 150], clamp=True))
        )
    )
    target_marker = (
        alt.Chart(pd.DataFrame([{"Position": 100}]))
        .mark_rule(strokeWidth=2, strokeDash=[4, 3], color="#ffffff")
        .encode(
            x=alt.X("Position:Q", scale=alt.Scale(domain=[0, 150], clamp=True))
        )
    )
    return (zone_chart + target_marker + current_marker + current_point).properties(
        height=75
    )


def build_macro_chart_data(
    meals: list[dict[str, Any]], profile: dict[str, Any], granularity: str
) -> pd.DataFrame:
    """Agrège les macros et calcule un objectif comparable pour chaque période."""
    macro_columns = ["carbs_g", "protein_g", "fat_g"]
    valid_meals = [
        meal for meal in meals if all(meal.get(column) is not None for column in macro_columns)
    ]
    if not valid_meals:
        return pd.DataFrame()

    frame = pd.DataFrame(valid_meals)
    frame["Moment"] = [
        (
            meal_occurred_at(meal).astimezone(LOCAL_TIMEZONE).replace(tzinfo=None)
            if meal_occurred_at(meal) is not None
            else pd.NaT
        )
        for meal in valid_meals
    ]
    frame = frame.dropna(subset=["Moment"])
    if frame.empty:
        return pd.DataFrame()

    if granularity == "Repas":
        frame = frame.sort_values("Moment")
        aggregated = frame[["Moment", "meal_type", *macro_columns]].copy()
        aggregated["Période"] = aggregated.apply(
            lambda row: (
                f"{row['Moment']:%d/%m %H:%M} · "
                f"{row.get('meal_type') or 'Repas'}"
            ),
            axis=1,
        )
        aggregated["Objectif_facteur"] = 1 / max(
            1, int(profile.get("meals_per_day", DEFAULT_MEALS_PER_DAY))
        )
    else:
        if granularity == "Jour":
            frame["Groupe"] = frame["Moment"].dt.floor("D")
            label_format = "%d/%m/%Y"
            factor = lambda period: 1
        elif granularity == "Semaine":
            frame["Groupe"] = frame["Moment"].dt.to_period("W-SUN").dt.start_time
            label_format = "Sem. du %d/%m/%Y"
            factor = lambda period: 7
        else:
            frame["Groupe"] = frame["Moment"].dt.to_period("M").dt.start_time
            label_format = "%m/%Y"
            factor = lambda period: period.days_in_month

        aggregated = (
            frame.groupby("Groupe", as_index=False)[macro_columns]
            .sum()
            .sort_values("Groupe")
        )
        aggregated["Moment"] = aggregated["Groupe"]
        aggregated["Période"] = aggregated["Groupe"].dt.strftime(label_format)
        aggregated["Objectif_facteur"] = aggregated["Groupe"].map(factor)

    macro_config = [
        ("Glucides", "carbs_g", float(profile["carbs_goal_g"])),
        ("Protéines", "protein_g", float(profile["protein_goal_g"])),
        ("Lipides", "fat_g", float(profile["fat_goal_g"])),
    ]
    rows: list[dict[str, Any]] = []
    for _, period in aggregated.iterrows():
        for macro_name, column, daily_goal in macro_config:
            rows.append(
                {
                    "Période": period["Période"],
                    "Ordre": period["Moment"],
                    "Macro": macro_name,
                    "Série": "Réalité estimée",
                    "Grammes": round(float(period[column]), 1),
                }
            )
            rows.append(
                {
                    "Période": period["Période"],
                    "Ordre": period["Moment"],
                    "Macro": macro_name,
                    "Série": "Objectif",
                    "Grammes": round(daily_goal * float(period["Objectif_facteur"]), 1),
                }
            )
    return pd.DataFrame(rows)


def inject_styles() -> None:
    st.markdown(
        """
        <style>
        :root { --ink:#17211b; --muted:#647067; --cream:#f8faf7; }
        .stApp {
          background:
            radial-gradient(circle at 8% 0%, rgba(167,243,208,.28), transparent 30rem),
            radial-gradient(circle at 95% 10%, rgba(254,215,170,.25), transparent 28rem),
            #f8faf7;
        }
        .block-container { max-width: 1380px; padding-top: 1rem; }
        .hero {
          padding:.85rem 1.2rem; border:1px solid rgba(20,83,45,.1);
          border-radius:18px; background:rgba(255,255,255,.82);
          box-shadow:0 12px 35px rgba(22,58,37,.06); margin-bottom:.7rem;
        }
        .hero h1 { color:var(--ink); margin:0; font-size:clamp(1.7rem,3vw,2.35rem); }
        .hero p { color:var(--muted); margin:.2rem 0 0; font-size:.95rem; }
        .eyebrow { color:#15803d; font-weight:800; letter-spacing:.12em; font-size:.73rem; }
        div[data-testid="stMetric"] {
          background:rgba(255,255,255,.82); border:1px solid rgba(22,58,37,.09);
          padding:.8rem 1rem; border-radius:16px;
        }
        div[data-testid="stForm"] { background:rgba(255,255,255,.72); border-radius:18px; }
        .small-card {
          background:white; padding:1rem 1.1rem; border-radius:15px;
          border:1px solid rgba(22,58,37,.09); margin:.45rem 0;
        }

        /* Navigation tactile : les onglets restent sur une seule ligne et
           défilent horizontalement au lieu d'être comprimés. */
        .st-key-main_navigation > [data-baseweb="tab-list"],
        .st-key-journal_navigation > [data-baseweb="tab-list"],
        .st-key-assistant_navigation > [data-baseweb="tab-list"] {
          overflow-x:auto;
          overflow-y:hidden;
          scrollbar-width:none;
          -webkit-overflow-scrolling:touch;
          overscroll-behavior-inline:contain;
        }
        .st-key-main_navigation > [data-baseweb="tab-list"]::-webkit-scrollbar,
        .st-key-journal_navigation > [data-baseweb="tab-list"]::-webkit-scrollbar,
        .st-key-assistant_navigation > [data-baseweb="tab-list"]::-webkit-scrollbar {
          display:none;
        }
        .st-key-main_navigation > [data-baseweb="tab-list"] button,
        .st-key-journal_navigation > [data-baseweb="tab-list"] button,
        .st-key-assistant_navigation > [data-baseweb="tab-list"] button {
          min-height:44px;
          flex:0 0 auto;
          white-space:nowrap;
        }

        @media (max-width: 640px) {
          html { -webkit-text-size-adjust:100%; }
          .block-container {
            max-width:100%;
            padding-top:.45rem;
            padding-left:.75rem;
            padding-right:.75rem;
            padding-bottom:max(1.25rem, env(safe-area-inset-bottom));
          }
          .hero {
            padding:.72rem .85rem;
            margin-bottom:.35rem;
            border-radius:14px;
            box-shadow:0 6px 18px rgba(22,58,37,.05);
          }
          .hero h1 { font-size:1.55rem; line-height:1.15; }
          .hero p { font-size:.88rem; line-height:1.35; }
          h1 { font-size:1.75rem !important; line-height:1.2 !important; }
          h2 { font-size:1.42rem !important; line-height:1.25 !important; }
          h3 { font-size:1.16rem !important; line-height:1.3 !important; }

          /* Navigation principale accessible pendant le défilement. */
          .st-key-main_navigation > [data-baseweb="tab-list"] {
            position:sticky;
            top:2.8rem;
            z-index:90;
            margin:0 -.75rem .35rem;
            padding:0 .65rem;
            background:rgba(248,250,247,.97);
            backdrop-filter:blur(10px);
            -webkit-backdrop-filter:blur(10px);
            border-bottom:1px solid rgba(22,58,37,.10);
          }
          .st-key-main_navigation > [data-baseweb="tab-list"] button {
            min-height:48px;
            padding-left:.72rem;
            padding-right:.72rem;
            font-size:.9rem;
          }
          .st-key-journal_navigation > [data-baseweb="tab-list"],
          .st-key-assistant_navigation > [data-baseweb="tab-list"] {
            margin-bottom:.25rem;
          }
          .st-key-journal_navigation > [data-baseweb="tab-list"] button,
          .st-key-assistant_navigation > [data-baseweb="tab-list"] button {
            padding-left:.65rem;
            padding-right:.65rem;
            font-size:.87rem;
          }

          /* Cibles tactiles et saisie sans zoom automatique d'iOS. */
          div[data-testid="stButton"] button,
          div[data-testid="stFormSubmitButton"] button,
          div[data-testid="stDownloadButton"] button {
            min-height:44px;
          }
          input, textarea, select { font-size:16px !important; }
          textarea { min-height:112px; }
          div[data-testid="stForm"] {
            padding:.75rem;
            border-radius:14px;
          }
          div[data-testid="stMetric"] {
            padding:.68rem .78rem;
            border-radius:13px;
            min-height:auto;
          }
          div[data-testid="stMetricLabel"] {
            white-space:normal;
            overflow:visible;
          }
          div[data-testid="stMetricValue"] {
            font-size:1.55rem;
            line-height:1.2;
            overflow-wrap:anywhere;
          }
          .small-card {
            padding:.82rem .88rem;
            border-radius:13px;
          }
          div[data-testid="stDataFrame"],
          div[data-testid="stTable"] {
            max-width:100%;
            overflow-x:auto;
            -webkit-overflow-scrolling:touch;
          }
          div[data-testid="stAlert"] { padding:.75rem .8rem; }
          section[data-testid="stSidebar"] {
            width:min(92vw, 360px) !important;
            padding-bottom:env(safe-area-inset-bottom);
          }
        }
        </style>
        """,
        unsafe_allow_html=True,
    )


init_database()
inject_styles()
cloud_session = require_cloud_login()

st.markdown(
    """
    <section class="hero">
      <h1>🥗 Coach Glycémie</h1>
      <p>Notez simplement. Comprenez la tendance. Ajustez un pas à la fois.</p>
    </section>
    """,
    unsafe_allow_html=True,
)

with st.sidebar:
    st.success(f"Cloud privé connecté\n\n{cloud_session.email}", icon="☁️")
    if st.button(
        "Fermer ma session",
        icon=":material/logout:",
        width="stretch",
        key="logout_cloud_session",
    ):
        st.session_state.pop("supabase_session", None)
        st.session_state["remember_cloud_session"] = False
        st.session_state["persistent_session_checked"] = True
        queue_browser_session_action("clear")
        st.rerun()
    st.caption("Les nouvelles données sont enregistrées directement dans Supabase.")
    st.divider()
    st.header("Connexion IA")
    st.markdown("#### DeepSeek · principal")
    saved_deepseek_key = load_persistent_api_key(DEEPSEEK_SECRET_NAME)
    if st.session_state.pop("forget_deepseek_api_key", False):
        st.session_state["deepseek_api_key_input"] = ""
    st.session_state.setdefault("deepseek_api_key_input", saved_deepseek_key)
    deepseek_api_key = st.text_input(
        "Clé API DeepSeek",
        type="password",
        placeholder="Collez une nouvelle clé ici",
        help=(
            "Créez une nouvelle clé sur DeepSeek Platform. N'utilisez pas une "
            "clé déjà publiée dans un message ou une capture d'écran."
        ),
        key="deepseek_api_key_input",
    ).strip()
    st.markdown(
        "[Créer une clé dans DeepSeek Platform ↗]"
        "(https://platform.deepseek.com/api_keys)"
    )
    if saved_deepseek_key and deepseek_api_key == saved_deepseek_key:
        st.success("Clé DeepSeek chargée depuis ce Mac.", icon="🔐")
        if st.button(
            "Oublier la clé DeepSeek",
            icon=":material/delete:",
            width="stretch",
            key="forget_saved_deepseek_key",
        ):
            forget_persistent_api_key(DEEPSEEK_SECRET_NAME)
            st.session_state["forget_deepseek_api_key"] = True
            st.rerun()
    elif deepseek_api_key:
        st.info("Clé prête pour cette session, mais pas encore enregistrée.", icon="💡")
        if st.button(
            "Enregistrer la clé DeepSeek sur ce Mac",
            type="primary",
            icon=":material/save:",
            width="stretch",
            key="save_deepseek_key",
        ):
            try:
                save_persistent_api_key(DEEPSEEK_SECRET_NAME, deepseek_api_key)
                st.rerun()
            except (OSError, ValueError) as error:
                st.error(f"Impossible d'enregistrer la clé : {error}")
    else:
        st.info("Ajoutez une clé DeepSeek pour activer le fournisseur principal.", icon="💡")

    if deepseek_api_key and st.button(
        "Tester DeepSeek",
        icon=":material/wifi_tethering:",
        width="stretch",
        key="test_deepseek_connection",
    ):
        with st.spinner("Vérification de la clé DeepSeek…"):
            try:
                test_deepseek_connection(deepseek_api_key)
                st.success(f"Connexion réussie avec {DEEPSEEK_MODEL_NAME}.")
            except Exception as error:
                st.error(
                    safe_error_message(
                        error,
                        AISettings(deepseek_api_key=deepseek_api_key),
                    )
                )

    st.divider()
    st.markdown("#### Google Gemini · secours")
    saved_gemini_key = load_persistent_api_key(GEMINI_SECRET_NAME)
    if st.session_state.pop("forget_gemini_api_key", False):
        st.session_state["gemini_api_key_input"] = ""
    st.session_state.setdefault("gemini_api_key_input", saved_gemini_key)
    gemini_api_key = st.text_input(
        "Clé API Gemini",
        type="password",
        placeholder="Collez une nouvelle clé ici",
        help=(
            "Créez une clé Gemini Developer API dans Google AI Studio. Le numéro "
            "de projet n'est pas nécessaire dans cette application."
        ),
        key="gemini_api_key_input",
    ).strip()
    st.markdown(
        "[Créer une clé dans Google AI Studio ↗]"
        "(https://aistudio.google.com/app/apikey)"
    )
    if saved_gemini_key and gemini_api_key == saved_gemini_key:
        st.success("Clé Gemini chargée depuis ce Mac.", icon="🔐")
        if st.button(
            "Oublier la clé Gemini",
            icon=":material/delete:",
            width="stretch",
            key="forget_saved_gemini_key",
        ):
            forget_persistent_api_key(GEMINI_SECRET_NAME)
            st.session_state["forget_gemini_api_key"] = True
            st.rerun()
    elif gemini_api_key:
        st.info("Clé prête pour cette session, mais pas encore enregistrée.", icon="💡")
        if st.button(
            "Enregistrer la clé Gemini sur ce Mac",
            type="primary",
            icon=":material/save:",
            width="stretch",
            key="save_gemini_key",
        ):
            try:
                save_persistent_api_key(GEMINI_SECRET_NAME, gemini_api_key)
                st.rerun()
            except (OSError, ValueError) as error:
                st.error(f"Impossible d'enregistrer la clé : {error}")
    else:
        st.caption("Clé Gemini facultative pour le premier secours automatique.")

    if gemini_api_key and st.button(
        "Tester Gemini",
        icon=":material/wifi_tethering:",
        width="stretch",
        key="test_gemini_connection",
    ):
        with st.spinner("Vérification de la clé Gemini…"):
            try:
                test_gemini_connection(gemini_api_key)
                st.success(f"Connexion réussie avec {GEMINI_MODEL_NAME}.")
            except Exception as error:
                st.error(
                    safe_error_message(
                        error,
                        AISettings(gemini_api_key=gemini_api_key),
                    )
                )

    with st.expander(
        "Groq · second secours facultatif",
        expanded=not bool(deepseek_api_key or gemini_api_key),
    ):
        saved_groq_key = load_persistent_api_key(GROQ_SECRET_NAME)
        if st.session_state.pop("forget_groq_api_key", False):
            st.session_state["groq_api_key_input"] = ""
        st.session_state.setdefault("groq_api_key_input", saved_groq_key)
        groq_api_key = st.text_input(
            "Clé API Groq",
            type="password",
            placeholder="Clé de secours facultative",
            help="Groq sera essayé automatiquement si DeepSeek et Gemini échouent.",
            key="groq_api_key_input",
        ).strip()
        st.markdown(
            "[Gérer les clés dans Groq Console ↗]"
            "(https://console.groq.com/keys)"
        )
        if saved_groq_key and groq_api_key == saved_groq_key:
            st.success("Clé Groq chargée depuis ce Mac.", icon="🔐")
            if st.button(
                "Oublier la clé Groq",
                icon=":material/delete:",
                width="stretch",
                key="forget_saved_groq_key",
            ):
                forget_persistent_api_key(GROQ_SECRET_NAME)
                st.session_state["forget_groq_api_key"] = True
                st.rerun()
        elif groq_api_key:
            st.info("Clé Groq prête pour cette session.", icon="💡")
            if st.button(
                "Enregistrer la clé Groq sur ce Mac",
                icon=":material/save:",
                width="stretch",
                key="save_groq_key",
            ):
                try:
                    save_persistent_api_key(GROQ_SECRET_NAME, groq_api_key)
                    st.rerun()
                except (OSError, ValueError) as error:
                    st.error(f"Impossible d'enregistrer la clé : {error}")
        if groq_api_key and st.button(
            "Tester Groq",
            icon=":material/wifi_tethering:",
            width="stretch",
            key="test_groq_connection",
        ):
            with st.spinner("Vérification de la clé Groq…"):
                try:
                    test_groq_connection(groq_api_key)
                    st.success(f"Connexion réussie avec {GROQ_MODEL_NAME}.")
                except Exception as error:
                    st.error(
                        safe_error_message(
                            error,
                            AISettings(groq_api_key=groq_api_key),
                        )
                    )

    ai_settings = AISettings(
        deepseek_api_key=deepseek_api_key,
        gemini_api_key=gemini_api_key,
        groq_api_key=groq_api_key,
    )
    if deepseek_api_key:
        backups = []
        if gemini_api_key:
            backups.append("Gemini")
        if groq_api_key:
            backups.append("Groq")
        backup_text = (
            " · secours : " + " puis ".join(backups)
            if backups
            else ""
        )
        st.success(f"DeepSeek principal{backup_text}", icon="✅")
    elif gemini_api_key:
        st.warning("Gemini est actif sans DeepSeek principal.", icon="⚠️")
    elif groq_api_key:
        st.warning("Groq est actif seul ; ajoutez DeepSeek comme fournisseur principal.", icon="⚠️")
    else:
        st.warning("Aucune clé IA configurée.", icon="⚠️")

    st.divider()
    st.caption(f"Principal : `{DEEPSEEK_MODEL_NAME}`")
    st.caption(f"Secours 1 : `{GEMINI_MODEL_NAME}`")
    st.caption(f"Secours 2 : `{GROQ_MODEL_NAME}`")
    st.caption("Marché nutritionnel : 🇦🇷 Argentine")
    st.caption("Données : ☁️ Supabase privé")
    st.caption(
        "Les descriptions de repas, les fiches alimentaires associées, un résumé "
        "des activités, les symptômes, les descriptions de selles et les "
        "questions adressées à l'assistant nutritionnel sont envoyés "
        "à DeepSeek ou, en cas d'échec, à Gemini puis Groq. Pour une conversation "
        "ou un bilan IA, seul un contexte récent et utile du journal est joint. "
        "Le journal complet reste dans votre base Supabase privée."
    )
    st.warning(
        "Ce coach fournit une estimation éducative, pas un avis médical. "
        "En cas de diabète ou de traitement, consultez un professionnel de santé.",
        icon="⚕️",
    )
    st.caption(
        "[Consulter les signes d'alerte abdominale sur MedlinePlus ↗]"
        "(https://medlineplus.gov/abdominalpain.html)"
    )

meal_tab, journal_hub, coach_tab, assistant_hub = st.tabs(
    [
        ":material/edit_note: Aujourd'hui",
        ":material/menu_book: Journal",
        ":material/monitoring: Progression",
        ":material/chat: Coach IA",
    ],
    key="main_navigation",
)
with journal_hub:
    history_tab, symptom_tab, bowel_tab, activity_tab = st.tabs(
        [
            ":material/restaurant: Repas",
            ":material/ecg_heart: Symptômes",
            ":material/gastroenterology: Digestion",
            ":material/directions_run: Activité",
        ],
        key="journal_navigation",
    )
with assistant_hub:
    assistant_tab, food_tab = st.tabs(
        [
            ":material/chat: Discussion",
            ":material/grocery: Aliments",
        ],
        key="assistant_navigation",
    )
active_profile = load_profile()
active_eating_window_hours = normalized_eating_window_hours(
    active_profile.get("eating_window_hours", DEFAULT_EATING_WINDOW_HOURS)
)
active_window_label = eating_window_details(
    active_profile["eating_window_start"], active_eating_window_hours
)[0]

with meal_tab:
    favorite_foods = load_favorite_foods()
    favorite_food_by_id = {int(food["id"]): food for food in favorite_foods}
    today_value = date.today()
    today_meals = [
        meal
        for meal in load_meals()
        if (meal_time := meal_occurred_at(meal)) is not None
        and meal_time.date() == today_value
    ]
    today_calories = sum(
        calories
        for meal in today_meals
        if (calories := estimated_meal_calories(meal)) is not None
    )
    today_activity = sum(
        float(event.get("calories_kcal") or 0)
        for event in load_activity_events()
        if str(event.get("occurred_on") or "") == today_value.isoformat()
    )
    today_score = (
        sum(int(meal["score"]) for meal in today_meals) / len(today_meals)
        if today_meals
        else None
    )
    calorie_target = float(
        active_profile.get("calorie_goal_kcal") or DEFAULT_CALORIE_GOAL_KCAL
    )
    glycemic_status = score_color(today_score) if today_score is not None else None
    calorie_ratio, energy_status = daily_energy_status(
        today_calories,
        calorie_target,
        day_is_complete=eating_window_has_ended_today(
            str(active_profile.get("eating_window_start") or DEFAULT_EATING_WINDOW_START),
            active_eating_window_hours,
        ),
    )
    daily_status = combined_daily_status(glycemic_status, energy_status)
    glycemic_meta = (
        DAILY_STATUS_META[glycemic_status]
        if glycemic_status is not None
        else DAILY_STATUS_META["en_cours"]
    )
    energy_meta = DAILY_STATUS_META[energy_status]
    daily_meta = DAILY_STATUS_META[daily_status]

    with st.container(horizontal=True, horizontal_alignment="distribute"):
        st.subheader("Votre journée en un coup d'œil")
        with st.popover(
            "Comprendre le score",
            icon=":material/info:",
        ):
            st.markdown(
                """
**Deux axes sont volontairement séparés.**

- **Glycémie probable :** vert à 7–10, orange à 4–6, rouge à 0–3.
- **Énergie :** après la fin de la fenêtre alimentaire, vert entre 85 et
  110 % de la cible de perte de poids, orange entre 70–84 % ou 111–120 %,
  rouge en dehors de ces plages.
- **Cap du jour :** vert seulement lorsque les deux axes sont verts ; rouge si
  l'un des deux est rouge ; sinon orange.

Avant la fin de la fenêtre, l'énergie reste bleue « journée en cours ». La
cible calorique inclut déjà votre niveau d'activité habituel : les calories du
sport sont affichées séparément pour éviter de compter deux fois la dépense.
                """
            )
    summary_columns = st.columns(3)
    summary_columns[0].metric(
        "Profil glycémique estimé",
        (
            f"{glycemic_meta['emoji']} {today_score:.1f}/10"
            if today_score is not None
            else "🔵 —"
        ),
    )
    summary_columns[1].metric(
        "Énergie vs cible",
        f"{energy_meta['emoji']} {today_calories:.0f} / {calorie_target:.0f} kcal",
        delta=f"{calorie_ratio * 100:.0f} % de la cible",
        delta_color="off",
    )
    summary_columns[2].metric(
        "Cap du jour",
        f"{daily_meta['emoji']} {daily_meta['label']}",
    )
    st.caption(
        f"{len(today_meals)} repas notés · activité enregistrée : "
        f"{today_activity:.0f} kcal estimées. Votre priorité : observer la "
        "tendance sur plusieurs jours, pas corriger brutalement une seule journée."
    )

    left = st.container()
    with left:
        st.subheader("Que souhaitez-vous noter ?")
        st.caption(
            "Décrivez librement un repas, une faim ou un inconfort digestif, avec "
            "le jour et l'heure. Vous pouvez aussi signaler les deux dans la même "
            "phrase. Votre fenêtre actuelle est "
            f"**{active_window_label}** ({eating_window_pattern(active_eating_window_hours)})."
        )
        if st.session_state.pop("clear_meal_form", False):
            st.session_state["meal_text_input"] = ""
            st.session_state["product_query_input"] = ""
            st.session_state["meal_favorite_foods"] = []
        if meal_save_notice := st.session_state.pop("meal_save_notice", ""):
            st.success(meal_save_notice)
        selected_food_ids: list[int] = []
        with st.form("meal_form", clear_on_submit=False):
            meal_text = st.text_area(
                "Repas, faim ou inconfort digestif",
                height=155,
                placeholder=(
                    "Ex. : Hier à 12 h 30, j'ai mangé 3 œufs et un avocat. "
                    "Aujourd'hui à 19 h 45, petite douleur du nombril au sternum."
                ),
                label_visibility="collapsed",
                key="meal_text_input",
            )
            product_query = st.text_input(
                "Produit ou marque à vérifier (facultatif)",
                placeholder="Ex. : Yogurísimo La Serenísima ou le code-barres",
                help=(
                    "Pour un produit emballé, indiquez son nom exact, sa marque ou "
                    "son code-barres. La recherche est limitée aux produits vendus "
                    "en Argentine."
                ),
                key="product_query_input",
            )
            if favorite_foods:
                selected_food_ids = st.multiselect(
                    "Aliments de ma bibliothèque utilisés dans ce repas",
                    options=list(favorite_food_by_id),
                    format_func=lambda food_id: favorite_food_label(
                        favorite_food_by_id[int(food_id)]
                    ),
                    help=(
                        "L'application essaie aussi de les reconnaître dans votre "
                        "texte. Sélectionnez-les ici si vous voulez lever toute "
                        "ambiguïté."
                    ),
                    key="meal_favorite_foods",
                )
            analyze_clicked = st.form_submit_button(
                "✨ Interpréter et enregistrer", type="primary", width="stretch"
            )

        if analyze_clicked:
            if not meal_text.strip():
                st.warning("Décrivez un repas, une faim ou un inconfort digestif.")
            elif not ai_settings.has_any_key:
                st.error("Ajoutez une clé DeepSeek, Gemini ou Groq dans la barre latérale.")
            else:
                with st.spinner("Interprétation de votre texte…"):
                    try:
                        intake = {
                            "events": [],
                            "contains_meal": True,
                            "meal_description": meal_text.strip(),
                        }
                        if text_may_contain_symptom(meal_text):
                            intake = parse_intake_text(ai_settings, meal_text)
                            saved_count, symptom_questions = save_ready_symptom_events(
                                intake["events"], meal_text, load_meals()
                            )
                            if saved_count:
                                st.success(
                                    f"{saved_count} signalement"
                                    f"{'s' if saved_count > 1 else ''} d'inconfort ou "
                                    f"de faim enregistré{'s' if saved_count > 1 else ''} "
                                    "dans l'onglet Faim & douleurs."
                                )
                            if symptom_questions:
                                st.warning(
                                    "Précision nécessaire avant d'enregistrer :\n\n- "
                                    + "\n- ".join(symptom_questions)
                                )
                            if not intake["events"] and not intake["contains_meal"]:
                                st.info(
                                    "Je n'ai identifié ni repas ni signalement exploitable. "
                                    "Précisez ce qui s'est passé, le moment et l'intensité."
                                )

                        if intake["contains_meal"]:
                            meal_description = intake["meal_description"]
                            personal_matches = match_favorite_foods(
                                f"{meal_description} {product_query}",
                                favorite_foods,
                                selected_food_ids,
                            )
                            product_matches = list(personal_matches)
                            lookup_error = ""
                            if product_query.strip():
                                matching_personal_query = match_favorite_foods(
                                    product_query, favorite_foods
                                )
                                if not matching_personal_query:
                                    try:
                                        product_matches.extend(
                                            find_argentine_product(product_query)
                                        )
                                    except (
                                        httpx.HTTPError,
                                        ValueError,
                                        json.JSONDecodeError,
                                        RuntimeError,
                                    ) as error:
                                        lookup_error = str(error)
                            analysis = analyze_meal(
                                ai_settings,
                                meal_description,
                                product_matches,
                                active_profile["country"],
                                active_profile["eating_window_start"],
                                active_eating_window_hours,
                            )
                            if analysis["requires_timing_clarification"]:
                                st.warning(
                                    "Repas non enregistré : "
                                    f"{analysis['timing_clarification_question']} "
                                    "Ajoutez la précision directement dans votre texte."
                                )
                            else:
                                save_meal(
                                    meal_description,
                                    analysis,
                                    product_matches,
                                    active_profile["country"],
                                    active_profile["eating_window_start"],
                                    active_eating_window_hours,
                                )
                                timing_source = (
                                    "interprété par l’IA · confiance "
                                    f"{analysis['confiance_horaire']}"
                                )
                                st.session_state["last_analysis"] = analysis
                                st.session_state["last_ai_provider"] = analysis.get(
                                    "_provider", "IA cloud"
                                )
                                st.session_state["last_meal"] = meal_description
                                st.session_state["last_products"] = product_matches
                                st.session_state["last_product_query"] = product_query.strip()
                                st.session_state["last_product_lookup_error"] = lookup_error
                                st.session_state["last_meal_type"] = analysis["meal_type"]
                                st.session_state["last_meal_type_source"] = timing_source
                                st.session_state["last_meal_date"] = analysis["meal_date"]
                                st.session_state["last_meal_time"] = analysis["meal_time"]
                                st.session_state["last_inside_window"] = analysis[
                                    "inside_eating_window"
                                ]
                                st.session_state["last_window_label"] = active_window_label
                                st.session_state["meal_save_notice"] = (
                                    "Repas interprété, analysé et ajouté au journal. "
                                    "La zone de saisie est prête pour un nouveau repas."
                                )
                                st.session_state["clear_meal_form"] = True
                                st.rerun()
                    except Exception as error:  # L'UI doit transformer toute erreur API en message lisible.
                        st.error(
                            "Analyse impossible. Vérifiez la clé, le quota et la connexion, puis réessayez.\n\n"
                            f"Détail : {safe_error_message(error, ai_settings)}"
                        )

        if "last_analysis" in st.session_state:
            st.markdown("#### Dernier repas analysé")
            st.caption(st.session_state.get("last_meal", ""))
            window_icon = "✅" if st.session_state.get("last_inside_window") else "⚠️"
            window_position = (
                "dans la fenêtre alimentaire"
                if st.session_state.get("last_inside_window")
                else "hors de la fenêtre alimentaire"
            )
            try:
                last_date_label = date.fromisoformat(
                    st.session_state.get("last_meal_date", "")
                ).strftime("%d/%m/%Y")
            except (TypeError, ValueError):
                last_date_label = "date non précisée"
            st.info(
                f"**{st.session_state.get('last_meal_type', 'Repas')}** le "
                f"**{last_date_label}** à "
                f"**{st.session_state.get('last_meal_time', '—')}** "
                f"({st.session_state.get('last_meal_type_source', '')}) · "
                f"{window_icon} {window_position} "
                f"{st.session_state.get('last_window_label', '')}"
            )
            render_analysis_card(st.session_state["last_analysis"])
            st.caption(
                "Analyse réalisée par "
                f"{st.session_state.get('last_ai_provider', 'IA cloud')}."
            )
            render_product_sources(st.session_state.get("last_products", []))
            last_query = st.session_state.get("last_product_query", "")
            if last_query and not st.session_state.get("last_products"):
                if st.session_state.get("last_product_lookup_error"):
                    lookup_error = st.session_state["last_product_lookup_error"]
                    if "503" in lookup_error or "temporairement" in lookup_error:
                        st.warning(
                            "Open Food Facts était temporairement indisponible, "
                            "malgré plusieurs tentatives automatiques. Le repas a "
                            "bien été enregistré avec une estimation prudente, "
                            "sans utiliser de fiche produit."
                        )
                    else:
                        st.warning(
                            "La base nutritionnelle n'était pas joignable. "
                            "L'analyse a été réalisée sans valeurs produit "
                            "vérifiées."
                        )
                else:
                    st.info(
                        f"Aucune fiche suffisamment proche de « {last_query} » n'a "
                        "été trouvée en Argentine. L'analyse reste qualitative. "
                        "Pour fiabiliser les prochains calculs, ouvrez "
                        "Nutritionniste IA → Mes aliments, puis recherchez le "
                        "code-barres ou faites lire le texte de l'étiquette."
                    )

with history_tab:
    meals = load_meals()
    st.subheader("Votre journal alimentaire")
    st.caption(
        "Chaque repas affiche directement son type, son évaluation, le commentaire "
        "complet et les estimations nutritionnelles. Aucun défilement horizontal "
        "n'est nécessaire."
    )
    if not meals:
        st.info("Aucun repas enregistré pour le moment. Votre première analyse apparaîtra ici.")
    else:
        average = sum(meal["score"] for meal in meals) / len(meals)
        green_share = 100 * sum(meal["color"] == "vert" for meal in meals) / len(meals)
        metric_1, metric_2, metric_3 = st.columns(3)
        metric_1.metric("Repas analysés", len(meals))
        metric_2.metric("Note moyenne", f"{average:.1f}/10")
        metric_3.metric("Repas au profil favorable", f"{green_share:.0f}%")

        history_scope = st.segmented_control(
            "Repas affichés",
            ["7 récents", "30 récents", "Tous"],
            default="7 récents",
            key="meal_history_scope",
        )
        history_limit = {"7 récents": 7, "30 récents": 30}.get(history_scope)
        visible_meals = meals[:history_limit] if history_limit else meals
        st.caption(
            f"{len(visible_meals)} repas affichés sur {len(meals)}. "
            "Utilisez le choix ci-dessus uniquement lorsque vous avez besoin "
            "de remonter plus loin."
        )
        for meal in visible_meals:
            render_meal_history_card(meal)


with symptom_tab:
    meals = load_meals()
    symptom_events = load_symptom_events()
    st.subheader("Journal de faim et douleurs abdominales")
    st.caption(
        "Enregistrez une sensation lorsqu'elle apparaît. Elle sera reliée "
        "automatiquement au dernier repas des 24 heures précédentes."
    )
    with st.form("natural_symptom_form", clear_on_submit=True, border=True):
        st.markdown("#### Décrire naturellement")
        natural_symptom_text = st.text_area(
            "Faim, douleur, jour, heure, intensité et origine supposée",
            placeholder=(
                "Ex. : Hier à 17 h 30, j'ai eu une faim forte 8/10 après le "
                "goûter. Aujourd'hui à 22 h, brûlure 6/10 entre le sternum et "
                "le nombril ; je pense que le café l'a déclenchée."
            ),
            height=125,
            key="natural_symptom_text",
        )
        natural_symptom_clicked = st.form_submit_button(
            "Interpréter et enregistrer",
            type="primary",
            icon=":material/auto_awesome:",
            width="stretch",
        )

    if natural_symptom_clicked:
        if not natural_symptom_text.strip():
            st.warning("Décrivez au moins une sensation de faim ou une douleur.")
        elif not ai_settings.has_any_key:
            st.error("Ajoutez une clé DeepSeek, Gemini ou Groq dans la barre latérale.")
        else:
            with st.spinner("Interprétation des événements et des horaires…"):
                try:
                    extraction = parse_symptom_text(ai_settings, natural_symptom_text)
                    extracted_events = extraction["events"]
                    saved_count, questions = save_ready_symptom_events(
                        extracted_events, natural_symptom_text, meals
                    )
                    if saved_count:
                        st.success(
                            f"{saved_count} événement"
                            f"{'s' if saved_count > 1 else ''} interprété"
                            f"{'s' if saved_count > 1 else ''} et enregistré"
                            f"{'s' if saved_count > 1 else ''}."
                        )
                        st.caption(
                            f"Interprétation réalisée par {extraction.get('_provider', 'IA cloud')}."
                        )
                        symptom_events = load_symptom_events()
                    if questions:
                        st.warning(
                            "Précision nécessaire avant d'enregistrer le reste :\n\n- "
                            + "\n- ".join(questions)
                        )
                    if not extracted_events:
                        st.info(
                            "Aucune sensation de faim ou douleur identifiable n'a "
                            "été trouvée dans ce texte."
                        )
                except Exception as error:
                    st.error(
                        "Interprétation impossible. Vérifiez la connexion IA, puis "
                        f"réessayez.\n\nDétail : {safe_error_message(error, ai_settings)}"
                    )

    st.markdown("##### Saisie manuelle — solution de secours")
    hunger_col, pain_col = st.columns(2, gap="large")
    with hunger_col:
        with st.form("hunger_event_form", clear_on_submit=True, border=True):
            st.markdown("#### Sensation de faim")
            hunger_date = st.date_input("Date", value=date.today(), key="hunger_date")
            hunger_time = st.time_input("Heure", value="now", step=300, key="hunger_time")
            hunger_intensity = st.slider(
                "Intensité de la faim", 1, 10, 5, key="hunger_intensity"
            )
            hunger_note = st.text_area(
                "Description facultative",
                placeholder="Ex. : faim soudaine, envie de sucre, sensation progressive…",
                key="hunger_note",
            )
            hunger_clicked = st.form_submit_button(
                "Enregistrer la faim", icon=":material/add:", width="stretch"
            )
        if hunger_clicked:
            meal_id, delay = save_symptom_event(
                "faim",
                hunger_date,
                hunger_time,
                hunger_intensity,
                hunger_note,
                meals,
            )
            if meal_id is None:
                st.success("Sensation de faim enregistrée, sans repas précédent associé.")
            else:
                st.success(
                    f"Sensation enregistrée et reliée au repas précédent "
                    f"({format_delay(delay)} après)."
                )
            symptom_events = load_symptom_events()

    with pain_col:
        with st.form("pain_event_form", clear_on_submit=True, border=True):
            st.markdown("#### Douleur abdominale")
            pain_date = st.date_input("Date", value=date.today(), key="pain_date")
            pain_time = st.time_input("Heure", value="now", step=300, key="pain_time")
            pain_type = st.selectbox("Type de douleur", PAIN_TYPES, key="pain_type")
            pain_intensity = st.slider(
                "Intensité de la douleur", 1, 10, 5, key="pain_intensity"
            )
            pain_note = st.text_area(
                "Description facultative",
                placeholder="Ex. : durée, côté concerné, ballonnement, nausée…",
                key="pain_note",
            )
            pain_clicked = st.form_submit_button(
                "Enregistrer la douleur", icon=":material/add:", width="stretch"
            )
        if pain_clicked:
            meal_id, delay = save_symptom_event(
                "douleur",
                pain_date,
                pain_time,
                pain_intensity,
                pain_note,
                meals,
                pain_type,
            )
            if meal_id is None:
                st.success("Douleur enregistrée, sans repas précédent associé.")
            else:
                st.success(
                    f"Douleur enregistrée et reliée au repas précédent "
                    f"({format_delay(delay)} après)."
                )
            symptom_events = load_symptom_events()

    st.warning(
        "Une douleur soudaine ou intense, une douleur thoracique, un ventre dur, "
        "des vomissements sanglants, des selles sanglantes ou noires, un malaise "
        "ou une difficulté à respirer nécessitent une aide médicale urgente. "
        "Ce journal ne pose aucun diagnostic.",
        icon="⚕️",
    )

    if symptom_events:
        hunger_count = sum(event["kind"] == "faim" for event in symptom_events)
        pain_count = sum(event["kind"] == "douleur" for event in symptom_events)
        symptom_metrics = st.columns(3, border=True)
        symptom_metrics[0].metric("Faim enregistrée", hunger_count)
        symptom_metrics[1].metric("Douleurs enregistrées", pain_count)
        symptom_metrics[2].metric("Événements au total", len(symptom_events))

        symptom_frame = pd.DataFrame(symptom_events)
        symptom_frame["Date et heure"] = pd.to_datetime(
            symptom_frame["occurred_at"], errors="coerce", utc=True
        ).dt.tz_convert("America/Argentina/Buenos_Aires")
        symptom_frame["Sensation"] = symptom_frame.apply(
            lambda row: (
                "Faim"
                if row["kind"] == "faim"
                else row["pain_type"] or "Douleur abdominale"
            ),
            axis=1,
        )
        symptom_chart = (
            alt.Chart(symptom_frame.dropna(subset=["Date et heure"]))
            .mark_circle(size=115, opacity=0.8)
            .encode(
                x=alt.X("Date et heure:T", title="Date et heure"),
                y=alt.Y(
                    "intensity:Q",
                    title="Intensité / 10",
                    scale=alt.Scale(domain=[0, 10]),
                ),
                color=alt.Color("Sensation:N", title="Sensation"),
                tooltip=[
                    alt.Tooltip("Date et heure:T", title="Moment"),
                    alt.Tooltip("Sensation:N"),
                    alt.Tooltip("intensity:Q", title="Intensité"),
                    alt.Tooltip("meal_description:N", title="Repas précédent"),
                ],
            )
            .properties(height=280)
        )
        st.altair_chart(symptom_chart)

        symptom_rows = [
            {
                "Date": format_datetime(event["occurred_at"]),
                "Sensation": (
                    "Faim"
                    if event["kind"] == "faim"
                    else event["pain_type"] or "Douleur abdominale"
                ),
                "Intensité": f"{event['intensity']}/10",
                "Repas précédent": event.get("meal_description") or "—",
                "Délai": format_delay(event.get("delay_minutes")),
                "Origine rapportée": event.get("reported_trigger") or "—",
                "Saisie": (
                    "Manuelle"
                    if event.get("extraction_confidence") == "manuelle"
                    else f"IA · {event.get('extraction_confidence') or '—'}"
                ),
                "Description": event.get("note") or "—",
            }
            for event in symptom_events
        ]
        st.dataframe(
            pd.DataFrame(symptom_rows),
            hide_index=True,
            column_config={
                "Repas précédent": st.column_config.TextColumn(width="large"),
                "Origine rapportée": st.column_config.TextColumn(width="medium"),
                "Description": st.column_config.TextColumn(width="large"),
            },
        )
    else:
        st.info("Aucune sensation de faim ou douleur enregistrée pour le moment.")


with bowel_tab:
    bowel_events = load_bowel_events()
    st.subheader("Journal des selles")
    st.caption(
        "Notez le moment et la forme. L'échelle de Bristol décrit l'aspect de 1 "
        "(très dur) à 7 (liquide) ; elle aide à observer une tendance, sans poser "
        "de diagnostic."
    )

    with st.form("natural_bowel_form", clear_on_submit=True, border=True):
        st.markdown("#### Décrire naturellement")
        natural_bowel_text = st.text_area(
            "Jour, heure et forme des selles",
            placeholder=(
                "Ex. : Ce matin à 8 h 30, selles lisses et souples, en forme de "
                "saucisse, sans difficulté."
            ),
            height=110,
            key="natural_bowel_text",
        )
        natural_bowel_clicked = st.form_submit_button(
            "Interpréter et enregistrer",
            type="primary",
            icon=":material/auto_awesome:",
            width="stretch",
        )

    if natural_bowel_clicked:
        if not natural_bowel_text.strip():
            st.warning("Décrivez au moins le moment et la forme des selles.")
        elif not ai_settings.has_any_key:
            st.error("Ajoutez une clé DeepSeek, Gemini ou Groq dans la barre latérale.")
        else:
            with st.spinner("Interprétation du moment et de la forme…"):
                try:
                    extraction = parse_bowel_text(ai_settings, natural_bowel_text)
                    saved_count, questions = save_ready_bowel_events(
                        extraction["events"], natural_bowel_text
                    )
                    if saved_count:
                        st.success(
                            f"{saved_count} passage"
                            f"{'s' if saved_count > 1 else ''} enregistré"
                            f"{'s' if saved_count > 1 else ''}."
                        )
                        st.caption(
                            "Interprétation réalisée par "
                            f"{extraction.get('_provider', 'IA cloud')}."
                        )
                        bowel_events = load_bowel_events()
                    if questions:
                        st.warning(
                            "Précision nécessaire avant l'enregistrement :\n\n- "
                            + "\n- ".join(questions)
                        )
                    if not extraction["events"]:
                        st.info("Aucune selle identifiable n'a été trouvée dans ce texte.")
                except Exception as error:
                    st.error(
                        "Interprétation impossible. Vérifiez la connexion IA, puis "
                        f"réessayez.\n\nDétail : {safe_error_message(error, ai_settings)}"
                    )

    with st.expander("Saisie manuelle et échelle de Bristol"):
        with st.form("manual_bowel_form", clear_on_submit=True, border=False):
            bowel_date_col, bowel_time_col = st.columns(2)
            bowel_date = bowel_date_col.date_input(
                "Date", value=date.today(), key="bowel_date"
            )
            bowel_time = bowel_time_col.time_input(
                "Heure", value="now", step=300, key="bowel_time"
            )
            bowel_type = st.selectbox(
                "Forme selon l'échelle de Bristol",
                list(BRISTOL_TYPES),
                index=3,
                format_func=lambda value: f"Type {value} — {BRISTOL_TYPES[value]}",
                key="bowel_type",
            )
            bowel_note = st.text_area(
                "Précisions facultatives",
                placeholder="Ex. : difficulté, urgence, couleur inhabituelle…",
                key="bowel_note",
            )
            bowel_clicked = st.form_submit_button(
                "Enregistrer le passage",
                icon=":material/add:",
                width="stretch",
            )
        if bowel_clicked:
            save_bowel_event(
                bowel_date,
                bowel_time,
                int(bowel_type),
                bowel_note,
            )
            bowel_events = load_bowel_events()
            st.success("Passage enregistré localement.")

        bristol_guide = pd.DataFrame(
            [
                {"Type": number, "Description": description}
                for number, description in BRISTOL_TYPES.items()
            ]
        )
        st.dataframe(bristol_guide, hide_index=True, width="stretch")

    st.warning(
        "Du sang, des selles noires, une douleur importante, de la fièvre, une "
        "déshydratation ou un changement persistant inquiétant justifient un avis "
        "médical. Ce journal ne remplace pas une consultation.",
        icon="⚕️",
    )

    if bowel_events:
        bowel_frame = pd.DataFrame(bowel_events)
        bowel_frame["Date et heure"] = pd.to_datetime(
            bowel_frame["occurred_at"], errors="coerce", utc=True
        ).dt.tz_convert("America/Argentina/Buenos_Aires")
        bowel_frame["Catégorie"] = bowel_frame["bristol_type"].map(
            lambda value: (
                "Dure (1–2)"
                if int(value) <= 2
                else "Formée (3–4)"
                if int(value) <= 4
                else "Molle/liquide (5–7)"
            )
        )
        cutoff_14_days = pd.Timestamp.now(tz="America/Argentina/Buenos_Aires") - pd.Timedelta(days=14)
        recent_bowel_frame = bowel_frame[
            bowel_frame["Date et heure"] >= cutoff_14_days
        ]
        bowel_metrics = st.columns(3, border=True)
        bowel_metrics[0].metric("Passages enregistrés", len(bowel_events))
        bowel_metrics[1].metric("Sur 14 jours", len(recent_bowel_frame))
        bowel_metrics[2].metric(
            "Types 3–4 sur 14 jours",
            (
                f"{recent_bowel_frame['bristol_type'].between(3, 4).mean() * 100:.0f}%"
                if not recent_bowel_frame.empty
                else "—"
            ),
        )

        bowel_chart = (
            alt.Chart(bowel_frame.dropna(subset=["Date et heure"]))
            .mark_circle(size=125, opacity=0.85)
            .encode(
                x=alt.X("Date et heure:T", title="Date et heure"),
                y=alt.Y(
                    "bristol_type:Q",
                    title="Type de Bristol",
                    scale=alt.Scale(domain=[0.5, 7.5]),
                    axis=alt.Axis(tickMinStep=1),
                ),
                color=alt.Color(
                    "Catégorie:N",
                    title=None,
                    scale=alt.Scale(
                        domain=["Dure (1–2)", "Formée (3–4)", "Molle/liquide (5–7)"],
                        range=["#f59e0b", "#22c55e", "#3b82f6"],
                    ),
                ),
                tooltip=[
                    alt.Tooltip("Date et heure:T", title="Moment"),
                    alt.Tooltip("bristol_type:Q", title="Type"),
                    alt.Tooltip("description:N", title="Description"),
                ],
            )
            .properties(height=275)
        )
        st.altair_chart(bowel_chart)

        bowel_rows = [
            {
                "Date": format_datetime(event["occurred_at"]),
                "Type": f"Bristol {event['bristol_type']}",
                "Forme": BRISTOL_TYPES[int(event["bristol_type"])],
                "Précisions": event.get("description") or "—",
                "Saisie": (
                    "Manuelle"
                    if event.get("extraction_confidence") == "manuelle"
                    else f"IA · {event.get('extraction_confidence') or '—'}"
                ),
            }
            for event in bowel_events
        ]
        st.dataframe(
            pd.DataFrame(bowel_rows),
            hide_index=True,
            width="stretch",
            column_config={
                "Forme": st.column_config.TextColumn(width="large"),
                "Précisions": st.column_config.TextColumn(width="large"),
            },
        )
    else:
        st.info("Aucun passage enregistré pour le moment.")


with activity_tab:
    activity_events = load_activity_events()
    activity_profile = load_profile()
    activity_weights = load_weights()
    activity_weight = float(
        activity_weights[-1]["weight_kg"] if activity_weights else 70.0
    )
    st.subheader("Activité physique et dépense énergétique")
    st.caption(
        "Consignez la natation en minutes, la marche en pas et le vélo en "
        "kilomètres. La dépense est estimée à partir de votre dernière pesée "
        f"({activity_weight:.1f} kg) et reste un ordre de grandeur."
    )

    activity_type = st.segmented_control(
        "Type d'activité",
        ACTIVITY_TYPES,
        default="Natation",
        selection_mode="single",
        key="activity_type_selector",
    )

    with st.form("activity_form", clear_on_submit=True, border=True):
        activity_header = st.columns(2)
        activity_date = activity_header[0].date_input(
            "Date", value=date.today(), key="activity_date"
        )
        activity_intensity = activity_header[1].selectbox(
            "Intensité ressentie",
            ACTIVITY_INTENSITIES,
            index=1,
            key="activity_intensity",
        )
        duration_minutes = 0.0
        activity_steps = 0
        distance_km = 0.0
        if activity_type == "Natation":
            duration_minutes = st.number_input(
                "Durée de nage (minutes)",
                min_value=1.0,
                max_value=600.0,
                value=60.0,
                step=5.0,
                key="swim_duration",
            )
        elif activity_type == "Marche":
            activity_steps = st.number_input(
                "Nombre de pas",
                min_value=1,
                max_value=100000,
                value=7000,
                step=500,
                key="walking_steps",
            )
        else:
            cycling_inputs = st.columns(2)
            distance_km = cycling_inputs[0].number_input(
                "Distance à vélo (km)",
                min_value=0.1,
                max_value=500.0,
                value=7.0,
                step=0.5,
                key="cycling_distance",
            )
            duration_minutes = cycling_inputs[1].number_input(
                "Durée (minutes, facultative)",
                min_value=0.0,
                max_value=600.0,
                value=0.0,
                step=5.0,
                key="cycling_duration",
                help="Avec une durée, l'estimation utilise une formule MET.",
            )
        activity_note = st.text_input(
            "Précision facultative",
            placeholder="Ex. : bassin de 25 m, trajet vallonné, marche tranquille…",
            key="activity_note",
        )
        activity_clicked = st.form_submit_button(
            "Enregistrer l'activité",
            type="primary",
            icon=":material/add:",
            width="stretch",
        )

    if activity_clicked:
        try:
            saved_activity_calories = save_activity_event(
                activity_date,
                activity_type,
                activity_weight,
                activity_intensity,
                duration_minutes,
                int(activity_steps),
                distance_km,
                activity_note,
            )
        except ValueError as error:
            st.error(str(error))
        else:
            st.success(
                f"Activité enregistrée · dépense estimée : "
                f"{saved_activity_calories:.0f} kcal."
            )
            activity_events = load_activity_events()

    energy_balance = build_daily_energy_balance(
        load_meals(), activity_events, activity_profile
    )
    today_timestamp = pd.Timestamp(date.today())
    today_energy = (
        energy_balance[energy_balance["Date"] == today_timestamp]
        if not energy_balance.empty
        else pd.DataFrame()
    )
    today_intake = (
        float(today_energy.iloc[0]["Calories estimées"])
        if not today_energy.empty
        else 0.0
    )
    today_activity = (
        float(today_energy.iloc[0]["Dépense activité"])
        if not today_energy.empty
        else 0.0
    )
    today_net = today_intake - today_activity
    food_goal = float(
        activity_profile.get("calorie_goal_kcal") or DEFAULT_CALORIE_GOAL_KCAL
    )
    with st.container(horizontal=True, horizontal_alignment="distribute"):
        st.metric(
            "Apports consignés",
            f"{today_intake:.0f} kcal",
            border=True,
            icon=":material/restaurant:",
        )
        st.metric(
            "Activité enregistrée",
            f"{today_activity:.0f} kcal",
            border=True,
            icon=":material/directions_run:",
        )
        st.metric(
            "Apport net indicatif",
            f"{today_net:.0f} kcal",
            border=True,
            icon=":material/balance:",
        )
        st.metric(
            "Écart net au repère",
            f"{today_net - food_goal:+.0f} kcal",
            border=True,
            icon=":material/compare_arrows:",
        )

    st.info(
        "Le repère alimentaire n'est pas augmenté automatiquement après le "
        "sport. Le solde net sert à observer les tendances. Votre niveau "
        "d'activité habituel peut déjà intégrer une partie de ces séances : "
        "évitez d'utiliser ce chiffre comme une autorisation à compenser toutes "
        "les calories dépensées.",
        icon=":material/info:",
    )

    if energy_balance.empty:
        st.info("Le graphique apparaîtra après un repas ou une activité enregistrée.")
    else:
        energy_chart_data = energy_balance.tail(14).copy()
        energy_chart_data["Jour"] = energy_chart_data["Date"].dt.strftime("%d/%m")
        energy_chart_data["Apport net affiché"] = energy_chart_data[
            "Apport net indicatif"
        ].clip(lower=0)
        energy_bars = (
            alt.Chart(energy_chart_data)
            .mark_bar(color="#0f766e", cornerRadiusTopLeft=6, cornerRadiusTopRight=6)
            .encode(
                x=alt.X("Jour:O", title=None, sort=None),
                y=alt.Y("Apport net affiché:Q", title="Apport net estimé (kcal)"),
                tooltip=[
                    alt.Tooltip("Date:T", title="Date", format="%d/%m/%Y"),
                    alt.Tooltip("Calories estimées:Q", title="Alimentation", format=".0f"),
                    alt.Tooltip("Dépense activité:Q", title="Activité", format=".0f"),
                    alt.Tooltip("Apport net indicatif:Q", title="Net", format=".0f"),
                    alt.Tooltip("Repère alimentaire:Q", title="Repère", format=".0f"),
                ],
            )
        )
        energy_goal_line = (
            alt.Chart(pd.DataFrame({"Repère": [food_goal]}))
            .mark_rule(color="#f97316", strokeDash=[7, 5], strokeWidth=3)
            .encode(y="Repère:Q")
        )
        st.altair_chart((energy_bars + energy_goal_line).properties(height=320))
        st.caption(
            "Barres vertes : calories alimentaires consignées moins dépense "
            "sportive estimée · ligne orange : repère alimentaire quotidien. "
            "Survolez une barre pour voir les trois valeurs séparément."
        )

    if activity_events:
        st.markdown("#### Activités enregistrées")
        activity_rows = []
        for event in activity_events:
            if event["activity_type"] == "Natation":
                quantity = f"{float(event['duration_minutes']):g} min"
            elif event["activity_type"] == "Marche":
                quantity = f"{int(event['steps']):,} pas".replace(",", " ")
            else:
                quantity = f"{float(event['distance_km']):g} km"
                if float(event.get("duration_minutes") or 0) > 0:
                    quantity += f" · {float(event['duration_minutes']):g} min"
            activity_rows.append(
                {
                    "Date": date.fromisoformat(event["occurred_on"]).strftime("%d/%m/%Y"),
                    "Activité": event["activity_type"],
                    "Quantité": quantity,
                    "Intensité": event["intensity"],
                    "Dépense estimée": event["calories_kcal"],
                    "Précision": event.get("note") or "—",
                }
            )
        st.dataframe(
            pd.DataFrame(activity_rows),
            hide_index=True,
            width="stretch",
            column_config={
                "Dépense estimée": st.column_config.NumberColumn(
                    format="%.0f kcal"
                ),
                "Précision": st.column_config.TextColumn(width="large"),
            },
        )


with food_tab:
    favorite_foods = load_favorite_foods()
    food_by_id = {int(food["id"]): food for food in favorite_foods}
    st.subheader("Ma bibliothèque d'aliments")
    st.caption(
        "Recherchez un produit ou faites lire son étiquette, vérifiez les valeurs, "
        "puis enregistrez-le. La fiche locale sera prioritaire lors des prochains repas."
    )
    if food_notice := st.session_state.pop("favorite_food_notice", ""):
        st.success(food_notice)
    if st.session_state.pop("favorite_food_prefilled", False):
        st.info(
            "Les champs ci-dessous ont été préremplis. Vérifiez-les avec le paquet, "
            "puis cliquez sur « Enregistrer la fiche »."
        )

    st.markdown("#### 1. Trouver les valeurs")
    lookup_mode = st.segmented_control(
        "Méthode",
        ["Rechercher le produit", "Faire lire l'étiquette"],
        default="Rechercher le produit",
        key="favorite_food_lookup_mode",
    )
    if lookup_mode == "Faire lire l'étiquette":
        st.caption(
            "Recopiez le nom, la marque, le poids de référence et les valeurs du "
            "tableau. DeepSeek les convertira pour 100 g sans en inventer."
        )
        with st.form("food_label_reader_form", border=True):
            label_text = st.text_area(
                "Texte de l'étiquette",
                height=130,
                placeholder=(
                    "Ex. : Yaourt Tregar nature, pot 150 g. Pour 30 g : "
                    "7,2 g protéines, 11 g lipides, 0 g glucides."
                ),
                key="favorite_food_label_text",
            )
            label_clicked = st.form_submit_button(
                "Lire et convertir l'étiquette",
                icon=":material/document_scanner:",
                type="primary",
                width="stretch",
            )
        if label_clicked:
            if not label_text.strip():
                st.warning("Recopiez d'abord les informations de l'étiquette.")
            elif not ai_settings.has_any_key:
                st.warning("Ajoutez une clé IA dans la barre latérale pour lire l'étiquette.")
            else:
                try:
                    with st.spinner("Lecture et conversion des valeurs pour 100 g…"):
                        label_candidate = parse_food_label(ai_settings, label_text)
                except Exception as error:
                    st.error("L'étiquette n'a pas pu être interprétée.")
                    st.caption(safe_error_message(error, ai_settings))
                else:
                    st.session_state["favorite_food_candidate"] = label_candidate
                    st.session_state["favorite_food_candidate_message"] = (
                        f"Étiquette interprétée par {label_candidate['_provider']} · "
                        f"{label_candidate['_model_name']}."
                    )
    else:
        st.caption(
            "Utilisez de préférence le code-barres. La recherche consulte Open Food "
            "Facts pour les produits vendus en Argentine."
        )
        with st.form("favorite_food_search_form", border=True):
            food_search_query = st.text_input(
                "Nom exact, marque ou code-barres",
                placeholder="Ex. : Milkaut yogur natural ou 779…",
                key="favorite_food_search_query",
            )
            search_clicked = st.form_submit_button(
                "Rechercher les valeurs",
                icon=":material/search:",
                type="primary",
                width="stretch",
            )
        if search_clicked:
            st.session_state["favorite_food_candidate"] = None
            if not food_search_query.strip():
                st.warning("Indiquez un nom de produit ou un code-barres.")
            else:
                try:
                    with st.spinner("Recherche de la fiche nutritionnelle…"):
                        product_results = find_argentine_product(food_search_query)
                except Exception as error:
                    st.error("La base de produits n'est pas joignable pour le moment.")
                    st.caption(safe_error_message(error, ai_settings))
                else:
                    if product_results:
                        st.session_state["favorite_food_candidate"] = product_results[0]
                        st.session_state["favorite_food_candidate_message"] = (
                            "Fiche trouvée dans Open Food Facts. Vérifiez qu'il "
                            "s'agit bien de la même variante."
                        )
                    else:
                        st.session_state["favorite_food_candidate_message"] = (
                            "Aucune fiche suffisamment proche n'a été trouvée. "
                            "Essayez le code-barres ou utilisez « Faire lire "
                            "l'étiquette »."
                        )

    candidate_message = str(
        st.session_state.get("favorite_food_candidate_message") or ""
    )
    candidate = st.session_state.get("favorite_food_candidate")
    if candidate_message:
        if candidate:
            st.success(candidate_message)
        else:
            st.info(candidate_message)
    if isinstance(candidate, dict):
        nutrition = candidate.get("nutrition") or {}
        preview_carbs = float(nutrition.get("glucides_100g") or 0)
        preview_protein = float(nutrition.get("proteines_100g") or 0)
        preview_fat = float(nutrition.get("lipides_100g") or 0)
        preview_calories = float(nutrition.get("energie_kcal_100g") or 0)
        calories_calculated = preview_calories <= 0 and any(
            (preview_carbs, preview_protein, preview_fat)
        )
        if calories_calculated:
            preview_calories = 4 * preview_carbs + 4 * preview_protein + 9 * preview_fat
        candidate_source = str(candidate.get("source") or "saisie_manuelle")
        can_import = candidate_source == "open_food_facts" or bool(
            candidate.get("donnees_explicites")
        )
        with st.container(border=True):
            st.markdown(
                f"**{candidate.get('marque') or 'Sans marque'} · "
                f"{candidate.get('nom') or 'Produit'}**"
            )
            st.caption(
                f"Source : {food_source_label(candidate_source)} · "
                f"confiance {candidate.get('correspondance') or 'non précisée'}"
            )
            preview_columns = st.columns(5)
            preview_columns[0].metric(
                "Calories", f"{preview_calories:g} kcal"
            )
            preview_columns[1].metric(
                "Glucides", f"{preview_carbs:g} g"
            )
            preview_columns[2].metric(
                "Protéines", f"{preview_protein:g} g"
            )
            preview_columns[3].metric(
                "Lipides", f"{preview_fat:g} g"
            )
            preview_columns[4].metric(
                "Fibres", f"{float(nutrition.get('fibres_100g') or 0):g} g"
            )
            st.caption(
                "Valeurs affichées pour 100 g."
                + (
                    " Les calories sont calculées à partir des macronutriments."
                    if calories_calculated
                    else ""
                )
            )
            if candidate.get("champs_manquants"):
                st.warning(
                    "Valeurs absentes de l'étiquette : "
                    + ", ".join(candidate["champs_manquants"])
                    + ". Vous pourrez les laisser à 0 ou les compléter manuellement."
                )
            if candidate.get("url_source"):
                st.markdown(
                    f"[Ouvrir la fiche source ↗]({candidate['url_source']})"
                )
            st.button(
                "Utiliser ces valeurs dans une nouvelle fiche",
                icon=":material/arrow_downward:",
                type="primary",
                width="stretch",
                disabled=not can_import,
                on_click=apply_food_candidate_to_form,
                args=(candidate,),
                key="apply_favorite_food_candidate",
            )
            if not can_import:
                st.caption(
                    "Aucune base exploitable n'a été reconnue. Ajoutez le poids de "
                    "la portion et au moins une valeur nutritionnelle."
                )

    st.markdown("#### 2. Vérifier et enregistrer")
    food_choice = st.selectbox(
        "Créer ou modifier une fiche",
        [0, *food_by_id],
        format_func=lambda value: (
            "Nouvel aliment"
            if int(value) == 0
            else favorite_food_label(food_by_id[int(value)])
        ),
        key="favorite_food_choice",
    )
    selected_food = food_by_id.get(int(food_choice), {})
    form_suffix = str(int(food_choice))
    data_source = str(
        st.session_state.get(
            f"favorite_data_source_{form_suffix}",
            selected_food.get("data_source") or "saisie_manuelle",
        )
    )
    barcode = str(
        st.session_state.get(
            f"favorite_barcode_{form_suffix}", selected_food.get("barcode") or ""
        )
    )
    source_url = str(
        st.session_state.get(
            f"favorite_source_url_{form_suffix}",
            selected_food.get("source_url") or "",
        )
    )
    with st.form(f"favorite_food_form_{form_suffix}", border=True):
        identity_columns = st.columns(2)
        food_name = identity_columns[0].text_input(
            "Nom de l'aliment",
            value=str(selected_food.get("name") or ""),
            placeholder="Ex. : Yaourt grec nature",
            key=f"favorite_name_{form_suffix}",
        )
        food_brand = identity_columns[1].text_input(
            "Marque",
            value=str(selected_food.get("brand") or ""),
            placeholder="Ex. : La Serenísima",
            key=f"favorite_brand_{form_suffix}",
        )
        food_aliases = st.text_input(
            "Autres noms pour le reconnaître",
            value=str(selected_food.get("aliases") or ""),
            placeholder="Ex. : yogurísimo, yaourt serenissima",
            help="Séparez plusieurs noms par une virgule.",
            key=f"favorite_aliases_{form_suffix}",
        )
        st.markdown("##### Valeurs pour 100 g")
        nutrition_columns = st.columns(3)
        food_calories = nutrition_columns[0].number_input(
            "Calories",
            min_value=0.0,
            max_value=1000.0,
            value=float(selected_food.get("calories_100g") or 0),
            step=1.0,
            key=f"favorite_calories_{form_suffix}",
            help="Si vous laissez 0, l'application les calcule à partir des macros.",
        )
        food_carbs = nutrition_columns[1].number_input(
            "Glucides (g)",
            min_value=0.0,
            max_value=100.0,
            value=float(selected_food.get("carbs_100g") or 0),
            step=0.1,
            key=f"favorite_carbs_{form_suffix}",
        )
        food_protein = nutrition_columns[2].number_input(
            "Protéines (g)",
            min_value=0.0,
            max_value=100.0,
            value=float(selected_food.get("protein_100g") or 0),
            step=0.1,
            key=f"favorite_protein_{form_suffix}",
        )
        nutrition_columns_2 = st.columns(3)
        food_fat = nutrition_columns_2[0].number_input(
            "Lipides (g)",
            min_value=0.0,
            max_value=100.0,
            value=float(selected_food.get("fat_100g") or 0),
            step=0.1,
            key=f"favorite_fat_{form_suffix}",
        )
        food_fiber = nutrition_columns_2[1].number_input(
            "Fibres (g)",
            min_value=0.0,
            max_value=100.0,
            value=float(selected_food.get("fiber_100g") or 0),
            step=0.1,
            key=f"favorite_fiber_{form_suffix}",
        )
        food_serving = nutrition_columns_2[2].number_input(
            "Portion habituelle (g)",
            min_value=0.0,
            max_value=2000.0,
            value=float(selected_food.get("serving_size_g") or 0),
            step=5.0,
            key=f"favorite_serving_{form_suffix}",
        )
        food_notes = st.text_input(
            "Précision facultative",
            value=str(selected_food.get("notes") or ""),
            placeholder="Ex. : chocolat 80 %, fromage en tranches, pot de 190 g…",
            key=f"favorite_notes_{form_suffix}",
        )
        st.caption(
            f"Source des valeurs : {food_source_label(data_source)}"
            + (f" · code-barres {barcode}" if barcode else "")
        )
        food_save_clicked = st.form_submit_button(
            "Enregistrer la fiche",
            type="primary",
            icon=":material/save:",
            width="stretch",
        )

    if food_save_clicked:
        try:
            save_favorite_food(
                food_name,
                food_brand,
                food_aliases,
                food_serving,
                food_calories,
                food_carbs,
                food_protein,
                food_fat,
                food_fiber,
                food_notes,
                str(load_profile().get("country") or DEFAULT_COUNTRY),
                int(food_choice) if int(food_choice) else None,
                data_source=data_source,
                barcode=barcode,
                source_url=source_url,
            )
        except (ValueError, sqlite3.IntegrityError) as error:
            st.error(f"Fiche non enregistrée : {error}")
        else:
            st.session_state["favorite_food_notice"] = (
                "Fiche alimentaire enregistrée localement. Elle peut maintenant "
                "être associée aux repas."
            )
            st.rerun()

    favorite_foods = load_favorite_foods()
    if favorite_foods:
        st.markdown("#### 3. Aliments disponibles pour les calculs")
        food_rows = [
            {
                "Aliment": food["name"],
                "Marque": food.get("brand") or "—",
                "Portion": food.get("serving_size_g") or None,
                "Calories": food["calories_100g"],
                "Glucides": food["carbs_100g"],
                "Protéines": food["protein_100g"],
                "Lipides": food["fat_100g"],
                "Fibres": food["fiber_100g"],
                "Source": food_source_label(food.get("data_source") or ""),
                "Autres noms": food.get("aliases") or "—",
            }
            for food in favorite_foods
        ]
        st.dataframe(
            pd.DataFrame(food_rows),
            hide_index=True,
            width="stretch",
            column_config={
                "Aliment": st.column_config.TextColumn(pinned=True),
                "Portion": st.column_config.NumberColumn(format="%.0f g"),
                "Calories": st.column_config.NumberColumn(format="%.0f kcal/100 g"),
                "Glucides": st.column_config.NumberColumn(format="%.1f g/100 g"),
                "Protéines": st.column_config.NumberColumn(format="%.1f g/100 g"),
                "Lipides": st.column_config.NumberColumn(format="%.1f g/100 g"),
                "Fibres": st.column_config.NumberColumn(format="%.1f g/100 g"),
                "Source": st.column_config.TextColumn(width="medium"),
                "Autres noms": st.column_config.TextColumn(width="large"),
            },
        )
    else:
        st.info(
            "Aucun aliment personnel enregistré. Commencez par un produit que "
            "vous consommez souvent et recopiez les valeurs de son étiquette."
        )


with assistant_tab:
    st.subheader("Échanger avec votre nutritionniste IA")
    st.caption(
        "Posez une question sur vos repas, vos aliments récurrents, vos activités, "
        "votre poids ou vos tendances récentes. L'assistant adopte la méthode "
        "d'un nutritionniste : il analyse les portions, l'équilibre des repas, la "
        "satiété, la dépense estimée et la progression dans le temps. Il utilise "
        "le profil et une sélection limitée du journal pour contextualiser sa réponse."
    )
    st.info(
        "Cet assistant IA adopte une approche de nutritionniste, mais n'est pas "
        "un professionnel diplômé. Ses réponses sont éducatives, reposent sur "
        "des estimations et ne remplacent pas un suivi diététique ou médical "
        "personnalisé.",
        icon=":material/health_and_safety:",
    )

    st.session_state.setdefault("coach_chat_messages", [])
    chat_messages: list[dict[str, Any]] = st.session_state[
        "coach_chat_messages"
    ]
    with st.container(
        horizontal=True,
        horizontal_alignment="right",
    ):
        if st.button(
            "Nouvelle conversation",
            icon=":material/refresh:",
            disabled=not bool(chat_messages),
            key="clear_coach_chat",
        ):
            st.session_state["coach_chat_messages"] = []
            st.rerun()

    suggested_questions = {
        "Bilan récent": (
            "Analyse mes repas des sept derniers jours comme le ferait un "
            "nutritionniste et indique-moi les deux priorités principales."
        ),
        "Objectifs": (
            "Mes portions et l'équilibre de mes repas semblent-ils cohérents "
            "avec mon objectif de perte de poids progressive ?"
        ),
        "Prochaine étape": (
            "Quel ajustement nutritionnel simple et réaliste puis-je tester "
            "cette semaine ?"
        ),
        "Sport & énergie": (
            "Mets en relation mes activités récentes, mes apports consignés, ma "
            "faim et mon objectif, sans compter deux fois la dépense sportive."
        ),
    }
    suggested_question: str | None = None
    if not chat_messages:
        with st.container(border=True):
            st.markdown(
                "**Vous pouvez demander, par exemple :** une lecture de la "
                "semaine, une explication d'un écart ou une idée simple pour le "
                "prochain repas. L'assistant signalera clairement les données "
                "manquantes au lieu de les inventer."
            )
        if ai_settings.has_any_key:
            selected_suggestion = st.pills(
                "Questions proposées",
                list(suggested_questions),
                selection_mode="single",
                label_visibility="collapsed",
                key="coach_chat_suggestions",
            )
            if selected_suggestion:
                suggested_question = suggested_questions[selected_suggestion]

    conversation_view = (
        st.container(height=520, border=True)
        if chat_messages
        else st.container()
    )
    with conversation_view:
        for message in chat_messages:
            role = str(message.get("role") or "assistant")
            with st.chat_message(role):
                st.markdown(str(message.get("content") or ""))
                if message.get("medical_attention") and message.get(
                    "safety_message"
                ):
                    st.warning(str(message["safety_message"]))
                suggestions = message.get("suggestions") or []
                if suggestions:
                    st.markdown(
                        "**Pour poursuivre :**\n"
                        + "\n".join(f"- {item}" for item in suggestions)
                    )
                if message.get("provider"):
                    st.caption(
                        f"Réponse générée par {message['provider']} · "
                        f"{message.get('model_name') or 'modèle cloud'}"
                    )

    question = st.chat_input(
        (
            "Posez votre question au nutritionniste IA…"
            if ai_settings.has_any_key
            else "Ajoutez une clé IA dans la barre latérale pour discuter."
        ),
        key="coach_chat_input",
        max_chars=1500,
        disabled=not ai_settings.has_any_key,
        submit_mode="disable",
    )
    submitted_question = (question or suggested_question or "").strip()
    if submitted_question:
        history_before_question = list(chat_messages)
        user_message = {"role": "user", "content": submitted_question}
        chat_messages.append(user_message)
        with conversation_view:
            with st.chat_message("user"):
                st.markdown(submitted_question)

            with st.chat_message("assistant"):
                with st.spinner(
                    "Le nutritionniste IA analyse vos données récentes…"
                ):
                    try:
                        chat_response = answer_coach_question(
                            ai_settings,
                            submitted_question,
                            history_before_question,
                            load_meals(),
                            load_weights(),
                            load_profile(),
                            load_symptom_events(),
                            load_bowel_events(),
                            load_activity_events(),
                            load_favorite_foods(),
                        )
                    except Exception as error:
                        chat_messages.pop()
                        st.error(
                            "Le coach n'a pas pu répondre. Vérifiez la connexion "
                            "et les crédits du fournisseur, puis réessayez."
                        )
                        st.caption(safe_error_message(error, ai_settings))
                    else:
                        st.markdown(chat_response["content"])
                        if chat_response["medical_attention"]:
                            st.warning(chat_response["safety_message"])
                        if chat_response["suggestions"]:
                            st.markdown(
                                "**Pour poursuivre :**\n"
                                + "\n".join(
                                    f"- {item}"
                                    for item in chat_response["suggestions"]
                                )
                            )
                        st.caption(
                            f"Réponse générée par {chat_response['provider']} · "
                            f"{chat_response['model_name']}"
                        )
                        chat_messages.append(
                            {
                                "role": "assistant",
                                **chat_response,
                            }
                        )


with coach_tab:
    meals = load_meals()
    weights = load_weights()
    symptom_events = load_symptom_events()
    bowel_events = load_bowel_events()
    profile = load_profile()
    latest_weight_kg = min(
        max(float(weights[-1]["weight_kg"]) if weights else 70.0, 20.0),
        400.0,
    )
    if profile.get("auto_macro_targets"):
        current_targets = calculate_macro_targets(
            latest_weight_kg,
            float(profile.get("height_cm") or DEFAULT_HEIGHT_CM),
            profile["objective"],
            int(profile.get("age_years") or DEFAULT_AGE_YEARS),
            str(profile.get("sex") or DEFAULT_SEX),
            str(profile.get("activity_level") or DEFAULT_ACTIVITY_LEVEL),
            float(profile.get("target_weight_kg") or DEFAULT_TARGET_WEIGHT_KG),
        )
        profile["carbs_goal_g"] = current_targets["carbs_g"]
        profile["protein_goal_g"] = current_targets["protein_g"]
        profile["fat_goal_g"] = current_targets["fat_g"]
        profile["calorie_goal_kcal"] = current_targets["estimated_energy_kcal"]

    strategy_targets = calculate_macro_targets(
        latest_weight_kg,
        float(profile.get("height_cm") or DEFAULT_HEIGHT_CM),
        profile["objective"],
        int(profile.get("age_years") or DEFAULT_AGE_YEARS),
        str(profile.get("sex") or DEFAULT_SEX),
        str(profile.get("activity_level") or DEFAULT_ACTIVITY_LEVEL),
        float(profile.get("target_weight_kg") or DEFAULT_TARGET_WEIGHT_KG),
    )
    target_weight_value = float(
        profile.get("target_weight_kg") or DEFAULT_TARGET_WEIGHT_KG
    )
    stabilization_targets = calculate_macro_targets(
        target_weight_value,
        float(profile.get("height_cm") or DEFAULT_HEIGHT_CM),
        "Perdre du poids progressivement",
        int(profile.get("age_years") or DEFAULT_AGE_YEARS),
        str(profile.get("sex") or DEFAULT_SEX),
        str(profile.get("activity_level") or DEFAULT_ACTIVITY_LEVEL),
        target_weight_value,
    )

    with st.container(border=True):
        st.markdown("#### Votre stratégie à long terme")
        strategy_columns = st.columns(4)
        short_phase = {
            "Perte progressive": "Perte",
            "Transition vers la stabilisation": "Transition",
            "Stabilisation": "Maintien",
            "Repère stable": "Maintien",
            "Prise progressive": "Prise",
        }.get(strategy_targets["plan_phase"], strategy_targets["plan_phase"])
        strategy_columns[0].metric("Phase actuelle", short_phase)
        strategy_columns[1].metric(
            "Poids → cible", f"{latest_weight_kg:.1f} → {target_weight_value:.1f} kg"
        )
        strategy_columns[2].metric(
            "Repère actuel",
            f"{strategy_targets['estimated_energy_kcal']} kcal/j",
            f"{strategy_targets['daily_adjustment_kcal']} kcal",
        )
        strategy_columns[3].metric(
            "Glucides conservés",
            f"{strategy_targets['carbs_g']:.0f} g/j",
        )
        strategy_columns[3].caption(
            f"≈ {strategy_targets['carb_energy_share_pct']} % de l'énergie"
        )
        st.caption(
            "Plan équilibré non keto : la qualité et la portion des glucides "
            "comptent davantage que leur suppression. Dans le dernier kilo, le "
            "déficit calorique diminue progressivement ; à la cible, le repère "
            f"passe automatiquement vers le maintien (environ "
            f"{stabilization_targets['estimated_energy_kcal']} kcal/j, à calibrer "
            "avec la courbe réelle du poids)."
        )

    profile_col, weight_col = st.columns([0.9, 1.1], gap="large")
    with profile_col:
        st.subheader("Votre profil")
        current_index = (
            OBJECTIVE_OPTIONS.index(profile["objective"])
            if profile["objective"] in OBJECTIVE_OPTIONS
            else 0
        )
        with st.form("profile_form"):
            display_name = st.text_input("Prénom (facultatif)", value=profile["display_name"])
            objective = st.selectbox(
                "Objectif principal", OBJECTIVE_OPTIONS, index=current_index
            )
            height_cm = st.number_input(
                "Taille (cm)",
                min_value=120.0,
                max_value=230.0,
                value=float(profile.get("height_cm") or DEFAULT_HEIGHT_CM),
                step=1.0,
            )
            age_years = st.number_input(
                "Âge",
                min_value=18,
                max_value=100,
                value=int(profile.get("age_years") or DEFAULT_AGE_YEARS),
                step=1,
            )
            current_sex = str(profile.get("sex") or DEFAULT_SEX)
            sex = st.selectbox(
                "Profil utilisé pour le calcul métabolique",
                SEX_OPTIONS,
                index=SEX_OPTIONS.index(current_sex)
                if current_sex in SEX_OPTIONS
                else 0,
            )
            activity_options = list(ACTIVITY_FACTORS)
            current_activity = str(
                profile.get("activity_level") or DEFAULT_ACTIVITY_LEVEL
            )
            activity_level = st.selectbox(
                "Niveau d'activité habituel",
                activity_options,
                index=(
                    activity_options.index(current_activity)
                    if current_activity in activity_options
                    else activity_options.index(DEFAULT_ACTIVITY_LEVEL)
                ),
                help=(
                    "Modérée correspond ici à votre profil : environ 7 000 pas "
                    "par jour, deux séances de natation et les trajets à vélo."
                ),
            )
            target_weight_kg = st.number_input(
                "Poids cible (kg)",
                min_value=20.0,
                max_value=400.0,
                value=float(
                    profile.get("target_weight_kg") or DEFAULT_TARGET_WEIGHT_KG
                ),
                step=0.5,
                help="76 kg correspond au milieu de votre objectif de 75 à 77 kg.",
            )
            country = st.text_input(
                "Pays / marché alimentaire",
                value=profile["country"],
                disabled=True,
                help="Les recherches de produits sont actuellement configurées pour l'Argentine.",
            )
            eating_window_hours = st.segmented_control(
                "Durée quotidienne pour manger",
                EATING_WINDOW_HOUR_OPTIONS,
                default=normalized_eating_window_hours(
                    profile.get(
                        "eating_window_hours",
                        DEFAULT_EATING_WINDOW_HOURS,
                    )
                ),
                required=True,
                format_func=lambda hours: (
                    f"{hours} h · jeûne {24 - hours} h"
                ),
                width="stretch",
                help=(
                    "Ce réglage sert à organiser les horaires et à interpréter "
                    "le journal. Il ne constitue pas un traitement médical."
                ),
            )
            eating_window_hours = normalized_eating_window_hours(
                eating_window_hours
            )
            eating_window_start = st.time_input(
                "Début de la fenêtre alimentaire",
                value=parse_clock(profile["eating_window_start"]),
                step=900,
                format="24h",
                help=(
                    "Choisissez une heure compatible avec votre faim réelle. "
                    "Vous pouvez la modifier plus tard."
                ),
            )
            profile_window = eating_window_details(
                eating_window_start.strftime("%H:%M"),
                eating_window_hours,
            )[0]
            st.caption(
                f"Fenêtre calculée : **{profile_window}** · "
                f"{eating_window_pattern(eating_window_hours)}"
            )
            st.markdown("##### Repères quotidiens d'énergie et de nutriments")
            auto_macro_targets = st.checkbox(
                "Adapter automatiquement au poids, à la taille et à l'objectif",
                value=bool(profile.get("auto_macro_targets", 1)),
            )
            suggested_targets = calculate_macro_targets(
                latest_weight_kg,
                height_cm,
                objective,
                age_years,
                sex,
                activity_level,
                target_weight_kg,
            )
            if auto_macro_targets:
                carbs_goal_g = suggested_targets["carbs_g"]
                protein_goal_g = suggested_targets["protein_g"]
                fat_goal_g = suggested_targets["fat_g"]
                calorie_goal_kcal = float(
                    suggested_targets["estimated_energy_kcal"]
                )
                st.markdown(
                    f"""
- **Calories : {calorie_goal_kcal:.0f} kcal par jour**
- **Glucides : {carbs_goal_g:.0f} g par jour**
- **Protéines : {protein_goal_g:.0f} g par jour**
- **Lipides : {fat_goal_g:.0f} g par jour**
                    """
                )
                st.caption(
                    f"Zone calorique indicative : "
                    f"{suggested_targets['energy_range_low_kcal']}–"
                    f"{suggested_targets['energy_range_high_kcal']} kcal/j · "
                    f"métabolisme de repos estimé : "
                    f"{suggested_targets['resting_energy_kcal']} kcal/j · "
                    f"maintien estimé : "
                    f"{suggested_targets['maintenance_energy_kcal']} kcal/j."
                )
                st.success(
                    "Réglage équilibré non keto : les glucides restent autour de "
                    f"{suggested_targets['carb_energy_share_pct']} % de l'énergie "
                    f"(repère souhaité : {BALANCED_CARB_SHARE_RANGE[0]}–"
                    f"{BALANCED_CARB_SHARE_RANGE[1]} %). À l'approche du poids "
                    "cible, le déficit se réduit automatiquement jusqu'au maintien."
                )
                st.caption(
                    f"IMC actuel indicatif : {suggested_targets['bmi']:.1f} · "
                    f"IMC au poids cible : {suggested_targets['target_bmi']:.1f}. "
                    "Le repère doit être calibré avec l'évolution réelle du poids "
                    "sur plusieurs semaines."
                )
                if suggested_targets["safety_note"]:
                    st.warning(suggested_targets["safety_note"])
            else:
                calorie_goal_kcal = st.number_input(
                    "Calories (kcal/jour)",
                    min_value=800.0,
                    max_value=6000.0,
                    value=float(
                        profile.get("calorie_goal_kcal")
                        or DEFAULT_CALORIE_GOAL_KCAL
                    ),
                    step=50.0,
                )
                goal_row_1 = st.columns(2)
                carbs_goal_g = goal_row_1[0].number_input(
                    "Glucides (g/jour)", 0.0, 1000.0,
                    value=float(profile["carbs_goal_g"]), step=5.0,
                )
                protein_goal_g = goal_row_1[1].number_input(
                    "Protéines (g/jour)", 0.0, 1000.0,
                    value=float(profile["protein_goal_g"]), step=5.0,
                )
                fat_goal_g = st.number_input(
                    "Lipides (g/jour)", 0.0, 1000.0,
                    value=float(profile["fat_goal_g"]), step=5.0,
                )
            meals_per_day = st.number_input(
                "Repas habituels par jour",
                min_value=1,
                max_value=8,
                value=int(profile["meals_per_day"]),
                step=1,
                help="Sert uniquement à calculer le repère par repas.",
            )
            st.caption(
                "Ces chiffres sont des repères éducatifs pour comparer vos "
                "journées, pas une prescription nutritionnelle personnalisée."
            )
            profile_clicked = st.form_submit_button(
                "Enregistrer le profil et les repères", width="stretch"
            )
        if profile_clicked:
            save_profile(
                display_name,
                objective,
                country,
                eating_window_start,
                eating_window_hours,
                height_cm,
                age_years,
                sex,
                activity_level,
                target_weight_kg,
                auto_macro_targets,
                carbs_goal_g,
                protein_goal_g,
                fat_goal_g,
                calorie_goal_kcal,
                meals_per_day,
            )
            if auto_macro_targets:
                st.success(
                    f"Profil enregistré. Repères recalculés à partir de la "
                    f"dernière pesée ({latest_weight_kg:.1f} kg)."
                )
            else:
                st.success("Profil et repères manuels enregistrés localement.")
            profile = load_profile()
        st.caption(
            "Méthode : équation de Mifflin–St Jeor pour le métabolisme de repos, "
            "ajustement par l'activité, puis déficit prudent maximal de 350 kcal "
            "lorsque l'objectif est une perte progressive. Le déficit diminue dans "
            "le dernier kilo avant la cible, puis disparaît pour la stabilisation. "
            "Ce résultat reste une estimation."
        )
        st.caption(
            "[Référence protéines EFSA ↗]"
            "(https://www.efsa.europa.eu/en/press/news/120209) · "
            "[Fourchettes de macronutriments National Academies ↗]"
            "(https://www.nationalacademies.org/read/10872/chapter/13)"
        )

    with weight_col:
        st.subheader("Suivi du poids")
        with st.form("weight_form", clear_on_submit=False):
            weight_date_col, weight_value_col = st.columns(2)
            measured_on = weight_date_col.date_input("Date", value=date.today())
            weight_kg = weight_value_col.number_input(
                "Poids (kg)", min_value=20.0, max_value=400.0,
                value=latest_weight_kg, step=0.1
            )
            weight_note = st.text_input("Note facultative", placeholder="Ex. : mesure du matin")
            weight_clicked = st.form_submit_button("Ajouter la mesure", width="stretch")
        if weight_clicked:
            save_weight(measured_on, weight_kg, weight_note)
            weights = load_weights()
            if profile.get("auto_macro_targets"):
                refreshed_targets = calculate_macro_targets(
                    weight_kg,
                    float(profile.get("height_cm") or DEFAULT_HEIGHT_CM),
                    profile["objective"],
                    int(profile.get("age_years") or DEFAULT_AGE_YEARS),
                    str(profile.get("sex") or DEFAULT_SEX),
                    str(profile.get("activity_level") or DEFAULT_ACTIVITY_LEVEL),
                    float(
                        profile.get("target_weight_kg")
                        or DEFAULT_TARGET_WEIGHT_KG
                    ),
                )
                save_profile(
                    profile["display_name"],
                    profile["objective"],
                    profile["country"],
                    parse_clock(profile["eating_window_start"]),
                    int(
                        profile.get(
                            "eating_window_hours",
                            DEFAULT_EATING_WINDOW_HOURS,
                        )
                    ),
                    float(profile.get("height_cm") or DEFAULT_HEIGHT_CM),
                    int(profile.get("age_years") or DEFAULT_AGE_YEARS),
                    str(profile.get("sex") or DEFAULT_SEX),
                    str(profile.get("activity_level") or DEFAULT_ACTIVITY_LEVEL),
                    float(
                        profile.get("target_weight_kg")
                        or DEFAULT_TARGET_WEIGHT_KG
                    ),
                    True,
                    refreshed_targets["carbs_g"],
                    refreshed_targets["protein_g"],
                    refreshed_targets["fat_g"],
                    refreshed_targets["estimated_energy_kcal"],
                    int(profile["meals_per_day"]),
                )
                profile = load_profile()
                st.success("Mesure ajoutée et repères quotidiens recalculés.")
            else:
                st.success("Mesure ajoutée.")

    st.divider()
    score_chart_col, weight_chart_col = st.columns(2, gap="large")
    with score_chart_col:
        st.subheader("Évolution des notes moyennes")
        if not meals:
            st.info("Le graphique apparaîtra après votre première analyse.")
        else:
            score_frame = pd.DataFrame(meals)
            score_frame["Jour"] = [
                (
                    meal_occurred_at(meal).astimezone(LOCAL_TIMEZONE).date()
                    if meal_occurred_at(meal) is not None
                    else None
                )
                for meal in meals
            ]
            score_frame = score_frame.dropna(subset=["Jour"])
            daily_scores = (
                score_frame.groupby("Jour", as_index=False)["score"]
                .mean()
                .rename(columns={"score": "Note moyenne"})
            )
            daily_scores["Moyenne mobile"] = daily_scores["Note moyenne"].rolling(
                window=7, min_periods=1
            ).mean()
            score_long = daily_scores.melt(
                id_vars="Jour",
                value_vars=["Note moyenne", "Moyenne mobile"],
                var_name="Série",
                value_name="Note",
            )
            score_chart = (
                alt.Chart(score_long)
                .mark_line(point=True, strokeWidth=3)
                .encode(
                    x=alt.X("Jour:T", title="Date"),
                    y=alt.Y("Note:Q", title="Note / 10", scale=alt.Scale(domain=[0, 10])),
                    color=alt.Color(
                        "Série:N",
                        scale=alt.Scale(
                            domain=["Note moyenne", "Moyenne mobile"],
                            range=["#22c55e", "#0f766e"],
                        ),
                        title=None,
                    ),
                    tooltip=["Jour:T", "Série:N", alt.Tooltip("Note:Q", format=".1f")],
                )
                .properties(height=310)
            )
            st.altair_chart(score_chart)

    with weight_chart_col:
        st.subheader("Évolution du poids")
        if not weights:
            st.info("Ajoutez une première mesure pour afficher la courbe.")
        else:
            weight_frame = pd.DataFrame(weights)
            weight_frame["Date"] = pd.to_datetime(weight_frame["measured_on"])
            weight_chart = (
                alt.Chart(weight_frame)
                .mark_line(point=True, strokeWidth=3, color="#f97316")
                .encode(
                    x=alt.X("Date:T", title="Date"),
                    y=alt.Y("weight_kg:Q", title="Poids (kg)", scale=alt.Scale(zero=False)),
                    tooltip=["Date:T", alt.Tooltip("weight_kg:Q", title="Poids", format=".1f")],
                )
                .properties(height=310)
            )
            st.altair_chart(weight_chart)

    st.divider()
    st.subheader("Projection indicative du poids")
    st.caption(
        "Cette simulation utilise uniquement les journées où tous vos repas "
        "habituels ont été consignés. Elle illustre votre plan — perte jusqu'à "
        "la cible puis maintien — et non une promesse ni une prescription."
    )
    projection_period = st.segmented_control(
        "Horizon de projection",
        ["4 semaines", "8 semaines", "12 semaines"],
        default="12 semaines",
        key="weight_projection_period",
    )
    projection_days = {
        "4 semaines": 28,
        "8 semaines": 56,
        "12 semaines": 84,
    }.get(projection_period or "12 semaines", 84)
    projection_frame, projection_meta = build_weight_projection(
        meals, weights, profile, projection_days
    )
    if projection_frame.empty:
        st.info(str(projection_meta.get("reason", "Données insuffisantes.")))
    else:
        planned_intake = float(
            profile.get("calorie_goal_kcal") or DEFAULT_CALORIE_GOAL_KCAL
        )
        if projection_meta["average_intake_kcal"] < planned_intake * 0.80:
            st.warning(
                f"La moyenne consignée ({projection_meta['average_intake_kcal']:.0f} "
                f"kcal/j) est nettement sous le repère du plan ({planned_intake:.0f} "
                "kcal/j). Vérifiez d'abord que les portions et tous les repas sont "
                "bien saisis : l'application ne recommande pas de prolonger un "
                "déficit aussi marqué."
            )
        projected_change = (
            float(projection_meta["projected_weight_kg"])
            - float(projection_meta["latest_weight_kg"])
        )
        projection_metrics = st.columns(4, border=True)
        projection_metrics[0].metric(
            "Apport moyen consigné",
            f"{projection_meta['average_intake_kcal']:.0f} kcal/j",
            help=(
                f"Moyenne de {projection_meta['complete_days']} journées complètes "
                "parmi les 28 derniers jours."
            ),
        )
        projection_metrics[1].metric(
            "Maintien estimé",
            f"{projection_meta['maintenance_kcal']:.0f} kcal/j",
        )
        projection_metrics[2].metric(
            f"Poids projeté à {projection_period}",
            f"{projection_meta['projected_weight_kg']:.1f} kg",
            delta=f"{projected_change:+.1f} kg",
            delta_color="inverse",
        )
        projection_metrics[3].metric(
            "Écart énergétique observé",
            f"{projection_meta['daily_gap_kcal']:+.0f} kcal/j",
        )

        projection_band = (
            alt.Chart(projection_frame)
            .mark_area(color="#0f766e", opacity=0.15)
            .encode(
                x=alt.X("Date:T", title="Date"),
                y=alt.Y(
                    "Scénario bas:Q",
                    title="Poids projeté (kg)",
                    scale=alt.Scale(zero=False),
                ),
                y2="Scénario haut:Q",
                tooltip=[
                    alt.Tooltip("Date:T", title="Date", format="%d/%m/%Y"),
                    alt.Tooltip("Scénario bas:Q", title="Scénario bas", format=".1f"),
                    alt.Tooltip("Scénario haut:Q", title="Scénario haut", format=".1f"),
                ],
            )
        )
        projection_line = (
            alt.Chart(projection_frame)
            .mark_line(color="#0f766e", strokeWidth=3)
            .encode(
                x="Date:T",
                y="Projection:Q",
                tooltip=[
                    alt.Tooltip("Date:T", title="Date", format="%d/%m/%Y"),
                    alt.Tooltip("Projection:Q", title="Projection", format=".1f"),
                ],
            )
        )
        target_frame = pd.DataFrame(
            {"Poids cible": [float(profile.get("target_weight_kg") or DEFAULT_TARGET_WEIGHT_KG)]}
        )
        target_line = (
            alt.Chart(target_frame)
            .mark_rule(color="#f97316", strokeDash=[7, 5], strokeWidth=2)
            .encode(y="Poids cible:Q")
        )
        actual_frame = pd.DataFrame(weights)
        actual_frame["Date"] = pd.to_datetime(actual_frame["measured_on"])
        actual_points = (
            alt.Chart(actual_frame)
            .mark_circle(color="#f97316", size=85)
            .encode(
                x="Date:T",
                y="weight_kg:Q",
                tooltip=[
                    alt.Tooltip("Date:T", title="Pesée", format="%d/%m/%Y"),
                    alt.Tooltip("weight_kg:Q", title="Poids", format=".1f"),
                ],
            )
        )
        st.altair_chart(
            (projection_band + projection_line + target_line + actual_points)
            .resolve_scale(y="shared")
            .properties(height=330)
        )
        st.caption(
            "Ligne verte : projection centrale · zone verte : scénarios avec "
            f"±{projection_meta['uncertainty_kcal']:.0f} kcal/j pour refléter "
            "l'incertitude des portions · points orange : pesées réelles · ligne "
            "orange pointillée : poids cible. La projection s'arrête à la cible "
            "pour représenter le passage prévu au maintien ; elle ne suppose pas "
            "que le déficit continue indéfiniment."
        )
        st.caption(
            "Les variations d'eau, de sel, de glycogène, du transit et de "
            "l'activité peuvent déplacer fortement le poids à court terme. "
            "Calibrez la projection avec plusieurs pesées comparables."
        )

    st.divider()
    st.subheader("Calories : consommation quotidienne et écart")
    calorie_goal = float(
        profile.get("calorie_goal_kcal") or DEFAULT_CALORIE_GOAL_KCAL
    )
    daily_calories = build_daily_calorie_data(meals, profile)
    today_start = pd.Timestamp(date.today())
    today_rows = (
        daily_calories[daily_calories["Date"] == today_start]
        if not daily_calories.empty
        else pd.DataFrame()
    )
    calories_today = (
        float(today_rows.iloc[0]["Calories estimées"])
        if not today_rows.empty
        else 0.0
    )
    meals_estimated_today = (
        int(today_rows.iloc[0]["Repas estimés"]) if not today_rows.empty else 0
    )
    calorie_gap_today = calories_today - calorie_goal
    gap_value = int(round(calorie_gap_today))
    st.caption(
        "Lecture d'aujourd'hui · consommation, repère et écart sont exprimés "
        "en kcal. Les valeurs restent provisoires tant que la journée et tous "
        "les repas ne sont pas terminés."
    )
    with st.container(
        horizontal=True,
        horizontal_alignment="distribute",
        gap="small",
    ):
        st.metric(
            "Consommé",
            int(round(calories_today)),
            format="%,d",
            border=True,
            width=190,
            icon=":material/local_fire_department:",
            help="Calories estimées à partir des repas consignés aujourd'hui.",
        )
        st.metric(
            "Repère",
            int(round(calorie_goal)),
            format="%,d",
            border=True,
            width=190,
            icon=":material/flag:",
            help="Repère calorique quotidien défini dans votre profil.",
        )
        st.metric(
            "Écart",
            gap_value,
            format="%+,d",
            border=True,
            width=190,
            icon=":material/compare_arrows:",
            help=(
                "Consommé moins repère. Un écart négatif en cours de journée "
                "ne signifie pas automatiquement qu'il faut manger davantage."
            ),
        )
        st.metric(
            "Repas",
            meals_estimated_today,
            border=True,
            width=190,
            icon=":material/restaurant:",
            help="Nombre de repas du jour disposant d'une estimation complète.",
        )

    if daily_calories.empty:
        st.info(
            "Le suivi calorique apparaîtra dès qu'un repas comportera une "
            "estimation de glucides, protéines et lipides."
        )
    else:
        calorie_chart_data = daily_calories.tail(10).copy()
        calorie_chart_data["Jour"] = calorie_chart_data["Date"].dt.strftime(
            "%d/%m"
        )
        day_order = calorie_chart_data["Jour"].tolist()
        calorie_chart_data["Consommé affiché"] = calorie_chart_data[
            "Calories estimées"
        ].map(lambda value: f"{value:,.0f} kcal".replace(",", " "))
        calorie_chart_data["Écart affiché"] = calorie_chart_data["Écart"].map(
            lambda value: f"écart {value:+,.0f}".replace(",", " ")
        )
        chart_upper_bound = max(
            float(calorie_chart_data["Calories estimées"].max()),
            calorie_goal,
        ) * 1.18
        calorie_bars = (
            alt.Chart(calorie_chart_data)
            .mark_bar(
                color="#0f766e",
                cornerRadiusTopLeft=7,
                cornerRadiusTopRight=7,
            )
            .encode(
                x=alt.X(
                    "Jour:O",
                    title=None,
                    sort=day_order,
                    axis=alt.Axis(labelAngle=0, labelPadding=8),
                    scale=alt.Scale(paddingInner=0.48, paddingOuter=0.3),
                ),
                y=alt.Y(
                    "Calories estimées:Q",
                    title="Calories estimées (kcal)",
                    scale=alt.Scale(domain=[0, chart_upper_bound]),
                ),
                tooltip=[
                    alt.Tooltip("Date:T", title="Date", format="%d/%m/%Y"),
                    alt.Tooltip(
                        "Calories estimées:Q", title="Consommé", format=".0f"
                    ),
                    alt.Tooltip("Objectif:Q", title="Objectif", format=".0f"),
                    alt.Tooltip("Écart:Q", title="Écart", format="+.0f"),
                    alt.Tooltip("Repas estimés:Q", title="Repas estimés"),
                ],
            )
        )
        calorie_goal_line = (
            alt.Chart(pd.DataFrame({"Objectif": [calorie_goal]}))
            .mark_rule(color="#f97316", strokeDash=[7, 5], strokeWidth=3)
            .encode(y="Objectif:Q")
        )
        calorie_value_labels = (
            alt.Chart(calorie_chart_data)
            .mark_text(dy=-22, color="#1e293b", fontWeight="bold", fontSize=12)
            .encode(
                x=alt.X("Jour:O", sort=day_order),
                y="Calories estimées:Q",
                text="Consommé affiché:N",
            )
        )
        calorie_gap_labels = (
            alt.Chart(calorie_chart_data)
            .mark_text(dy=-7, color="#64748b", fontSize=10)
            .encode(
                x=alt.X("Jour:O", sort=day_order),
                y="Calories estimées:Q",
                text="Écart affiché:N",
            )
        )
        target_label = (
            alt.Chart(
                pd.DataFrame(
                    {
                        "Jour": [day_order[-1]],
                        "Objectif": [calorie_goal],
                        "Libellé": [
                            f"Repère {calorie_goal:,.0f} kcal".replace(",", " ")
                        ],
                    }
                )
            )
            .mark_text(
                align="right",
                baseline="bottom",
                dy=-7,
                color="#ea580c",
                fontWeight="bold",
            )
            .encode(
                x=alt.X("Jour:O", sort=day_order),
                y="Objectif:Q",
                text="Libellé:N",
            )
        )
        st.altair_chart(
            (
                calorie_bars
                + calorie_goal_line
                + calorie_value_labels
                + calorie_gap_labels
                + target_label
            ).properties(height=340)
        )
        st.caption(
            "Chaque date apparaît une seule fois. Le nombre principal indique "
            "les calories consignées ; la petite ligne indique l'écart au repère. "
            "La ligne orange pointillée représente le repère quotidien."
        )

        calorie_history = daily_calories.sort_values("Date", ascending=False).copy()
        calorie_history["Date"] = calorie_history["Date"].dt.strftime("%d/%m/%Y")
        calorie_history["Lecture de l'écart"] = calorie_history["Écart"].map(
            lambda value: (
                f"{value:+.0f} kcal vs repère"
                if value != 0
                else "Objectif atteint"
            )
        )
        st.dataframe(
            calorie_history[
                [
                    "Date",
                    "Calories estimées",
                    "Objectif",
                    "Lecture de l'écart",
                    "Repas estimés",
                ]
            ],
            hide_index=True,
            width="stretch",
            column_config={
                "Calories estimées": st.column_config.NumberColumn(
                    "Consommé", format="%.0f kcal"
                ),
                "Objectif": st.column_config.NumberColumn(
                    "Objectif", format="%.0f kcal"
                ),
            },
        )

    st.caption(
        "Calcul indicatif : 4 kcal par gramme de glucides, 4 kcal par gramme "
        "de protéines et 9 kcal par gramme de lipides. Les journées ou repas "
        "non consignés ne sont pas comptés. Un écart négatif ne signifie pas "
        "qu'il faut automatiquement manger davantage."
    )

    st.divider()
    st.subheader("Macronutriments : réalité estimée et objectif")
    macro_metric_cols = st.columns(4, border=True)
    macro_metric_cols[0].metric("Calories / jour", f"{calorie_goal:.0f} kcal")
    macro_metric_cols[1].metric("Glucides / jour", f"{profile['carbs_goal_g']:.0f} g")
    macro_metric_cols[2].metric("Protéines / jour", f"{profile['protein_goal_g']:.0f} g")
    macro_metric_cols[3].metric("Lipides / jour", f"{profile['fat_goal_g']:.0f} g")
    st.caption(f"Repère actuel : {int(profile['meals_per_day'])} repas par jour.")

    st.markdown("#### Où en êtes-vous par rapport aux objectifs ?")
    indicator_period = st.segmented_control(
        "Période des indicateurs",
        ["Aujourd'hui", "Cette semaine", "Ce mois"],
        default="Aujourd'hui",
        required=True,
        key="macro_indicator_period",
    )
    macro_progress = build_current_macro_progress(
        meals, profile, indicator_period or "Aujourd'hui"
    )
    st.caption(
        f"{macro_progress['period_label']} · "
        f"{macro_progress['elapsed_days']} jour"
        f"{'s' if macro_progress['elapsed_days'] > 1 else ''} calendaire"
        f"{'s' if macro_progress['elapsed_days'] > 1 else ''} · "
        "consommation cumulée comparée au repère quotidien multiplié par ce "
        "même nombre de jours."
    )
    if not macro_progress["meal_count"]:
        st.info(
            "Aucun repas avec macronutriments n'est encore enregistré sur cette "
            "période : les jauges restent donc à 0 %."
        )
    indicator_columns = st.columns(3, gap="large", border=True)
    for column, indicator in zip(
        indicator_columns, macro_progress["indicators"], strict=True
    ):
        status_meta = MACRO_STATUS_META[indicator["status"]]
        ratio = indicator["ratio"]
        difference_g = indicator["target_g"] - indicator["actual_g"]
        if ratio < 0.80:
            direction = "Sous le repère estimatif — sans obligation de compenser"
            explanation = (
                f"Écart estimé : **{max(0, difference_g):.0f} g** sous le repère. "
                "Vérifiez d'abord les quantités saisies, votre faim, votre énergie "
                "et la tendance du poids avant de modifier les portions."
            )
        elif ratio <= 1.10:
            direction = "✅ Maintenir : vous êtes dans la cible"
            if difference_g >= 0:
                explanation = (
                    f"Vous êtes dans la zone verte, à **{difference_g:.0f} g** "
                    "du repère central."
                )
            else:
                explanation = (
                    f"Vous êtes dans la zone verte, à **{abs(difference_g):.0f} g** "
                    "au-dessus du repère central."
                )
        else:
            direction = "Au-dessus du repère indicatif"
            explanation = (
                f"Le cumul se situe environ **{abs(difference_g):.0f} g** au-dessus "
                "du repère central."
            )
        with column:
            st.markdown(
                f"### {indicator['label']}  \n"
                f"**{indicator['actual_g']:.0f} g cumulés** / "
                f"**{indicator['target_g']:.0f} g de repère cumulé** · "
                f"**{ratio * 100:.0f} %**"
            )
            st.altair_chart(build_macro_gauge(indicator))
            st.markdown(f"{status_meta['emoji']} **{direction}**")
            st.markdown(explanation)
            if macro_progress["elapsed_days"] < macro_progress["total_days"]:
                st.caption(
                    f"À titre indicatif, repère de la période complète : "
                    f"{indicator['full_period_target_g']:.0f} g"
                )
    st.caption(
        "La jauge se lit de gauche à droite : bleu = sous le repère estimatif, "
        "vert = zone de comparaison, orange puis rouge = au-dessus du repère. "
        "Le trait noir représente votre cumul et le trait blanc "
        "pointillé le repère central à 100 %. Vert = 80–110 % ; orange = "
        "111–130 % ; rouge = plus de 130 %. "
        "Pour la semaine, les cumuls commencent le lundi ; pour le mois, ils "
        "commencent le 1er. Le repère avance d'un jour à chaque nouvelle date "
        "calendaire. Ce sont des repères visuels, pas des seuils médicaux. Un "
        "journal incomplet sous-estime la semaine ou le mois."
    )

    st.markdown("#### Historique détaillé")
    granularity = st.segmented_control(
        "Regrouper l'historique par",
        ["Repas", "Jour", "Semaine", "Mois"],
        default="Jour",
        required=True,
        key="macro_granularity",
    )
    macro_data = build_macro_chart_data(meals, profile, granularity or "Jour")
    meals_with_macros = sum(
        meal.get("carbs_g") is not None
        and meal.get("protein_g") is not None
        and meal.get("fat_g") is not None
        for meal in meals
    )
    if macro_data.empty:
        st.info(
            "Le graphique commencera avec la prochaine analyse de repas. Les "
            "anciens repas n'ont pas de macronutriments enregistrés."
        )
    else:
        period_order = (
            macro_data.sort_values("Ordre")["Période"].drop_duplicates().tolist()
        )
        base = alt.Chart(macro_data).encode(
            x=alt.X(
                "Période:N",
                sort=period_order,
                title=None,
                axis=alt.Axis(labelAngle=-35, labelLimit=170),
            ),
            y=alt.Y("Grammes:Q", title="Grammes"),
        )
        actual_bars = (
            base.transform_filter(alt.datum["Série"] == "Réalité estimée")
            .mark_bar(cornerRadiusTopLeft=4, cornerRadiusTopRight=4)
            .encode(
                color=alt.Color(
                    "Macro:N",
                    scale=alt.Scale(
                        domain=["Glucides", "Protéines", "Lipides"],
                        range=["#f59e0b", "#0f766e", "#8b5cf6"],
                    ),
                    legend=None,
                ),
                tooltip=[
                    "Période:N",
                    "Macro:N",
                    alt.Tooltip("Grammes:Q", title="Réalité estimée", format=".1f"),
                ],
            )
        )
        goal_line = (
            base.transform_filter(alt.datum["Série"] == "Objectif")
            .mark_line(
                color="#334155",
                point=alt.OverlayMarkDef(filled=True, size=55),
                strokeDash=[7, 5],
                strokeWidth=2,
            )
            .encode(
                tooltip=[
                    "Période:N",
                    "Macro:N",
                    alt.Tooltip("Grammes:Q", title="Objectif", format=".1f"),
                ]
            )
        )
        macro_chart = (
            (actual_bars + goal_line)
            .properties(height=155)
            .facet(
                row=alt.Row(
                    "Macro:N",
                    sort=["Glucides", "Protéines", "Lipides"],
                    title=None,
                    header=alt.Header(labelFontSize=14, labelFontWeight="bold"),
                )
            )
            .resolve_scale(y="independent")
        )
        st.altair_chart(macro_chart)
        st.caption(
            f"{meals_with_macros} repas avec estimation sur {len(meals)}. "
            "Barres colorées : réalité estimée · ligne pointillée : objectif. "
            "Les objectifs hebdomadaires et mensuels supposent que tous les repas "
            "sont consignés."
        )

    st.divider()
    st.subheader("Bilan global personnalisé")
    st.caption(
        "DeepSeek analyse jusqu'aux 60 repas, 20 mesures, les activités récentes, "
        "les aliments personnels, les sensations encore récentes ou répétées et, "
        "après au moins trois observations, la tendance des selles. Gemini puis "
        "Groq prennent le relais si besoin."
    )
    report_clicked = st.button(
        "🧭 Générer mon bilan IA",
        type="primary",
        disabled=not bool(meals),
    )
    if report_clicked:
        if not ai_settings.has_any_key:
            st.error("Ajoutez une clé DeepSeek, Gemini ou Groq dans la barre latérale.")
        else:
            with st.spinner("Construction de votre bilan personnalisé…"):
                try:
                    report = build_global_report(
                        ai_settings,
                        meals,
                        weights,
                        profile,
                        symptom_events,
                        bowel_events,
                        load_activity_events(),
                        load_favorite_foods(),
                    )
                    st.session_state["global_report"] = report
                except Exception as error:
                    st.error(
                        "Bilan impossible. Vérifiez la clé, le quota et la connexion, puis réessayez.\n\n"
                        f"Détail : {safe_error_message(error, ai_settings)}"
                    )

    if (
        "global_report" in st.session_state
        and st.session_state["global_report"].get("_context_version")
        != REPORT_CONTEXT_VERSION
    ):
        del st.session_state["global_report"]

    if "global_report" in st.session_state:
        report = st.session_state["global_report"]
        trend_labels = {
            "amelioration": "📈 Tendance en amélioration",
            "stable": "➡️ Tendance stable",
            "a_surveille": "🔎 Tendance à surveiller",
        }
        st.success(trend_labels[report["tendance"]])
        st.caption(f"Bilan réalisé par {report.get('_provider', 'IA cloud')}.")
        st.write(report["resume"])
        st.markdown("##### Vos 3 prochains leviers")
        for index, tip in enumerate(report["conseils"], start=1):
            st.markdown(f"**{index}.** {tip}")
        st.info(report["encouragement"], icon="🌱")
