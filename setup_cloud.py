"""Assistant graphique de migration du Coach Glycemie vers Supabase."""

from __future__ import annotations

import re
from pathlib import Path

import streamlit as st

from supabase_sync import (
    SupabaseSession,
    SupabaseSyncError,
    local_counts,
    migrate_sqlite,
    remote_counts,
    sign_in,
    sign_up,
)


APP_DIR = Path(__file__).parent
DB_PATH = APP_DIR / "coach_glycemie.db"
SECRETS_PATH = APP_DIR / ".streamlit" / "secrets.toml"
DEFAULT_PROJECT_URL = "https://lyxdixmqwjoistjimazh.supabase.co"
DEFAULT_EMAIL = "nicolas.foliot@gmail.com"


st.set_page_config(
    page_title="Configurer le cloud · Coach Glycémie",
    page_icon="☁️",
    layout="centered",
    initial_sidebar_state="collapsed",
)


def configured_value(name: str, default: str = "") -> str:
    try:
        return str(st.secrets.get(name, default)).strip()
    except (FileNotFoundError, KeyError):
        return default


def save_public_configuration(project_url: str, publishable_key: str) -> None:
    """Conserve uniquement l'URL et la cle publiable, jamais le mot de passe."""
    if publishable_key.startswith("sb_secret_") or "service_role" in publishable_key:
        raise ValueError(
            "Utilisez la clé publiable, jamais une clé secrète ou service_role."
        )
    SECRETS_PATH.parent.mkdir(parents=True, exist_ok=True)
    existing = SECRETS_PATH.read_text(encoding="utf-8") if SECRETS_PATH.exists() else ""
    for key in ("SUPABASE_URL", "SUPABASE_PUBLISHABLE_KEY"):
        existing = re.sub(
            rf"(?m)^\s*{re.escape(key)}\s*=.*(?:\n|$)",
            "",
            existing,
        )
    safe_url = project_url.replace("\\", "\\\\").replace('"', '\\"')
    safe_key = publishable_key.replace("\\", "\\\\").replace('"', '\\"')
    updated = existing.rstrip() + (
        f'\nSUPABASE_URL = "{safe_url}"\n'
        f'SUPABASE_PUBLISHABLE_KEY = "{safe_key}"\n'
    )
    SECRETS_PATH.write_text(updated.lstrip(), encoding="utf-8")


def counts_table(counts: dict[str, int]) -> list[dict[str, object]]:
    labels = {
        "meals": "Repas",
        "weights": "Pesées",
        "symptom_events": "Faim et douleurs",
        "bowel_events": "Digestion",
        "activity_events": "Activités",
        "favorite_foods": "Aliments personnels",
        "profile": "Profil",
    }
    return [
        {"Donnée": labels.get(table, table), "Nombre": count}
        for table, count in counts.items()
    ]


st.title("Passer le Coach Glycémie dans le cloud")
st.caption(
    "Cette opération prépare l’accès depuis l’iPhone. La base SQLite du Mac "
    "reste intacte comme sauvegarde."
)

with st.container(border=True):
    st.subheader("1. Vérifier la sauvegarde locale")
    local = local_counts(DB_PATH)
    st.dataframe(counts_table(local), hide_index=True)
    st.success(
        f"Sauvegarde locale détectée : {sum(local.values())} lignes au total.",
        icon=":material/check_circle:",
    )

with st.container(border=True):
    st.subheader("2. Relier le projet Supabase")
    project_url = st.text_input(
        "URL du projet",
        value=configured_value("SUPABASE_URL", DEFAULT_PROJECT_URL),
    ).strip()
    publishable_key = st.text_input(
        "Clé publiable Supabase",
        value=configured_value("SUPABASE_PUBLISHABLE_KEY"),
        type="password",
        placeholder="sb_publishable_… ou ancienne clé anon",
        help=(
            "Dans Supabase : Connect > API Keys. La clé publiable peut être "
            "utilisée dans l'application. Ne collez jamais service_role ou sb_secret_."
        ),
    ).strip()
    if st.button(
        "Enregistrer la connexion publique sur ce Mac",
        icon=":material/save:",
        disabled=not bool(project_url and publishable_key),
        width="stretch",
    ):
        try:
            save_public_configuration(project_url, publishable_key)
            st.success("URL et clé publiable enregistrées. Aucun mot de passe conservé.")
        except (OSError, ValueError) as error:
            st.error(str(error))

with st.container(border=True):
    st.subheader("3. Créer ou ouvrir votre accès privé")
    st.info(
        "Utilisez un mot de passe propre à cette application, différent de celui "
        "du tableau de bord Supabase. Il ne sera pas enregistré sur le Mac."
    )
    email = st.text_input("Adresse e-mail", value=DEFAULT_EMAIL).strip()
    password = st.text_input(
        "Mot de passe de l'application",
        type="password",
        help="Au moins 8 caractères. Ce champ est effacé à la fermeture de la page.",
    )
    action_row = st.container(horizontal=True)
    create_clicked = action_row.button(
        "Créer mon accès",
        icon=":material/person_add:",
        disabled=not bool(project_url and publishable_key and email and len(password) >= 8),
    )
    login_clicked = action_row.button(
        "Me connecter",
        type="primary",
        icon=":material/login:",
        disabled=not bool(project_url and publishable_key and email and password),
    )
    if create_clicked:
        try:
            st.success(sign_up(project_url, publishable_key, email, password))
        except (ValueError, SupabaseSyncError) as error:
            st.error(f"Création impossible : {error}")
    if login_clicked:
        try:
            st.session_state["supabase_session"] = sign_in(
                project_url,
                publishable_key,
                email,
                password,
            )
            st.success("Connexion privée établie.")
        except (ValueError, SupabaseSyncError) as error:
            st.error(f"Connexion impossible : {error}")

session = st.session_state.get("supabase_session")
if isinstance(session, SupabaseSession):
    with st.container(border=True):
        st.subheader("4. Copier l’historique")
        st.success(f"Connecté avec {session.email}", icon=":material/lock:")
        try:
            cloud_before = remote_counts(session)
            st.markdown("**État actuel du cloud**")
            st.dataframe(counts_table(cloud_before), hide_index=True)
        except SupabaseSyncError as error:
            cloud_before = None
            st.error(
                "Les tables ne sont pas encore accessibles. Exécutez d'abord le "
                f"schéma préparé dans Supabase. Détail : {error}"
            )

        consent = st.checkbox(
            "J’autorise l’envoi de mon profil, repas, poids, activités, faim, "
            "douleurs, selles et aliments personnels vers mon projet Supabase privé.",
            disabled=cloud_before is None,
        )
        migrate_clicked = st.button(
            "Copier mes données vers Supabase",
            type="primary",
            icon=":material/cloud_upload:",
            disabled=not consent or cloud_before is None,
            width="stretch",
        )
        if migrate_clicked:
            with st.spinner("Copie sécurisée en cours…"):
                try:
                    migrated = migrate_sqlite(DB_PATH, session)
                    cloud_after = remote_counts(session)
                    st.success(
                        "Copie terminée. La base locale n'a pas été modifiée.",
                        icon=":material/cloud_done:",
                    )
                    result_rows = []
                    for table, local_count in migrated.items():
                        result_rows.append(
                            {
                                "Table": table,
                                "Local": local_count,
                                "Cloud": cloud_after.get(table, 0),
                                "Vérification": (
                                    "OK"
                                    if cloud_after.get(table, 0) == local_count
                                    else "À vérifier"
                                ),
                            }
                        )
                    st.dataframe(result_rows, hide_index=True)
                except SupabaseSyncError as error:
                    st.error(f"La copie s'est arrêtée sans modifier SQLite : {error}")

st.divider()
st.caption(
    "Étape suivante après une copie vérifiée : connecter l'application principale "
    "à Supabase puis la publier en version mobile."
)

