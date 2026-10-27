# Coach Glycémie

Application Streamlit personnelle de suivi alimentaire, glycémique et pondéral.

## Données et sécurité

- Les données du journal sont stockées dans un projet Supabase privé.
- L'accès passe par Supabase Auth et les règles RLS limitent chaque compte à ses propres lignes.
- Les clés d'API ne sont jamais versionnées. En production, elles doivent être ajoutées dans les secrets de Streamlit Community Cloud.
- La base SQLite locale du Mac sert uniquement de sauvegarde historique et n'est pas incluse dans le dépôt.

## Lancement local

```bash
python -m pip install -r requirements.txt
streamlit run app.py
```

La configuration locale attend au minimum `SUPABASE_URL` et
`SUPABASE_PUBLISHABLE_KEY` dans `.streamlit/secrets.toml`.

## Déploiement

Le point d'entrée est `app.py`. Pour un déploiement sur Streamlit Community
Cloud, choisir Python 3.12 et renseigner les secrets via les réglages privés de
l'application, jamais dans GitHub.
