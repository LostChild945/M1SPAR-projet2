"""
Dashboard Streamlit — Recommandation Amazon P2

3 onglets :
  1. Recommandations live par user_id
  2. CTR par variante A/B (bar chart)
  3. Top produits populaires

Usage :
    streamlit run app.py
"""

import os

import pandas as pd
import plotly.express as px
import requests
import streamlit as st

API_URL = os.environ.get("API_URL", "http://api:8000")

st.set_page_config(page_title="P2 Recommandation Amazon", layout="wide")
st.title("Recommandation Amazon — Dashboard")

tab1, tab2, tab3 = st.tabs(["Recommandations live", "A/B Testing CTR", "Top Produits"])

# ── Tab 1 : Recommandations ─────────────────────────────────────────────────
with tab1:
    st.header("Recommandations personnalisées")

    col1, col2 = st.columns([2, 1])
    with col1:
        user_id = st.text_input("User ID", placeholder="ex: AHMY5CWKA...")
    with col2:
        n = st.slider("Nombre de recommandations", 1, 50, 10)

    category = st.text_input("Catégorie (optionnel)", placeholder="ex: All_Beauty")

    if st.button("Obtenir les recommandations", type="primary"):
        if user_id:
            params = {"n": n, "exclude_purchased": True}
            if category:
                params["category"] = category
            try:
                resp = requests.get(f"{API_URL}/recommend/{user_id}", params=params, timeout=5)
                if resp.status_code == 200:
                    data = resp.json()
                    st.success(
                        f"Segment : **{data['segment']}** | "
                        f"Latence : **{data['latency_ms']} ms** | "
                        f"Cache : **{'HIT' if data['cache_hit'] else 'MISS'}**"
                    )
                    recs = pd.DataFrame(data["recommendations"])
                    st.dataframe(recs, use_container_width=True)
                elif resp.status_code == 404:
                    st.warning("Aucune recommandation trouvée pour cet utilisateur.")
                else:
                    st.error(f"Erreur API : {resp.status_code} — {resp.text}")
            except requests.ConnectionError:
                st.error("Impossible de contacter l'API. Vérifiez que le service est démarré.")
        else:
            st.warning("Entrez un User ID.")

# ── Tab 2 : A/B Testing ─────────────────────────────────────────────────────
with tab2:
    st.header("Résultats A/B Testing")

    experiment_id = st.text_input("Experiment ID", value="hybrid_vs_als_only")

    if st.button("Charger les résultats A/B"):
        try:
            resp = requests.get(
                f"{API_URL}/ab_results",
                params={"experiment_id": experiment_id},
                timeout=5,
            )
            if resp.status_code == 200:
                data = resp.json()

                col1, col2, col3, col4 = st.columns(4)
                col1.metric("CTR Control", f"{data['control_ctr']:.4f}")
                col2.metric("CTR Treatment", f"{data['treatment_ctr']:.4f}")
                col3.metric("Lift", f"{data['lift_pct']:+.1f}%")
                col4.metric("p-value", f"{data['p_value']:.4f}")

                st.write(f"Significatif (p < 0.05) : **{'Oui' if data['significant'] else 'Non'}**")
                st.write(
                    f"Users control : {data['n_users_control']} | "
                    f"Users treatment : {data['n_users_treatment']}"
                )

                # Bar chart CTR
                ctr_df = pd.DataFrame({
                    "Variante": ["Control", "Treatment"],
                    "CTR": [data["control_ctr"], data["treatment_ctr"]],
                })
                fig = px.bar(
                    ctr_df, x="Variante", y="CTR", color="Variante",
                    title="CTR par variante A/B",
                    color_discrete_map={"Control": "#636EFA", "Treatment": "#EF553B"},
                )
                st.plotly_chart(fig, use_container_width=True)
            else:
                st.error(f"Erreur API : {resp.status_code}")
        except requests.ConnectionError:
            st.error("Impossible de contacter l'API.")

# ── Tab 3 : Top Produits ─────────────────────────────────────────────────────
with tab3:
    st.header("Top Produits Populaires")

    try:
        resp = requests.get(f"{API_URL}/health", timeout=5)
        if resp.status_code == 200:
            health = resp.json()
            col1, col2, col3 = st.columns(3)
            col1.metric("Produits", f"{health['products']:,}")
            col2.metric("Utilisateurs", f"{health['users']:,}")
            col3.metric("Produits populaires", f"{health['popular_products']:,}")
    except requests.ConnectionError:
        st.info("API non disponible — les métriques seront affichées au démarrage.")

    product_id = st.text_input("Product ID pour produits similaires", placeholder="ex: B00YQ6...")
    n_similar = st.slider("Nombre de produits similaires", 1, 20, 5)

    if st.button("Trouver les produits similaires"):
        if product_id:
            try:
                resp = requests.get(
                    f"{API_URL}/similar/{product_id}",
                    params={"n": n_similar},
                    timeout=5,
                )
                if resp.status_code == 200:
                    data = resp.json()
                    st.write(f"Latence : **{data['latency_ms']} ms**")
                    similar = pd.DataFrame(data["similar_products"])
                    st.dataframe(similar, use_container_width=True)
                elif resp.status_code == 404:
                    st.warning("Produit non trouvé.")
                else:
                    st.error(f"Erreur : {resp.status_code}")
            except requests.ConnectionError:
                st.error("Impossible de contacter l'API.")
