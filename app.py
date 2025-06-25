import streamlit as st
import os
import numpy as np
import pickle
from collections import deque
from sklearn.linear_model import SGDRegressor

st.set_page_config(page_title="Prédicteur de valeurs", layout="centered")
st.title("🔢 Prédicteur de valeurs (à 2 décimales)")

st.subheader("📁 Choisir ou créer un projet")
projects_dir = "projets"
os.makedirs(projects_dir, exist_ok=True)

projets = [f.replace(".pkl", "") for f in os.listdir(projects_dir) if f.endswith(".pkl")]
choix_projet = st.selectbox("Sélectionner un projet existant", [""] + projets)
nouveau_projet = st.text_input("Ou créer un nouveau projet")

projet = nouveau_projet if nouveau_projet else choix_projet
modele_path = os.path.join(projects_dir, f"{projet}.pkl") if projet else None

if not projet:
    st.warning("Veuillez sélectionner ou créer un projet pour continuer.")
    st.stop()

if os.path.exists(modele_path):
    with open(modele_path, "rb") as f:
        modele, historique = pickle.load(f)
else:
    modele = SGDRegressor(max_iter=1000, tol=1e-3)
    historique = deque(maxlen=30)

st.subheader(f"🔢 Prédiction pour le projet : `{projet}`")

if len(historique) < 30:
    val = st.number_input(f"Entrez la valeur {len(historique)+1}/30", format="%.2f")
    if st.button("Ajouter cette valeur"):
        historique.append(val)
        st.experimental_rerun()
else:
    entree = np.array(historique).reshape(1, -1)
    prediction = modele.predict(entree)[0]
    st.success(f"✅ Prochaine valeur prédite : **{prediction:.2f}**")

    nouvelle_val = st.number_input("Entrez la vraie valeur obtenue", format="%.2f")
    if st.button("Apprendre cette valeur réelle"):
        modele.partial_fit(entree, [nouvelle_val])
        historique.append(nouvelle_val)
        st.success("✔️ Valeur ajoutée et modèle mis à jour")
        st.experimental_rerun()

if st.button("💾 Sauvegarder le modèle actuel"):
    with open(modele_path, "wb") as f:
        pickle.dump((modele, historique), f)
    st.success(f"🧠 Modèle sauvegardé dans `{modele_path}`")

if historique:
    st.markdown("### 📜 Historique des dernières valeurs")
    st.write([round(x, 2) for x in list(historique)])