
import streamlit as st
import numpy as np
import pandas as pd
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from sklearn.preprocessing import MinMaxScaler

st.title("🔮 Prédicteur de séries temporelles (LSTM)")
st.markdown("Saisis des valeurs (ex : 1.05, 2.3...) pour générer des prédictions avec un modèle LSTM.")

if "data" not in st.session_state:
    st.session_state.data = []

value = st.text_input("➕ Ajouter une nouvelle valeur :", placeholder="ex: 1.23")
if st.button("Ajouter"):
    try:
        num = float(value.replace(",", "."))
        st.session_state.data.append(num)
        st.success(f"Ajouté : {num}")
    except ValueError:
        st.error("Entrez un nombre valide.")

if st.button("🗑️ Réinitialiser la série"):
    st.session_state.data = []
    st.success("Série réinitialisée.")

n_preds = st.slider("🔮 Combien de valeurs à prédire ?", 1, 50, 5)

data = st.session_state.data
if len(data) >= 10:
    st.line_chart(data, height=200, use_container_width=True)

    series = np.array(data).reshape(-1, 1)
    scaler = MinMaxScaler()
    scaled = scaler.fit_transform(series)

    X, y = [], []
    for i in range(len(scaled) - 10):
        X.append(scaled[i:i+10])
        y.append(scaled[i+10])
    X, y = np.array(X), np.array(y)

    model = Sequential([
        LSTM(64, activation='relu', input_shape=(X.shape[1], X.shape[2])),
        Dropout(0.2),
        Dense(1)
    ])
    model.compile(optimizer='adam', loss='mse')
    model.fit(X, y, epochs=50, verbose=0)

    last_input = scaled[-10:].reshape(1, 10, 1)
    preds = []
    for _ in range(n_preds):
        pred = model.predict(last_input)[0][0]
        preds.append(pred)
        last_input = np.append(last_input[:, 1:, :], [[[pred]]], axis=1)

    preds_rescaled = scaler.inverse_transform(np.array(preds).reshape(-1, 1)).flatten()
    st.subheader("📈 Prédictions")
    st.line_chart(np.concatenate([data, preds_rescaled]))

    df = pd.DataFrame({
        "Valeurs Saisies": data + [None]*n_preds,
        "Prédictions": [None]*len(data) + list(preds_rescaled)
    })
    csv = df.to_csv(index=False).encode("utf-8")
    st.download_button("⬇️ Télécharger CSV", csv, "predictions.csv", "text/csv")
