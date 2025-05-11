import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from tensorflow.keras.models import load_model
from sklearn.preprocessing import MinMaxScaler

st.title("Demand Forecasting Using LSTM")

@st.cache_data
def load_data():
    months = np.arange(1, 97)
    demand = 100 + 10 * np.sin(2 * np.pi * months / 12) + np.random.normal(0, 5, 96)
    dates = pd.date_range(start='2016-01-01', periods=96, freq='M')
    return pd.DataFrame({'Date': dates, 'Demand': demand})

df = load_data()
st.line_chart(df.set_index('Date'))

scaler = MinMaxScaler()
scaled_data = scaler.fit_transform(df[['Demand']])
SEQ_LENGTH = 12

def create_sequences(data, seq_length):
    X = []
    for i in range(len(data) - seq_length):
        X.append(data[i:i+seq_length])
    return np.array(X)

X = create_sequences(scaled_data, SEQ_LENGTH)

try:
    model = load_model("lstm_model.h5")
    y_pred = model.predict(X)
    y_pred_inv = scaler.inverse_transform(y_pred)

    fig, ax = plt.subplots()
    ax.plot(df['Date'][SEQ_LENGTH:], df['Demand'][SEQ_LENGTH:], label='Actual Demand')
    ax.plot(df['Date'][SEQ_LENGTH:], y_pred_inv, label='Predicted Demand')
    ax.set_title("Actual vs Predicted Demand")
    ax.set_xlabel("Date")
    ax.set_ylabel("Demand")
    ax.legend()
    st.pyplot(fig)
except OSError:
    st.warning("Please train and save your model as 'lstm_model.h5' in the project folder.")
