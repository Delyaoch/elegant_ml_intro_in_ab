import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler

st.set_page_config(page_title="Car Price Predictor", layout="wide")
st.title("Car Price Predictor")

def clean_col(x):
    if pd.isna(x):
        return np.nan
    try:
        return float(str(x).split()[0].replace(',', ''))
    except:
        return np.nan

uploaded_file = st.file_uploader("Загрузите cars_train.csv", type=["csv"])

if uploaded_file:
    df = pd.read_csv(uploaded_file)
    
    for col in ['mileage', 'engine', 'max_power']:
        df[col] = df[col].apply(clean_col)
        df[col] = df[col].fillna(df[col].median())
    
    features = ['year', 'km_driven', 'engine', 'max_power']
    X = df[features]
    y = df['selling_price']
    
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    model = LinearRegression().fit(X_scaled, y)
    
    st.header("Exploratory Data Analysis")
    col1, col2 = st.columns(2)
    with col1:
        fig, ax = plt.subplots()
        sns.histplot(df['selling_price'], bins=50, kde=True, ax=ax)
        st.pyplot(fig)
    with col2:
        fig, ax = plt.subplots()
        sns.scatterplot(data=df, x='max_power', y='selling_price', alpha=0.5, ax=ax)
        st.pyplot(fig)
    
    st.header("Price Prediction")
    col1, col2, col3, col4 = st.columns(4)
    with col1: year = st.number_input("Year", 1990, 2025, 2015)
    with col2: km = st.number_input("Km Driven", 0, 500000, 50000)
    with col3: engine = st.number_input("Engine (cc)", 500, 8000, 1500)
    with col4: power = st.number_input("Power (bhp)", 30, 1000, 100)
    
    if st.button("Predict"):
        X_input = scaler.transform([[year, km, engine, power]])
        price = model.predict(X_input)[0]
        st.success(f"Predicted Price: {price:,.0f}")
    
    st.header("Feature Importance")
    weights = pd.DataFrame({'Feature': features, 'Weight': model.coef_}).sort_values('Weight', key=abs)
    fig, ax = plt.subplots()
    colors = ['#FF69B4' if w > 0 else '#FFB6C1' for w in weights['Weight']]
    ax.barh(weights['Feature'], weights['Weight'], color=colors)
    ax.axvline(0, color='black')
    st.pyplot(fig)

else:
    st.info("Загрузите файл cars_train.csv для начала работы")