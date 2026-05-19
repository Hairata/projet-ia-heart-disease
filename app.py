# ============================================================
# PROJET IA - IFOAD | Haida
# Application Streamlit - Heart Disease Prediction
# ============================================================

import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score

# ---- Chargement des données ----
@st.cache_data
def load_data():
    url = "https://archive.ics.uci.edu/ml/machine-learning-databases/heart-disease/processed.cleveland.data"
    colonnes = ['age','sex','cp','trestbps','chol','fbs',
                'restecg','thalach','exang','oldpeak','slope','ca','thal','target']
    df = pd.read_csv(url, names=colonnes, na_values='?')
    df['target'] = df['target'].apply(lambda x: 1 if x > 0 else 0)
    df['ca'].fillna(df['ca'].median(), inplace=True)
    df['thal'].fillna(df['thal'].median(), inplace=True)
    return df

# ---- Entraînement des modèles ----
@st.cache_resource
def train_models(df):
    X = df.drop('target', axis=1)
    y = df['target']
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42)
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)
    modeles = {
        'KNN': KNeighborsClassifier(),
        'SVM': SVC(probability=True, random_state=42),
        'Logistic Regression': LogisticRegression(random_state=42),
        'Random Forest': RandomForestClassifier(random_state=42),
        'AdaBoost' : AdaBoostClassifier(algorithm="SAMME", random_state=42)
        'Decision Tree': DecisionTreeClassifier(random_state=42)
    }
    resultats = {}
    for nom, modele in modeles.items():
        modele.fit(X_train, y_train)
        y_pred = modele.predict(X_test)
        y_prob = modele.predict_proba(X_test)[:, 1]
        resultats[nom] = {
            'Accuracy': accuracy_score(y_test, y_pred),
            'Précision': precision_score(y_test, y_pred),
            'Rappel': recall_score(y_test, y_pred),
            'F1-Score': f1_score(y_test, y_pred),
            'AUC-ROC': roc_auc_score(y_test, y_prob)
        }
    return modeles, scaler, resultats

# ---- Interface ----
st.title("🫀 Prédiction de Maladie Cardiaque")
st.markdown("**Projet IA - IFOAD | Haida**")

df = load_data()
modeles, scaler, resultats = train_models(df)

# Menu de navigation
menu = st.sidebar.selectbox("Navigation", 
    ["Accueil", "Exploration des données", 
     "Comparaison des modèles", "Prédiction"])

if menu == "Accueil":
    st.header("Bienvenue !")
    st.write("Cette application prédit si un patient a une maladie cardiaque.")
    col1, col2, col3 = st.columns(3)
    col1.metric("Patients", "303")
    col2.metric("Variables", "13")
    col3.metric("Meilleur modèle", "KNN 91.8%")

elif menu == "Exploration des données":
    st.header("Exploration des données")
    st.dataframe(df.head(10))
    fig, ax = plt.subplots()
    df['target'].value_counts().plot(kind='bar', color=['#2ecc71','#e74c3c'], ax=ax)
    ax.set_xticklabels(['Sain', 'Malade'], rotation=0)
    ax.set_title("Distribution des patients")
    st.pyplot(fig)

elif menu == "Comparaison des modèles":
    st.header("Comparaison des 6 algorithmes")
    df_res = pd.DataFrame(resultats).T.round(3)
    st.dataframe(df_res.style.highlight_max(color='lightgreen'))
    fig, ax = plt.subplots(figsize=(10, 5))
    df_res['Accuracy'].sort_values().plot(kind='barh', color='#3498db', ax=ax)
    ax.set_title("Accuracy par algorithme")
    ax.set_xlabel("Accuracy")
    st.pyplot(fig)

elif menu == "Prédiction":
    st.header("Prédire pour un nouveau patient")
    col1, col2 = st.columns(2)
    with col1:
        age = st.slider("Âge", 20, 80, 50)
        sex = st.selectbox("Sexe", [0, 1], format_func=lambda x: "Femme" if x==0 else "Homme")
        cp = st.slider("Type de douleur thoracique (0-3)", 0, 3, 1)
        trestbps = st.slider("Pression artérielle au repos", 80, 200, 130)
        chol = st.slider("Cholestérol (mg/dl)", 100, 600, 250)
        fbs = st.selectbox("Glycémie > 120 mg/dl", [0, 1], 
                          format_func=lambda x: "Non" if x==0 else "Oui")
        restecg = st.slider("Résultat ECG au repos (0-2)", 0, 2, 1)
    with col2:
        thalach = st.slider("Fréquence cardiaque max", 60, 220, 150)
        exang = st.selectbox("Angine à l'effort", [0, 1],
                            format_func=lambda x: "Non" if x==0 else "Oui")
        oldpeak = st.slider("Dépression ST", 0.0, 6.0, 1.0)
        slope = st.slider("Pente ST (0-2)", 0, 2, 1)
        ca = st.slider("Nombre de vaisseaux (0-3)", 0, 3, 0)
        thal = st.selectbox("Thalassémie", [3, 6, 7],
                           format_func=lambda x: "Normal" if x==3 else "Défaut fixe" if x==6 else "Défaut réversible")
        modele_choisi = st.selectbox("Algorithme", list(modeles.keys()))

    if st.button("Prédire"):
        patient = np.array([[age, sex, cp, trestbps, chol, fbs,
                            restecg, thalach, exang, oldpeak, slope, ca, thal]])
        patient_scaled = scaler.transform(patient)
        prediction = modeles[modele_choisi].predict(patient_scaled)[0]
        proba = modeles[modele_choisi].predict_proba(patient_scaled)[0][1]
        if prediction == 1:
            st.error(f"⚠️ Risque de maladie cardiaque détecté ! (Probabilité : {proba*100:.1f}%)")
        else:
            st.success(f"✅ Pas de maladie cardiaque détectée (Probabilité : {proba*100:.1f}%)")
