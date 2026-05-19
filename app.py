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

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score
)

# ------------------------------------------------------------
# CONFIGURATION PAGE
# ------------------------------------------------------------

st.set_page_config(
    page_title="Heart Disease Prediction",
    page_icon="🫀",
    layout="wide"
)

# ------------------------------------------------------------
# CHARGEMENT DES DONNÉES
# ------------------------------------------------------------

@st.cache_data
def load_data():

    url = "https://archive.ics.uci.edu/ml/machine-learning-databases/heart-disease/processed.cleveland.data"

    colonnes = [
        'age', 'sex', 'cp', 'trestbps', 'chol',
        'fbs', 'restecg', 'thalach', 'exang',
        'oldpeak', 'slope', 'ca', 'thal', 'target'
    ]

    # Chargement du dataset
    df = pd.read_csv(url, names=colonnes, na_values='?')

    # Conversion en numérique
    df = df.apply(pd.to_numeric, errors='coerce')

    # Conversion de la cible
    df['target'] = df['target'].apply(
        lambda x: 1 if x > 0 else 0
    )

    # Remplissage des valeurs manquantes
    for col in df.columns:
        df[col] = df[col].fillna(df[col].median())

    return df

# ------------------------------------------------------------
# ENTRAINEMENT DES MODÈLES
# ------------------------------------------------------------

@st.cache_resource
def train_models(df):

    X = df.drop('target', axis=1)
    y = df['target']

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=42,
        stratify=y
    )

    # Standardisation
    scaler = StandardScaler()

    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # Modèles
    modeles = {
        'KNN': KNeighborsClassifier(),

        'SVM': SVC(
            probability=True,
            random_state=42
        ),

        'Logistic Regression': LogisticRegression(
            max_iter=1000,
            random_state=42
        ),

        'Random Forest': RandomForestClassifier(
            random_state=42
        ),

        'AdaBoost': AdaBoostClassifier(
            n_estimators=50,
            random_state=42
        ),

        'Decision Tree': DecisionTreeClassifier(
            random_state=42
        )
    }

    resultats = {}

    # Entrainement
    for nom, modele in modeles.items():

        modele.fit(X_train_scaled, y_train)

        y_pred = modele.predict(X_test_scaled)

        y_prob = modele.predict_proba(
            X_test_scaled
        )[:, 1]

        resultats[nom] = {
            'Accuracy': accuracy_score(y_test, y_pred),
            'Précision': precision_score(y_test, y_pred),
            'Rappel': recall_score(y_test, y_pred),
            'F1-Score': f1_score(y_test, y_pred),
            'AUC-ROC': roc_auc_score(y_test, y_prob)
        }

    return modeles, scaler, resultats

# ------------------------------------------------------------
# APPLICATION
# ------------------------------------------------------------

st.title("🫀 Prédiction de Maladie Cardiaque")
st.markdown("### Projet IA - IFOAD | Haida")

# Chargement des données
df = load_data()

# Vérification des NaN
if df.isnull().sum().sum() > 0:
    st.error("⚠️ Des valeurs manquantes existent encore.")
else:
    st.success("✅ Données chargées correctement")

# Entrainement
modeles, scaler, resultats = train_models(df)

# ------------------------------------------------------------
# MENU LATÉRAL
# ------------------------------------------------------------

menu = st.sidebar.selectbox(
    "Navigation",
    [
        "Accueil",
        "Exploration des données",
        "Comparaison des modèles",
        "Prédiction"
    ]
)

# ------------------------------------------------------------
# PAGE ACCUEIL
# ------------------------------------------------------------

if menu == "Accueil":

    st.header("Bienvenue !")

    st.write(
        """
        Cette application utilise plusieurs modèles
        de Machine Learning pour prédire le risque
        de maladie cardiaque.
        """
    )

    col1, col2, col3 = st.columns(3)

    col1.metric("Patients", len(df))
    col2.metric("Variables", 13)

    meilleur_modele = max(
        resultats,
        key=lambda x: resultats[x]['Accuracy']
    )

    meilleure_accuracy = resultats[meilleur_modele]['Accuracy']

    col3.metric(
        "Meilleur modèle",
        f"{meilleur_modele} ({meilleure_accuracy:.2%})"
    )

# ------------------------------------------------------------
# EXPLORATION DES DONNÉES
# ------------------------------------------------------------

elif menu == "Exploration des données":

    st.header("Exploration des données")

    st.subheader("Aperçu du dataset")

    st.dataframe(df.head(10))

    st.subheader("Valeurs manquantes")

    st.write(df.isnull().sum())

    st.subheader("Distribution des patients")

    fig, ax = plt.subplots()

    df['target'].value_counts().plot(
        kind='bar',
        ax=ax
    )

    ax.set_xticklabels(
        ['Sain', 'Malade'],
        rotation=0
    )

    ax.set_title("Distribution des patients")

    st.pyplot(fig)

# ------------------------------------------------------------
# COMPARAISON DES MODÈLES
# ------------------------------------------------------------

elif menu == "Comparaison des modèles":

    st.header("Comparaison des algorithmes")

    df_res = pd.DataFrame(resultats).T.round(3)

    st.dataframe(
        df_res.style.highlight_max(
            color='lightgreen'
        )
    )

    fig, ax = plt.subplots(figsize=(10, 5))

    df_res['Accuracy'].sort_values().plot(
        kind='barh',
        ax=ax
    )

    ax.set_title("Accuracy des modèles")
    ax.set_xlabel("Accuracy")

    st.pyplot(fig)

# ------------------------------------------------------------
# PRÉDICTION
# ------------------------------------------------------------

elif menu == "Prédiction":

    st.header("Prédire pour un nouveau patient")

    col1, col2 = st.columns(2)

    with col1:

        age = st.slider("Âge", 20, 80, 50)

        sex = st.selectbox(
            "Sexe",
            [0, 1],
            format_func=lambda x:
            "Femme" if x == 0 else "Homme"
        )

        cp = st.slider(
            "Type douleur thoracique",
            0, 3, 1
        )

        trestbps = st.slider(
            "Pression artérielle",
            80, 200, 130
        )

        chol = st.slider(
            "Cholestérol",
            100, 600, 250
        )

        fbs = st.selectbox(
            "Glycémie > 120",
            [0, 1],
            format_func=lambda x:
            "Non" if x == 0 else "Oui"
        )

        restecg = st.slider(
            "ECG repos",
            0, 2, 1
        )

    with col2:

        thalach = st.slider(
            "Fréquence cardiaque max",
            60, 220, 150
        )

        exang = st.selectbox(
            "Angine effort",
            [0, 1],
            format_func=lambda x:
            "Non" if x == 0 else "Oui"
        )

        oldpeak = st.slider(
            "Dépression ST",
            0.0, 6.0, 1.0
        )

        slope = st.slider(
            "Pente ST",
            0, 2, 1
        )

        ca = st.slider(
            "Nombre de vaisseaux",
            0, 3, 0
        )

        thal = st.selectbox(
            "Thalassémie",
            [3, 6, 7],
            format_func=lambda x:
            "Normal" if x == 3
            else "Défaut fixe"
            if x == 6
            else "Défaut réversible"
        )

        modele_choisi = st.selectbox(
            "Algorithme",
            list(modeles.keys())
        )

    # --------------------------------------------------------
    # BOUTON PRÉDICTION
    # --------------------------------------------------------

    if st.button("Prédire"):

        patient = np.array([[
            age,
            sex,
            cp,
            trestbps,
            chol,
            fbs,
            restecg,
            thalach,
            exang,
            oldpeak,
            slope,
            ca,
            thal
        ]])

        # Standardisation
        patient_scaled = scaler.transform(patient)

        # Prédiction
        prediction = modeles[
            modele_choisi
        ].predict(patient_scaled)[0]

        proba = modeles[
            modele_choisi
        ].predict_proba(patient_scaled)[0][1]

        st.subheader("Résultat")

        if prediction == 1:

            st.error(
                f"""
                ⚠️ Risque de maladie cardiaque détecté !

                Probabilité : {proba * 100:.1f}%
                """
            )

        else:

            st.success(
                f"""
                ✅ Pas de maladie cardiaque détectée

                Probabilité : {(1 - proba) * 100:.1f}%
                """
            )
