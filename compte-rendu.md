# 📘 GRAND GUIDE : ANALYSE D’UN DATASET — WORLD DEVELOPMENT INDICATORS

*Compte rendu inspiré du document “Correction Projet.md”*

Ce rapport décortique étape par étape un projet complet d’analyse et de modélisation à partir du dataset **World Development Indicators**, un jeu de données majeur regroupant des statistiques socio-économiques et démographiques provenant de la Banque Mondiale.

---

# 1. 🎯 Contexte Métier et Mission

## 🌍 Le Problème (Business Case)

Les décideurs publics, ONG, économistes et institutions internationales doivent comprendre :

* Pourquoi certains pays progressent plus vite que d’autres ?
* Quels indicateurs expliquent réellement le développement humain ?
* Comment prédire la croissance ou identifier les zones à risque ?

Le dataset WDI regroupe **plus de 1 500 indicateurs** pour des centaines de pays sur plusieurs années (PIB, mortalité, éducation, investissement, émissions CO₂, etc.).

### 🎯 Objectif du projet

Construire un système d’analyse et de modélisation permettant de :

1. **Nettoyer et préparer** les indicateurs (dataset souvent incomplet).
2. **Explorer les dynamiques clés du développement** (EDA).
3. **Construire un modèle prédictif**, par exemple :

   * prédire **le PIB/habitant**,
   * ou prédire le **niveau de développement (basse/moyenne/haute catégorie)**.

Le but est de transformer des données massives en **intuition économique**.

### 🧩 Les Données (Input)

Votre dataset contient généralement les colonnes suivantes :

* **Country Name**
* **Country Code**
* **Indicator Name**
* **Indicator Code**
* **Year**
* **Value**

Chaque ligne représente **un indicateur pour un pays donné à une année donnée**.

---

# 2. 🧪 Le Code Python (Laboratoire)

Le code standard pour un tel projet suit les étapes suivantes :

```python
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.impute import SimpleImputer
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score

# Chargement
df = pd.read_csv("world_development_indicators.csv")

# Préparation : sélection d’un indicateur (ex. PIB/habitant)
gdp = df[df["Indicator Name"] == "GDP per capita (current US$)"]

# Pivot : pays × années
gdp_pivot = gdp.pivot(index="Country Name", columns="Year", values="Value")

# Nettoyage
imputer = SimpleImputer(strategy="mean")
gdp_clean = pd.DataFrame(imputer.fit_transform(gdp_pivot),
                         columns=gdp_pivot.columns,
                         index=gdp_pivot.index)

# Variable cible : année récente
y = gdp_clean[2020]
X = gdp_clean.drop(columns=[2020])

# Split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Modèle
model = RandomForestRegressor()
model.fit(X_train, y_train)

# Prédiction
y_pred = model.predict(X_test)

print("R2 :", r2_score(y_test, y_pred))
print("RMSE :", mean_squared_error(y_test, y_pred, squared=False))
```

Cette base est ensuite enrichie selon le besoin : visualisations, corrélations, analyses avancées.

---

# 3. 🧼 Analyse Approfondie : Nettoyage (Data Wrangling)

## 🔍 Problèmes caractéristiques du dataset WDI

1. **Beaucoup de valeurs manquantes**
   Certains indicateurs sont renseignés seulement pour certains pays ou certaines années.

2. **Données en format long**
   Chaque ligne = un indicateur pour un pays → nécessite un pivot.

3. **Unités et échelles différentes**
   Par exemple, un indicateur peut être en dollars, un autre en pourcentage.

4. **Pays disparus, changement de codes (ex : Soudan/Soudan du Sud)**.

---

## 🧠 Technique de Nettoyage

### 👉 Pivot (Réorganisation)

On passe de :

| Country | Year | Value |
| ------- | ---- | ----- |
| Maroc   | 2015 | 2970  |

À :

| Country | 2010 | 2011 | 2012 | ... |
| ------- | ---- | ---- | ---- | --- |
| Maroc   | 2870 | 2910 | ...  | ... |

### 👉 Imputation (SimpleImputer)

Comme dans “Correction Projet.md”, on utilise :

```
SimpleImputer(strategy='mean')
```

1. **fit()** : calcule la moyenne de chaque colonne année.
2. **transform()** : remplit chaque année manquante par la moyenne des pays pour cette année.

### 💡 Coin de l’Expert : Data Leakage

Comme expliqué dans le document source :

❗ Il faut **séparer Train/Test avant d’imputer**, sinon la moyenne est influencée par les données test → fuite d’information.

---

# 4. 🔎 Analyse Exploratoire (EDA)

### 👉 Questions explorées

* Comment évolue le PIB/habitant ?
* Quels pays ont les croissances les plus volatiles ?
* Quels indicateurs corrèlent le plus avec le développement ?

### 📊 Visualisations typiques

#### 1. Courbe PIB/habitant pour un pays

Tendance sur 20 ans → croissance, stagnation, choc.

#### 2. Heatmap des corrélations

Certaines variables fortes :

* Éducation ↗ PIB
* Espérance de vie ↗ PIB
* Emissions CO₂ ↗ industrialisation
* Inflation ↘ stabilité économique

#### 3. Boxplots pour comparer régions (MENA, Sub-Saharan, EU)

---

# 5. 🧪 Méthodologie (Train/Test Split)

Exactement comme dans le document modèle :

* **Objectif :** généraliser, pas mémoriser.
* **Split 80/20** recommandé.
* **random_state=42** pour reproductibilité.

---

# 6. 🌲 Focus Théorique : Random Forest (Régression)

La logique suit la même structure que dans le corrigé.

### Pourquoi Random Forest est idéal ici ?

1. **Tolère les données bruitées.**
2. **Gère bien les non-linéarités économiques.**
3. **Capte les interactions entre indicateurs.**

### Fonctionnement rapide :

* **Bagging** : chaque arbre voit une version légèrement différente des données.
* **Feature randomness** : chaque arbre utilise un sous-ensemble d’indicateurs.
* **Consensus** : la forêt vote → stabilité.

---

# 7. 📈 Évaluation du Modèle

### Pour un modèle de régression, on utilise :

#### 🟪 R²

Pourcentage de variance expliquée.

> Un bon modèle sur WDI se situe entre **0.70 et 0.85**.

#### 🟦 RMSE

Erreur moyenne de prédiction en “dollars” (pour le PIB).

#### 🟥 Visualisation :

« Courbe réelle vs prédite ».

---

# Conclusion du Projet

Ce projet montre que l’analyse socio-économique requiert :

1. **Une compréhension fine des indicateurs.**
2. **Un pipeline rigoureux** (nettoyage → exploration → modélisation).
3. **Un modèle robuste** (Random Forest) capable de résumer la complexité des dynamiques mondiales.
4. **Une lecture experte des résultats** pour guider les politiques publiques ou les analyses financières.

Avec le dataset WDI, on passe de milliers de lignes brutes à une **vision claire, structurée et interprétable du développement mondial.**



envoie-moi simplement ton code Python ou quelques extraits du dataset.

