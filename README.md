# Prédiction Pluviométrique au Sénégal

Ce projet implémente un système de prédiction pluviométrique pour le Sénégal, capable de générer des prévisions à court terme (pour le lendemain) et à long terme (sur 10 ans). Les prédictions aident à anticiper les risques d'inondation ou de sécheresse et à optimiser la gestion des ressources en eau.

## Structure du projet

```
/BigData/
|-- data/                      # Données pluviométriques
|-- src/                       # Code source modulaire
|   |-- utils.py               # Fonctions utilitaires
|   |-- data_processing.py     # Traitement des données
|   |-- feature_engineering.py # Ingénierie des caractéristiques
|   |-- model_training.py      # Entraînement des modèles
|   |-- prediction.py          # Logique de prédiction
|-- models/                    # Modèles sauvegardés
|-- output/                    # Résultats et visualisations
|   |-- predictions/           # Visualisations de prédictions
|-- generate_data.py           # Génération des données pluviométriques synthétiques
|-- main.py                    # Point d'entrée principal (approche modulaire)
|-- predict.py                 # Interface CLI de prédiction à court terme
|-- predict_long_term.py       # Prédiction à long terme (10 ans) avec Prophet
```

## Description détaillée des fichiers

### Modules principaux (dossier src/)

1. **utils.py**
   - **Rôle** : Contient des fonctions utilitaires utilisées par les autres modules.
   - **Fonctionnalités** : Installation de packages, vérification de répertoires, formatage de dates, classification des précipitations.
   - **Appelé par** : Tous les autres modules du projet.

2. **data_processing.py**
   - **Rôle** : Gère le chargement et le nettoyage des données.
   - **Fonctionnalités** : Définition du schéma, lecture des CSV, traitement des valeurs manquantes, analyse exploratoire.
   - **Appelé par** : main.py, model_training.py.
   - **Dépendances** : utils.py.

3. **feature_engineering.py**
   - **Rôle** : Préparation des caractéristiques pour l'entraînement des modèles.
   - **Fonctionnalités** : Ajout de variables temporelles (mois, jour, saison), encodage des variables catégorielles, normalisation.
   - **Appelé par** : model_training.py, prediction.py.
   - **Dépendances** : utils.py.

4. **model_training.py**
   - **Rôle** : Entraîne les différents modèles prédictifs.
   - **Fonctionnalités** : Entraînement Random Forest, GBT, et Prophet, évaluation des performances, sauvegarde des modèles.
   - **Appelé par** : main.py.
   - **Dépendances** : data_processing.py, feature_engineering.py.

5. **prediction.py**
   - **Rôle** : Utilise les modèles entraînés pour faire des prédictions.
   - **Fonctionnalités** : Chargement des modèles, préparation des données, prédiction, visualisation des résultats.
   - **Appelé par** : main.py.
   - **Dépendances** : feature_engineering.py, utils.py.

### Points d'entrée et scripts

1. **generate_data.py**
   - **Rôle** : Génération des données pluviométriques synthétiques pour le Sénégal.
   - **Fonctionnalités** : Création de données réalistes par région, modélisation des variations saisonnières et géographiques.
   - **Comment l'utiliser** : `python generate_data.py`.
   - **Produit** : Le fichier de données `data/donnees_pluviometriques_senegal.csv` nécessaire pour l'entraînement.

2. **main.py**
   - **Rôle** : Point d'entrée principal du projet avec interface par ligne de commande.
   - **Fonctionnalités** : Traitement des données, entraînement des modèles, prédictions via les modules src/.
   - **Comment l'utiliser** : `python main.py [process_data|train_models|predict]`.
   - **Appelle** : Tous les modules dans src/.

2. **predict.py**
   - **Rôle** : Interface en ligne de commande complète et flexible pour les prédictions.
   - **Fonctionnalités** : Prédiction pour une ou toutes les régions, pour n'importe quelle date, création de visualisations.
   - **Comment l'utiliser** : `python predict.py [--region REGION] [--date DATE] [--jours JOURS]`.
   - **Indépendant** : Contient sa propre logique de prédiction sans dépendre des modules src/.

3. **predict_long_term.py**
   - **Rôle** : Génère des prédictions pluviométriques à long terme (10 ans par défaut).
   - **Fonctionnalités** : Utilise le modèle Prophet pour prédire les tendances futures, analyse des tendances et cycles saisonniers.
   - **Comment l'utiliser** : `python predict_long_term.py [--region REGION] [--annees ANNEES]`.
   - **Dépendances** : Nécessite Prophet et pandas pour l'analyse des séries temporelles.

## Relations entre les fichiers

1. **Approche modulaire standard**
   - main.py → src/modules → models/ (sauvegarde/chargement)
   - Cette approche utilise pleinement PySpark et les modèles ML avancés.

2. **Approche simplifiée**
   - generate_data.py → génération des données synthétiques initiales
   - predict.py → prédictions à court terme pour une ou toutes les régions
   - predict_long_term.py → prédictions à long terme avec Prophet
   - Ces scripts utilisent pandas pour la manipulation des données et sont indépendants de l'approche PySpark

## Installation

### Prérequis
- Python 3.8+
- Apache Spark 3.1+
- PySpark
- Pandas
- NumPy
- Matplotlib
- Prophet (pour les prédictions à long terme)

### Installation des dépendances
Créer un environnement virtuel et activer-le :

```bash
python -m venv venv
source venv/bin/activate  # Sur Linux/Mac
venv\Scripts\activate    # Sur Windows
```

```bash
pip install pyspark pandas numpy matplotlib prophet
```

### Configuration initiale

**IMPORTANT** : Avant d'exécuter le reste du projet, vous devez d'abord générer les données pluviométriques synthétiques :

```bash
# Générer les données pluviométriques
python generate_data.py
```

Cette commande crée le fichier `data/donnees_pluviometriques_senegal.csv` qui est indispensable pour l'entraînement des modèles et les prédictions.

## Utilisation

### 1. Traitement des données et entraînement des modèles

```bash
python main.py process_data
python main.py train_models
```

### 2. Faire des prédictions à court terme

#### Prédiction pour demain

#### Interface de ligne de commande complète
```bash
# Prédiction pour toutes les régions (demain)
python predict.py

# Prédiction pour une région spécifique
python predict.py --region Dakar

# Prédiction pour une date spécifique
python predict.py --date 2025-06-15

# Prédiction pour plusieurs jours dans le futur
python predict.py --jours 5

# Combinaison des options
python predict.py --region Ziguinchor --date 2025-07-15 --output output/resultats_ete
```

### 3. Faire des prédictions à long terme

Le script `predict_long_term.py` permet de générer des prédictions sur plusieurs années (10 ans par défaut) en utilisant le modèle Prophet. Il produit des visualisations et des analyses de tendances.

```bash
# Prédiction à long terme pour toutes les régions (10 ans)
python predict_long_term.py

# Prédiction à long terme pour une région spécifique
python predict_long_term.py --region Dakar

# Prédiction sur une période personnalisée (ex: 20 ans)
python predict_long_term.py --region Thiès --annees 20

# Personnalisation du répertoire de sortie
python predict_long_term.py --output output/tendances_futures
```

Les visualisations générées incluent :
- Graphique de prédiction sur la période spécifiée avec intervalles de confiance
- Analyse du pattern mensuel des précipitations (saisonnalité)
- Rapport détaillé sur les tendances et les années/mois extrêmes

## Approches de modélisation

### Modèles à court terme
- **Random Forest** : Utilisé pour les prédictions quotidiennes, exploite plusieurs variables météorologiques
- **Gradient Boosted Trees** : Alternative au Random Forest, offre souvent de meilleures performances

### Modèle à long terme
- **Prophet** : Modèle de Facebook Research pour les prédictions à long terme avec saisonnalité annuelle
  - Capture les tendances non-linéaires avec saisonnalité
  - Capable de détecter les changements annuels et mensuels
  - Génère des intervalles de confiance pour quantifier l'incertitude
  - Particulièrement adapté aux séries temporelles avec variations saisonnières comme les précipitations

## Évaluation des performances

Les modèles ont été évalués en utilisant :
- Le coefficient de détermination (R²)
- L'erreur quadratique moyenne (RMSE)
- L'erreur absolue moyenne (MAE)

Les modèles à court terme ont atteint un R² d'environ 0.74, indiquant une bonne capacité prédictive.

## Défis et solutions

### Problèmes rencontrés
- Valeurs manquantes dans les données météorologiques
- Erreurs de type dans le DataFrame
- Incompatibilité entre les features d'entraînement et de prédiction

### Solutions implémentées
- Nettoyage robuste des données
- Vérification des types avant l'analyse
- Approche simplifiée basée sur les moyennes historiques et tendances annuelles

## Extensions futures

1. **Intégration de données en temps réel** : Connecter l'API à des services météorologiques
2. **Interface web** : Développer une interface utilisateur accessible via navigateur
3. **Amélioration des modèles** : Incorporer des variables exogènes comme El Niño/La Niña
4. **Alertes automatisées** : Système de notification pour les événements pluviométriques extrêmes
