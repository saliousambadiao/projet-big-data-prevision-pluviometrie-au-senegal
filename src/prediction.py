"""
Prédiction pluviométrique au Sénégal
=====================================
Ce module utilise les modèles entraînés pour effectuer des prédictions à court et long terme.
"""

import os
import sys
import findspark
findspark.init()

from pyspark.sql import SparkSession
from pyspark.sql.types import StructType, StructField, StringType, IntegerType, DoubleType, DateType
from pyspark.sql.functions import col, lit, when
from pyspark.ml.regression import GBTRegressionModel, RandomForestRegressionModel
from pyspark.ml.feature import VectorAssembler, StandardScaler, StringIndexer, OneHotEncoder
from pyspark.sql.functions import col as spark_col  # Alias pour éviter les conflits
from datetime import datetime, timedelta
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

from src.utils import ensure_directory_exists, get_tomorrow_date, format_date, classify_rainfall
from src.data_processing import SCHEMA, create_spark_session

def load_short_term_model(model_type="gbt", model_dir="models/short_term"):
    """
    Charge un modèle de prédiction à court terme.
    
    Args:
        model_type: Type de modèle ('gbt' ou 'rf')
        model_dir: Répertoire contenant les modèles
        
    Returns:
        Modèle chargé
    """
    # Déterminer le chemin du modèle
    if model_type.lower() == "gbt":
        model_path = os.path.join(model_dir, "gbt_model")
        model_class = GBTRegressionModel
        print("Chargement du modèle Gradient Boosted Trees...")
    elif model_type.lower() == "rf":
        model_path = os.path.join(model_dir, "rf_model")
        model_class = RandomForestRegressionModel
        print("Chargement du modèle Random Forest...")
    else:
        raise ValueError(f"Type de modèle non reconnu: {model_type}")
    
    # Vérifier si le modèle existe
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Modèle introuvable: {model_path}")
    
    # Charger le modèle
    model = model_class.load(model_path)
    print(f"Modèle chargé avec succès depuis {model_path}")
    
    return model

def get_weather_data_for_prediction(spark, region, date, data_path="data/donnees_pluviometriques_senegal.csv"):
    """
    Récupère les données météorologiques moyennes pour une région et une date.
    
    Dans un cas réel, ces données viendraient d'une API météo comme OpenWeatherMap
    ou de capteurs météorologiques en temps réel.
    
    Args:
        spark: Session Spark
        region: Région pour la prédiction
        date: Date pour la prédiction
        data_path: Chemin vers les données historiques
        
    Returns:
        Dictionnaire des variables météo et type de sol
    """
    print(f"Récupération des données météorologiques pour {region} le {format_date(date)}...")
    
    # Charger les données historiques
    try:
        data_df = spark.read \
            .option("header", "true") \
            .option("dateFormat", "yyyy-MM-dd") \
            .schema(SCHEMA) \
            .csv(data_path)
    except Exception as e:
        print(f"Erreur lors du chargement des données: {e}")
        sys.exit(1)
    
    # Filtrer pour la région spécifiée et le mois correspondant
    filtered_data = data_df.filter(
        (spark_col("region") == region) & (spark_col("mois") == date.month)
    )
    
    # Si pas de données pour le mois spécifique, utiliser toutes les données de la région
    if filtered_data.count() == 0:
        print(f"Pas de données disponibles pour {region} au mois {date.month}")
        filtered_data = data_df.filter(spark_col("region") == region)
        
        # Si toujours pas de données, utiliser les moyennes globales
        if filtered_data.count() == 0:
            print(f"Pas de données disponibles pour {region}. Utilisation des moyennes globales.")
            filtered_data = data_df
            
    print(f"Nombre d'enregistrements pour la prédiction: {filtered_data.count()}")
    
    # Obtenir les moyennes des colonnes numériques
    numeric_cols = ["temperature_min", "temperature_max", "humidite", "vent", 
                   "pression", "latitude", "longitude", "altitude", "distance_cote"]
    
    avg_values = filtered_data.select(numeric_cols).groupBy().avg().first()
    print("Valeurs moyennes extraites des données historiques:")
    for col in numeric_cols:
        print(f"  {col}: {avg_values[f'avg({col})']}")
    
    # Obtenir le type de sol le plus fréquent
    type_sol_row = filtered_data.groupBy("type_sol").count().orderBy("count", ascending=False).first()
    type_sol_mode = type_sol_row["type_sol"] if type_sol_row else "Standard"
    print(f"Type de sol le plus fréquent: {type_sol_mode}")
    
    # Créer un dictionnaire des données météo
    weather_data = {
        "region": region,
        "date": date,
        "annee": date.year,
        "mois": date.month,
        "jour": date.day,
        "temperature_min": avg_values["avg(temperature_min)"],
        "temperature_max": avg_values["avg(temperature_max)"],
        "humidite": avg_values["avg(humidite)"],
        "vent": avg_values["avg(vent)"],
        "pression": avg_values["avg(pression)"],
        "latitude": avg_values["avg(latitude)"],
        "longitude": avg_values["avg(longitude)"],
        "altitude": avg_values["avg(altitude)"],
        "type_sol": type_sol_mode,
        "distance_cote": avg_values["avg(distance_cote)"]
    }
    
    print("Données météo récupérées avec succès.")
    return weather_data

def prepare_data_for_prediction(spark, weather_data):
    """
    Prépare les données pour la prédiction en appliquant les mêmes transformations
    que lors de l'entraînement.
    
    Args:
        spark: Session Spark
        weather_data: Dictionnaire des données météo
        
    Returns:
        DataFrame Spark prêt pour la prédiction
    """
    print("Préparation des données pour la prédiction...")
    
    # Vérifier qu'aucune valeur n'est None et convertir les types de données si nécessaire
    for key in weather_data:
        if weather_data[key] is None and key != "precipitations":
            # Remplacement des valeurs None
            if key in ["temperature_min", "temperature_max", "humidite", "pression", "latitude", "longitude"]:
                weather_data[key] = 0.0
            elif key in ["vent", "altitude", "distance_cote"]:
                weather_data[key] = 0
            elif key == "type_sol":
                weather_data[key] = "Standard"
            print(f"ATTENTION: Valeur manquante pour {key}, remplacée par une valeur par défaut")
        
        # Conversion des types pour les champs entiers
        if key in ["vent", "altitude", "distance_cote"] and weather_data[key] is not None:
            weather_data[key] = int(weather_data[key])
            print(f"Conversion de {key} en entier: {weather_data[key]}")
    
    # Créer un DataFrame Spark à partir des données météo
    data_row = [(
        weather_data["region"],
        weather_data["date"],
        weather_data["annee"],
        weather_data["mois"],
        weather_data["jour"],
        0.0,  # valeur fictive pour precipitations (ce qu'on cherche à prédire)
        weather_data["temperature_min"],
        weather_data["temperature_max"],
        weather_data["humidite"],
        weather_data["vent"],
        weather_data["pression"],
        weather_data["latitude"],
        weather_data["longitude"],
        weather_data["altitude"],
        weather_data["type_sol"],
        weather_data["distance_cote"]
    )]
    
    # Créer le DataFrame
    pred_df = spark.createDataFrame(data_row, SCHEMA)
    
    # Ajouter les features temporelles
    jour_annee = weather_data["date"].timetuple().tm_yday
    pred_df = pred_df.withColumn("jour_annee", lit(jour_annee))
    
    # Déterminer la saison (saison des pluies: juin à octobre)
    saison = "saison_pluies" if 6 <= weather_data["mois"] <= 10 else "saison_seche"
    pred_df = pred_df.withColumn("saison", lit(saison))
    
    # Ajouter le trimestre
    trimestre = ((weather_data["mois"] - 1) // 3) + 1
    pred_df = pred_df.withColumn("trimestre", lit(trimestre).cast(IntegerType()))
    
    print(f"Features temporelles ajoutées: jour_annee={jour_annee}, saison={saison}, trimestre={trimestre}")
    
    # Encodage des variables catégorielles
    categorical_cols = ["region", "type_sol", "saison"]
    indexed_cols = [f"{col}_idx" for col in categorical_cols]
    encoded_cols = [f"{col}_enc" for col in categorical_cols]
    
    # Création et application des indexeurs
    for col_name in categorical_cols:
        indexer = StringIndexer(inputCol=col_name, outputCol=f"{col_name}_idx", handleInvalid="keep")
        pred_df = indexer.fit(pred_df).transform(pred_df)
    
    # Application de l'encodage one-hot
    encoder = OneHotEncoder(inputCols=indexed_cols, outputCols=encoded_cols)
    pred_df = encoder.fit(pred_df).transform(pred_df)
    
    # Préparation des features pour le modèle
    numeric_features = ["temperature_min", "temperature_max", "humidite", 
                       "vent", "pression", "jour_annee", "latitude", 
                       "longitude", "altitude", "distance_cote"]
    
    all_features = numeric_features + encoded_cols
    
    # Assemblage des features
    assembler = VectorAssembler(inputCols=all_features, outputCol="features", handleInvalid="skip")
    pred_features = assembler.transform(pred_df)
    
    # Normalisation des features
    scaler = StandardScaler(inputCol="features", outputCol="scaled_features")
    scaler_model = scaler.fit(pred_features)
    pred_scaled = scaler_model.transform(pred_features)
    
    print("Données préparées pour la prédiction.")
    return pred_scaled

def predict_rainfall(model, data):
    """
    Effectue une prédiction de pluviométrie.
    
    Args:
        model: Modèle entraîné
        data: DataFrame préparé pour la prédiction
        
    Returns:
        Valeur de la prédiction
    """
    print("Calcul de la prédiction...")
    
    # Appliquer le modèle aux données
    prediction = model.transform(data)
    
    # Extraire la valeur de la prédiction
    rainfall = prediction.select("prediction").first()[0]
    
    return rainfall

def visualize_prediction(rainfall, region, date, output_dir="output/predictions"):
    """
    Crée une visualisation de la prédiction.
    
    Args:
        rainfall: Valeur de la pluviométrie prédite
        region: Région concernée
        date: Date de la prédiction
        output_dir: Répertoire de sortie
        
    Returns:
        Chemin du fichier de visualisation
    """
    print("Création de la visualisation de la prédiction...")
    
    # S'assurer que le répertoire existe
    ensure_directory_exists(output_dir)
    
    # Créer la figure
    fig, ax = plt.subplots(figsize=(10, 6))
    
    # Dessiner le graphique à barres
    bars = ax.bar(["Prédiction"], [rainfall], color='royalblue', width=0.5)
    
    # Ajouter une étiquette sur la barre
    for bar in bars:
        height = bar.get_height()
        ax.text(
            bar.get_x() + bar.get_width()/2., 
            height + 0.1,
            f'{height:.2f} mm', 
            ha='center', 
            va='bottom',
            fontweight='bold'
        )
    
    # Ajouter les labels et le titre
    ax.set_ylabel('Précipitations (mm)')
    ax.set_ylim(bottom=0)
    ax.grid(axis='y', alpha=0.3)
    
    # Classer la prédiction
    category = classify_rainfall(rainfall)
    title = f'Prédiction pluviométrique pour {region} le {format_date(date)}\n{category}'
    ax.set_title(title)
    
    # Couleur de fond selon la catégorie
    if category == "Temps sec":
        fig.patch.set_facecolor('#FFF9C4')  # Jaune pâle
    elif category == "Pluie légère":
        fig.patch.set_facecolor('#BBDEFB')  # Bleu très pâle
    elif category == "Pluie modérée":
        fig.patch.set_facecolor('#90CAF9')  # Bleu pâle
    else:
        fig.patch.set_facecolor('#64B5F6')  # Bleu
    
    # Sauvegarder la figure
    date_str = date.strftime("%Y%m%d")
    filename = f'prediction_{region}_{date_str}.png'
    filepath = os.path.join(output_dir, filename)
    
    fig.savefig(filepath)
    print(f"Visualisation sauvegardée: {filepath}")
    
    return filepath

def predict_for_tomorrow(region="Dakar", model_type="gbt"):
    """
    Prédit la pluviométrie pour demain.
    
    Args:
        region: Région pour laquelle faire la prédiction
        model_type: Type de modèle à utiliser ('gbt' ou 'rf')
        
    Returns:
        Dictionnaire avec les résultats de la prédiction
    """
    # Initialiser Spark
    spark = create_spark_session("Prédiction Pluviométrique Quotidienne")
    
    # Date de demain
    tomorrow = get_tomorrow_date()
    print(f"Prédiction de la pluviométrie pour {region} le {format_date(tomorrow)}")
    
    try:
        # Charger le modèle
        model = load_short_term_model(model_type)
        
        # Obtenir les données météo
        weather_data = get_weather_data_for_prediction(spark, region, tomorrow)
        
        # Préparer les données pour la prédiction
        prediction_data = prepare_data_for_prediction(spark, weather_data)
        
        # Effectuer la prédiction
        rainfall = predict_rainfall(model, prediction_data)
        
        # Visualiser la prédiction
        viz_path = visualize_prediction(rainfall, region, tomorrow)
        
        # Déterminer la catégorie de pluie
        category = classify_rainfall(rainfall)
        
        # Afficher les résultats
        print("\n" + "=" * 50)
        print(f"PRÉDICTION PLUVIOMÉTRIQUE POUR DEMAIN - {format_date(tomorrow)}")
        print("=" * 50)
        print(f"Région: {region}")
        print(f"Pluviométrie prévue: {rainfall:.2f} mm")
        print(f"Prévision: {category}")
        print("=" * 50)
        
        # Retourner les résultats
        result = {
            "region": region,
            "date": tomorrow,
            "rainfall": rainfall,
            "category": category,
            "visualization": viz_path
        }
        
        return result
        
    except Exception as e:
        print(f"Erreur lors de la prédiction: {e}")
        raise
    finally:
        # Arrêter la session Spark
        spark.stop()

if __name__ == "__main__":
    # Si exécuté directement, prédire pour demain à Dakar
    try:
        result = predict_for_tomorrow()
        print("\nPrédiction terminée avec succès!")
    except Exception as e:
        print(f"Échec de la prédiction: {e}")
        sys.exit(1)
