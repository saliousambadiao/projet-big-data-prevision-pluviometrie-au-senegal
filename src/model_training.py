"""
Entraînement des modèles de prédiction pluviométrique au Sénégal
================================================================
Ce module gère l'entraînement et l'évaluation des modèles de prédiction.
"""

import os
import findspark
findspark.init()

from pyspark.ml.regression import RandomForestRegressor, GBTRegressor
from pyspark.ml.evaluation import RegressionEvaluator
from pyspark.sql.functions import col
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

from src.utils import ensure_directory_exists, save_visualization

def train_random_forest(train_data, features_col="scaled_features", label_col="precipitations"):
    """
    Entraîne un modèle Random Forest pour la prédiction des précipitations.
    
    Args:
        train_data: DataFrame Spark contenant les données d'entraînement
        features_col: Nom de la colonne contenant les features
        label_col: Nom de la colonne contenant les labels (précipitations)
        
    Returns:
        Modèle Random Forest entraîné
    """
    print("Entraînement du modèle Random Forest...")
    
    # Configuration du modèle
    rf = RandomForestRegressor(
        labelCol=label_col, 
        featuresCol=features_col,
        numTrees=100,
        maxDepth=10,
        seed=42
    )
    
    # Entraînement du modèle
    rf_model = rf.fit(train_data)
    
    print("Modèle Random Forest entraîné avec succès")
    return rf_model

def train_gradient_boosting(train_data, features_col="scaled_features", label_col="precipitations"):
    """
    Entraîne un modèle Gradient Boosted Trees pour la prédiction des précipitations.
    
    Args:
        train_data: DataFrame Spark contenant les données d'entraînement
        features_col: Nom de la colonne contenant les features
        label_col: Nom de la colonne contenant les labels (précipitations)
        
    Returns:
        Modèle GBT entraîné
    """
    print("Entraînement du modèle Gradient Boosted Trees...")
    
    # Configuration du modèle
    gbt = GBTRegressor(
        labelCol=label_col, 
        featuresCol=features_col,
        maxIter=100,
        maxDepth=8,
        stepSize=0.1,
        seed=42
    )
    
    # Entraînement du modèle
    gbt_model = gbt.fit(train_data)
    
    print("Modèle Gradient Boosted Trees entraîné avec succès")
    return gbt_model

def evaluate_model(model, test_data, model_name, features_col="scaled_features", label_col="precipitations"):
    """
    Évalue un modèle sur les données de test.
    
    Args:
        model: Modèle ML entraîné
        test_data: DataFrame Spark contenant les données de test
        model_name: Nom du modèle pour l'affichage
        features_col: Nom de la colonne contenant les features
        label_col: Nom de la colonne contenant les labels (précipitations)
        
    Returns:
        Dictionary contenant les métriques d'évaluation et les prédictions
    """
    print(f"Évaluation du modèle {model_name}...")
    
    # Faire des prédictions sur l'ensemble de test
    predictions = model.transform(test_data)
    
    # Initialiser l'évaluateur
    evaluator = RegressionEvaluator(labelCol=label_col, predictionCol="prediction")
    
    # Calculer les métriques
    rmse = evaluator.setMetricName("rmse").evaluate(predictions)
    mae = evaluator.setMetricName("mae").evaluate(predictions)
    r2 = evaluator.setMetricName("r2").evaluate(predictions)
    
    # Afficher les résultats
    print(f"Résultats pour {model_name}:")
    print(f"  RMSE: {rmse:.4f}")
    print(f"  MAE: {mae:.4f}")
    print(f"  R²: {r2:.4f}")
    
    return {
        "model_name": model_name,
        "rmse": rmse,
        "mae": mae,
        "r2": r2,
        "predictions": predictions
    }

def save_model(model, model_name, output_dir="models/short_term"):
    """
    Sauvegarde un modèle entraîné.
    
    Args:
        model: Modèle ML à sauvegarder
        model_name: Nom du modèle
        output_dir: Répertoire de sortie
    """
    # S'assurer que le répertoire existe
    ensure_directory_exists(output_dir)
    
    # Chemin complet du modèle
    model_path = os.path.join(output_dir, model_name)
    
    try:
        # Sauvegarder le modèle
        model.write().overwrite().save(model_path)
        print(f"Modèle sauvegardé avec succès: {model_path}")
    except Exception as e:
        print(f"Erreur lors de la sauvegarde du modèle: {e}")

def visualize_predictions(rf_results, gbt_results, output_dir="output/figures"):
    """
    Crée des visualisations comparant les prédictions des différents modèles.
    
    Args:
        rf_results: Résultats du modèle Random Forest
        gbt_results: Résultats du modèle Gradient Boosted Trees
        output_dir: Répertoire de sortie pour les visualisations
    """
    print("Création des visualisations de comparaison des modèles...")
    
    # Convertir les prédictions en pandas DataFrame pour la visualisation
    rf_pred_pd = rf_results["predictions"].select(
        "date", "region", "precipitations", "prediction"
    ).withColumnRenamed("prediction", "predicted_rf").toPandas()
    
    gbt_pred_pd = gbt_results["predictions"].select(
        "date", "region", "precipitations", "prediction"
    ).withColumnRenamed("prediction", "predicted_gbt").toPandas()
    
    # Fusionner les deux DataFrames
    merged_pred = pd.merge(
        rf_pred_pd, 
        gbt_pred_pd[["date", "region", "predicted_gbt"]], 
        on=["date", "region"], 
        how="inner"
    )
    
    # Échantillonnage pour limiter le nombre de points (si nécessaire)
    if len(merged_pred) > 1000:
        merged_pred = merged_pred.sample(1000, random_state=42)
    
    # Créer une visualisation pour comparer les prédictions
    fig, ax = plt.subplots(figsize=(12, 8))
    
    # Scatter plot
    ax.scatter(
        merged_pred["precipitations"], 
        merged_pred["predicted_rf"],
        alpha=0.5, 
        label='Random Forest', 
        color='blue'
    )
    
    ax.scatter(
        merged_pred["precipitations"], 
        merged_pred["predicted_gbt"],
        alpha=0.5, 
        label='Gradient Boosted Trees', 
        color='red'
    )
    
    # Ligne diagonale (prédiction parfaite)
    max_val = max(
        merged_pred["precipitations"].max(),
        max(merged_pred["predicted_rf"].max(), merged_pred["predicted_gbt"].max())
    )
    
    ax.plot([0, max_val], [0, max_val], 'k--', label='Prédiction parfaite')
    
    ax.set_xlabel('Précipitations réelles (mm)')
    ax.set_ylabel('Précipitations prédites (mm)')
    ax.set_title('Comparaison des modèles à court terme')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # Enregistrer la figure
    save_visualization(fig, 'comparaison_modeles_court_terme.png', output_dir=output_dir)
    
    # Créer un graphique à barres pour comparer les métriques
    models = ['Random Forest', 'Gradient Boosted Trees']
    rmse_values = [rf_results["rmse"], gbt_results["rmse"]]
    mae_values = [rf_results["mae"], gbt_results["mae"]]
    r2_values = [rf_results["r2"], gbt_results["r2"]]
    
    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(15, 5))
    
    # RMSE (plus bas = meilleur)
    ax1.bar(models, rmse_values, color=['blue', 'red'])
    ax1.set_title('RMSE (mm)')
    ax1.set_ylabel('Erreur (mm)')
    ax1.grid(axis='y', alpha=0.3)
    
    # MAE (plus bas = meilleur)
    ax2.bar(models, mae_values, color=['blue', 'red'])
    ax2.set_title('MAE (mm)')
    ax2.grid(axis='y', alpha=0.3)
    
    # R² (plus haut = meilleur)
    ax3.bar(models, r2_values, color=['blue', 'red'])
    ax3.set_title('Coefficient R²')
    ax3.grid(axis='y', alpha=0.3)
    
    # Enregistrer la figure
    save_visualization(fig, 'metriques_modeles_court_terme.png', output_dir=output_dir)

def train_prophet_model(df, region, output_dir="models/long_term"):
    """
    Entraîne un modèle Prophet pour la prédiction à long terme.
    Note: Cette fonction doit être executée dans un environnement où Prophet est installé.
    
    Args:
        df: DataFrame PySpark contenant les données temporelles
        region: Région pour laquelle entraîner le modèle
        output_dir: Répertoire de sortie pour le modèle
        
    Returns:
        Modèle Prophet entraîné (ou None si échec)
    """
    try:
        from prophet import Prophet
        import pickle
        
        print(f"Préparation des données pour Prophet (région: {region})...")
        
        # Conversion vers Pandas pour utiliser Prophet
        # Agrégation par mois et par région
        monthly_data = df.filter(col("region") == region) \
            .groupBy("region", "annee", "mois") \
            .agg({"precipitations": "avg"}) \
            .withColumnRenamed("avg(precipitations)", "precipitations") \
            .orderBy("annee", "mois")
        
        # Conversion en pandas DataFrame
        monthly_data_pd = monthly_data.toPandas()
        
        # Créer des colonnes au format attendu par Prophet
        monthly_data_pd['ds'] = pd.to_datetime(
            monthly_data_pd['annee'].astype(str) + '-' + 
            monthly_data_pd['mois'].astype(str) + '-01'
        )
        monthly_data_pd['y'] = monthly_data_pd['precipitations']
        
        # Division entraînement/test
        train_prophet = monthly_data_pd[monthly_data_pd['ds'].dt.year < 2021]
        
        print(f"Entraînement du modèle Prophet pour {region}...")
        model = Prophet(yearly_seasonality=True, weekly_seasonality=False, daily_seasonality=False)
        model.fit(train_prophet)
        
        # Sauvegarder le modèle avec pickle
        ensure_directory_exists(output_dir)
        model_path = os.path.join(output_dir, f"prophet_{region}.pkl")
        
        with open(model_path, "wb") as f:
            pickle.dump(model, f)
        
        print(f"Modèle Prophet pour {region} sauvegardé: {model_path}")
        
        # Visualiser les prédictions futures
        last_date = pd.to_datetime(monthly_data_pd['ds'].max())
        future_dates = pd.date_range(start=last_date, periods=120, freq='M')  # 10 ans
        future = pd.DataFrame(future_dates, columns=['ds'])
        
        forecast = model.predict(future)
        
        # Créer une visualisation
        fig = plt.figure(figsize=(12, 6))
        
        # Données historiques
        plt.plot(monthly_data_pd['ds'], monthly_data_pd['y'], 'k.', label='Observations')
        
        # Prédictions
        plt.plot(forecast['ds'], forecast['yhat'], 'b-', label='Prédiction')
        plt.fill_between(
            forecast['ds'], 
            forecast['yhat_lower'], 
            forecast['yhat_upper'], 
            color='blue', 
            alpha=0.2, 
            label='Intervalle de confiance 95%'
        )
        
        plt.xlabel('Date')
        plt.ylabel('Précipitations moyennes (mm)')
        plt.title(f'Prédiction à long terme des précipitations pour {region}')
        plt.legend()
        
        # Enregistrer la visualisation
        save_visualization(fig, f'prediction_long_terme_{region}.png', output_dir="output/figures")
        
        return model
        
    except ImportError:
        print("Prophet n'est pas installé. Impossible d'entraîner le modèle à long terme.")
        print("Pour installer Prophet: pip install prophet")
        return None
    except Exception as e:
        print(f"Erreur lors de l'entraînement du modèle Prophet: {e}")
        return None
