#!/usr/bin/env python
"""
Prédiction pluviométrique à long terme pour le Sénégal
======================================================
Ce script utilise Prophet pour prédire les tendances pluviométriques
sur une période de plusieurs années (par défaut 10 ans).
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from prophet import Prophet
from datetime import datetime
import os
import argparse
import warnings
warnings.filterwarnings('ignore')  # Ignorer les avertissements de Prophet

def load_data(file_path="data/donnees_pluviometriques_senegal.csv"):
    """Charge les données historiques en pandas DataFrame"""
    print(f"Chargement des données depuis {file_path}...")
    
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"Le fichier {file_path} n'existe pas. Veuillez vérifier le chemin.")
    
    df = pd.read_csv(file_path)
    
    # Convertir la colonne date en datetime
    if 'date' in df.columns:
        df['date'] = pd.to_datetime(df['date'])
    else:
        raise ValueError("Le fichier ne contient pas de colonne 'date'.")
    
    if 'region' not in df.columns:
        raise ValueError("Le fichier ne contient pas de colonne 'region'.")
    
    if 'precipitations' not in df.columns:
        raise ValueError("Le fichier ne contient pas de colonne 'precipitations'.")
    
    return df

def prepare_data_for_prophet(df, region=None):
    """
    Prépare les données pour le modèle Prophet.
    Si une région est spécifiée, filtre les données pour cette région uniquement.
    """
    # Filtrer par région si spécifiée
    if region:
        print(f"Filtrage des données pour la région: {region}")
        region_data = df[df['region'] == region].copy()
        if len(region_data) == 0:
            print(f"Aucune donnée trouvée pour la région {region}. Utilisation de toutes les données.")
            region_data = df.copy()
    else:
        print("Utilisation des données agrégées pour toutes les régions")
        region_data = df.copy()
    
    # Aggréger les données par mois pour chaque région
    region_data['year_month'] = region_data['date'].dt.to_period('M').dt.to_timestamp()
    monthly_data = region_data.groupby(['region', 'year_month'])['precipitations'].mean().reset_index()
    
    # Préparer le DataFrame au format requis par Prophet
    prophet_dfs = {}
    
    for reg in monthly_data['region'].unique():
        # Filtrer pour la région
        reg_data = monthly_data[monthly_data['region'] == reg]
        
        # Créer le DataFrame au format Prophet (ds, y)
        prophet_df = pd.DataFrame({
            'ds': reg_data['year_month'],
            'y': reg_data['precipitations']
        })
        
        prophet_dfs[reg] = prophet_df
    
    return prophet_dfs

def train_and_predict(prophet_df, periods=120, yearly_seasonality=True, monthly_seasonality=False):
    """
    Entraîne un modèle Prophet et génère des prédictions.
    
    Args:
        prophet_df: DataFrame au format Prophet (ds, y)
        periods: Nombre de mois à prédire
        yearly_seasonality: Activer la saisonnalité annuelle
        monthly_seasonality: Activer la saisonnalité mensuelle
        
    Returns:
        Tuple (modèle, prédictions)
    """
    print(f"Entraînement du modèle Prophet avec {len(prophet_df)} points de données...")
    model = Prophet(
        yearly_seasonality=yearly_seasonality,
        weekly_seasonality=False,
        daily_seasonality=False
    )
    
    if monthly_seasonality:
        model.add_seasonality(name='monthly', period=30.5, fourier_order=5)
    
    model.fit(prophet_df)
    
    # Générer des dates futures
    future = model.make_future_dataframe(periods=periods, freq='M')
    
    # Faire les prédictions
    forecast = model.predict(future)
    
    return model, forecast

def visualize_forecast(forecast, region, historical_df=None, output_dir="output/predictions", prefix=""):
    """
    Crée une visualisation des prédictions à long terme pour une région.
    """
    fig, ax = plt.subplots(figsize=(15, 8))
    
    # Formatter les dates sur l'axe x
    years = mdates.YearLocator()
    years_fmt = mdates.DateFormatter('%Y')
    ax.xaxis.set_major_locator(years)
    ax.xaxis.set_major_formatter(years_fmt)
    
    # Ajouter les prédictions
    ax.plot(forecast['ds'], forecast['yhat'], label='Prédiction', color='blue', linewidth=2)
    
    # Ajouter l'intervalle de confiance
    ax.fill_between(forecast['ds'], forecast['yhat_lower'], forecast['yhat_upper'], 
                    color='blue', alpha=0.2, label='Intervalle de confiance (95%)')
    
    # Ajouter les données historiques si disponibles
    if historical_df is not None:
        ax.scatter(historical_df['ds'], historical_df['y'], color='black', s=20, 
                   alpha=0.7, label='Observations historiques')
    
    # Style du graphique
    ax.grid(True, linestyle='--', alpha=0.7)
    ax.set_xlabel('Année')
    ax.set_ylabel('Précipitations (mm)')
    
    # Titre et légende
    future_years = forecast['ds'].dt.year.max() - forecast['ds'].dt.year.min()
    plt.title(f'Prédictions pluviométriques à long terme ({future_years} ans) pour {region}')
    plt.legend(loc='best')
    
    # Créer le répertoire de sortie s'il n'existe pas
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    # Sauvegarder la figure
    output_filename = f'{prefix}prediction_long_terme_{region.replace(" ", "_")}.png'
    output_path = os.path.join(output_dir, output_filename)
    plt.savefig(output_path)
    plt.close()
    
    return output_path

def analyze_trend(forecast, region):
    """
    Analyse la tendance générale des prédictions et génère un rapport.
    """
    # Calculer la tendance moyenne par année
    forecast['year'] = forecast['ds'].dt.year
    yearly_avg = forecast.groupby('year')['yhat'].mean()
    
    # Calculer la pente (tendance)
    years = np.array(yearly_avg.index)
    values = np.array(yearly_avg.values)
    
    if len(years) > 1:
        slope, intercept = np.polyfit(years, values, 1)
        trend_direction = "à la hausse" if slope > 0 else "à la baisse"
        trend_strength = abs(slope)
        
        # Calculer le changement en pourcentage sur la période
        start_value = yearly_avg.iloc[0]
        end_value = yearly_avg.iloc[-1]
        if start_value > 0:
            pct_change = ((end_value - start_value) / start_value) * 100
        else:
            pct_change = float('inf') if end_value > 0 else 0
        
        # Périodes extrêmes
        wettest_year = yearly_avg.idxmax()
        driest_year = yearly_avg.idxmin()
        
        # Saisonnalité
        forecast['month'] = forecast['ds'].dt.month
        monthly_avg = forecast.groupby('month')['yhat'].mean()
        wettest_month = monthly_avg.idxmax()
        driest_month = monthly_avg.idxmin()
        
        # Générer le rapport
        report = {
            "region": region,
            "trend_direction": trend_direction,
            "trend_strength": trend_strength,
            "pct_change": pct_change,
            "wettest_year": wettest_year,
            "driest_year": driest_year,
            "wettest_month": wettest_month,
            "driest_month": driest_month,
            "start_year": years[0],
            "end_year": years[-1],
            "start_value": start_value,
            "end_value": end_value
        }
        
        return report
    
    return None

def visualize_monthly_pattern(forecast, region, output_dir="output/predictions", prefix=""):
    """
    Visualise le pattern mensuel des précipitations pour identifier la saisonnalité.
    """
    # Ajouter des colonnes pour le mois
    forecast['month'] = forecast['ds'].dt.month
    
    # Calculer la moyenne par mois
    monthly_avg = forecast.groupby('month')['yhat'].mean().reset_index()
    
    # Créer la figure
    fig, ax = plt.subplots(figsize=(12, 6))
    
    # Noms des mois
    month_names = ['Jan', 'Fév', 'Mar', 'Avr', 'Mai', 'Juin', 'Juil', 'Août', 'Sep', 'Oct', 'Nov', 'Déc']
    
    # Dessiner le graphique à barres
    bars = ax.bar(monthly_avg['month'], monthly_avg['yhat'], color='skyblue')
    
    # Ajouter les étiquettes sur les barres
    for bar in bars:
        height = bar.get_height()
        ax.text(
            bar.get_x() + bar.get_width()/2., 
            height + 0.1,
            f'{height:.1f}', 
            ha='center', 
            va='bottom'
        )
    
    # Configurer les axes
    ax.set_xticks(range(1, 13))
    ax.set_xticklabels(month_names)
    ax.set_ylabel('Précipitations moyennes (mm)')
    ax.set_title(f'Pattern mensuel des précipitations pour {region}')
    ax.grid(axis='y', alpha=0.3)
    
    # Créer le répertoire de sortie s'il n'existe pas
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    # Sauvegarder la figure
    output_filename = f'{prefix}pattern_mensuel_{region.replace(" ", "_")}.png'
    output_path = os.path.join(output_dir, output_filename)
    plt.savefig(output_path)
    plt.close()
    
    return output_path

def run_long_term_prediction(df, region=None, years=10, output_dir="output/predictions", prefix=""):
    """
    Exécute le processus complet de prédiction à long terme pour une région ou toutes les régions.
    """
    # Préparer les données pour Prophet
    prophet_dfs = prepare_data_for_prophet(df, region)
    
    results = {}
    
    # Si une région spécifique est demandée et existe
    if region and region in prophet_dfs:
        regions_to_process = [region]
    else:
        regions_to_process = list(prophet_dfs.keys())
    
    print(f"Traitement des prédictions à long terme pour {len(regions_to_process)} régions...")
    
    for reg in regions_to_process:
        print(f"\nTraitement de la région: {reg}")
        prophet_df = prophet_dfs[reg]
        
        # Entraîner le modèle et faire les prédictions
        model, forecast = train_and_predict(prophet_df, periods=years*12)
        
        # Visualiser les prédictions
        viz_path = visualize_forecast(forecast, reg, prophet_df, output_dir, prefix)
        
        # Visualiser le pattern mensuel
        monthly_pattern_path = visualize_monthly_pattern(forecast, reg, output_dir, prefix)
        
        # Analyser la tendance
        trend_report = analyze_trend(forecast, reg)
        
        # Stocker les résultats
        results[reg] = {
            "forecast": forecast,
            "visualization": viz_path,
            "monthly_pattern": monthly_pattern_path,
            "trend_report": trend_report
        }
    
    return results

def main():
    # Configurer les arguments en ligne de commande
    parser = argparse.ArgumentParser(description='Prédictions pluviométriques à long terme pour le Sénégal')
    
    parser.add_argument('--region', type=str, default=None, 
                        help='Région spécifique à prédire (par défaut: toutes)')
    parser.add_argument('--annees', type=int, default=10, 
                        help='Nombre d\'années à prédire dans le futur (par défaut: 10)')
    parser.add_argument('--donnees', type=str, default='data/donnees_pluviometriques_senegal.csv',
                        help='Chemin vers le fichier de données (par défaut: data/donnees_pluviometriques_senegal.csv)')
    parser.add_argument('--output', type=str, default='output/predictions',
                        help='Répertoire de sortie pour les visualisations (par défaut: output/predictions)')
    parser.add_argument('--prefix', type=str, default='',
                        help='Préfixe pour les noms de fichiers générés (optionnel)')
    
    args = parser.parse_args()
    
    try:
        # Charger les données
        df = load_data(args.donnees)
        
        if args.region:
            if args.region not in df['region'].unique():
                print(f"AVERTISSEMENT: La région '{args.region}' n'est pas dans le jeu de données.")
                print(f"Régions disponibles: {', '.join(sorted(df['region'].unique()))}")
                return
        
        # Exécuter les prédictions à long terme
        results = run_long_term_prediction(
            df, 
            region=args.region, 
            years=args.annees, 
            output_dir=args.output,
            prefix=args.prefix
        )
        
        # Afficher un rapport récapitulatif
        print("\n" + "=" * 80)
        print(f"PRÉDICTIONS PLUVIOMÉTRIQUES À LONG TERME ({args.annees} ANS)")
        print("=" * 80)
        
        for region, result in sorted(results.items()):
            report = result["trend_report"]
            if report:
                print(f"\nRégion: {region}")
                print(f"  Tendance: {report['trend_direction']} ({report['trend_strength']:.4f} mm/an)")
                print(f"  Changement estimé {report['start_year']}-{report['end_year']}: {report['pct_change']:.1f}%")
                print(f"  Année la plus humide prévue: {report['wettest_year']}")
                print(f"  Année la plus sèche prévue: {report['driest_year']}")
                print(f"  Mois le plus humide: {report['wettest_month']} (correspondant généralement à {['Jan', 'Fév', 'Mar', 'Avr', 'Mai', 'Juin', 'Juil', 'Août', 'Sep', 'Oct', 'Nov', 'Déc'][report['wettest_month']-1]})")
                print(f"  Visualisations:")
                print(f"    - Prédiction à long terme: {result['visualization']}")
                print(f"    - Pattern mensuel: {result['monthly_pattern']}")
        
        print("\n" + "=" * 80)
        
    except Exception as e:
        print(f"Erreur lors des prédictions à long terme: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()
