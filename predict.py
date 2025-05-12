#!/usr/bin/env python
"""
Interface de prédiction pluviométrique pour le Sénégal
======================================================
Ce script fournit une interface en ligne de commande pour prédire la pluviométrie
au Sénégal, pour une région spécifique ou toutes les régions.
"""

import argparse
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime, timedelta
import os

def get_date(days_offset=1, date_str=None):
    """Retourne une date basée sur un offset ou une chaîne de caractères"""
    if date_str:
        try:
            return datetime.strptime(date_str, "%Y-%m-%d")
        except ValueError:
            print(f"Format de date invalide: {date_str}. Utilisation de l'offset.")
    
    return datetime.now() + timedelta(days=days_offset)

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

def predict_for_region_and_date(df, region, date):
    """
    Prédit la pluviométrie pour une région et une date spécifiques
    en utilisant les moyennes historiques du même mois.
    """
    # Filtrer les données pour la région spécifiée
    region_data = df[df['region'] == region]
    
    if len(region_data) == 0:
        print(f"Aucune donnée trouvée pour la région {region}. Utilisation de toutes les données.")
        region_data = df
    
    # Filtrer par mois
    month_data = region_data[region_data['date'].dt.month == date.month]
    
    if len(month_data) == 0:
        print(f"Aucune donnée trouvée pour le mois {date.month} dans la région {region}. Utilisation de toutes les données de la région.")
        month_data = region_data
    
    # Calculer la moyenne des précipitations pour ce mois
    avg_precipitation = month_data['precipitations'].mean()
    
    # Calculer l'écart-type pour donner une idée de la variabilité
    std_precipitation = month_data['precipitations'].std()
    
    # Nombre d'années de données disponibles
    years = month_data['date'].dt.year.nunique()
    
    # Calculer la tendance annuelle pour ce mois si suffisamment de données
    trend = 0
    if years >= 3:
        month_data = month_data.copy()  # Éviter SettingWithCopyWarning
        month_data['year'] = month_data['date'].dt.year
        annual_means = month_data.groupby('year')['precipitations'].mean()
        if len(annual_means) >= 2:
            years_array = np.array(annual_means.index)
            precip_array = np.array(annual_means.values)
            
            # Régression linéaire simple pour estimer la tendance
            if len(years_array) > 1:  # S'assurer qu'il y a suffisamment de points
                slope, _ = np.polyfit(years_array, precip_array, 1)
                trend = slope
    
    # Ajuster la prédiction avec la tendance (simplifié)
    adjusted_prediction = max(0, avg_precipitation + trend * (date.year - month_data['date'].dt.year.mean()))
    
    # Coordonnées géographiques moyennes pour la région (pour la carte)
    latitude = region_data['latitude'].mean() if 'latitude' in region_data.columns else None
    longitude = region_data['longitude'].mean() if 'longitude' in region_data.columns else None
    
    # Déterminer la catégorie de précipitation
    if adjusted_prediction < 0.1:
        category = "Temps sec"
    elif adjusted_prediction < 5:
        category = "Pluie légère"
    elif adjusted_prediction < 20:
        category = "Pluie modérée"
    else:
        category = "Forte pluie"
    
    # Créer un rapport
    report = {
        "region": region,
        "date": date,
        "precipitation": adjusted_prediction,
        "std_deviation": std_precipitation,
        "category": category,
        "data_points": len(month_data),
        "years_of_data": years,
        "annual_trend": trend,
        "latitude": latitude,
        "longitude": longitude
    }
    
    return report

def visualize_prediction(prediction, output_dir="output/predictions"):
    """Crée une visualisation de la prédiction pour une région"""
    region = prediction["region"]
    date = prediction["date"]
    rainfall = prediction["precipitation"]
    category = prediction["category"]
    
    # Créer une figure
    fig, ax = plt.subplots(figsize=(10, 6))
    
    # Dessiner le graphique à barres
    bars = ax.bar(["Prédiction"], [rainfall], color='royalblue', width=0.5)
    
    # Ajouter l'écart-type comme barres d'erreur
    if prediction["std_deviation"] > 0:
        ax.errorbar(
            ["Prédiction"], 
            [rainfall], 
            yerr=prediction["std_deviation"],
            fmt='none', 
            color='black', 
            capsize=5
        )
    
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
    
    # Définir les propriétés du graphique
    ax.set_ylabel('Précipitations (mm)')
    ax.set_ylim(bottom=0, top=max(rainfall * 1.5, rainfall + prediction["std_deviation"] * 2 + 5, 1.0))
    ax.grid(axis='y', alpha=0.3)
    
    # Titre du graphique
    title = f'Prédiction pluviométrique pour {region}\nle {date.strftime("%d/%m/%Y")}'
    subtitle = f'Catégorie: {category}'
    ax.set_title(f"{title}\n{subtitle}")
    
    # Information supplémentaire
    info_text = (
        f"Basé sur {prediction['data_points']} observations historiques\n"
        f"sur {prediction['years_of_data']} années\n"
        f"Écart-type: {prediction['std_deviation']:.2f} mm"
    )
    plt.figtext(0.02, 0.02, info_text, fontsize=8)
    
    # Changer la couleur du fond selon la catégorie
    if category == "Temps sec":
        fig.patch.set_facecolor('#FFF9C4')  # Jaune pâle
    elif category == "Pluie légère":
        fig.patch.set_facecolor('#BBDEFB')  # Bleu très pâle
    elif category == "Pluie modérée":
        fig.patch.set_facecolor('#90CAF9')  # Bleu pâle
    else:
        fig.patch.set_facecolor('#64B5F6')  # Bleu
    
    # Créer le répertoire de sortie s'il n'existe pas
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    # Sauvegarder la figure
    viz_filename = f'prediction_{region}_{date.strftime("%Y%m%d")}.png'
    viz_path = os.path.join(output_dir, viz_filename)
    plt.savefig(viz_path)
    plt.close()
    
    return viz_path

def create_precipitation_map(predictions, date, output_dir="output/predictions"):
    """Crée une carte des prédictions pluviométriques pour toutes les régions"""
    print("Création de la carte des précipitations...")
    
    # Extraire les données pour la carte, en filtrant les prédictions sans coordonnées
    valid_predictions = [p for p in predictions if p["latitude"] is not None and p["longitude"] is not None]
    
    if not valid_predictions:
        print("Aucune coordonnée valide trouvée pour créer la carte. Assurez-vous que votre dataset contient les colonnes 'latitude' et 'longitude'.")
        return None
    
    regions = [p["region"] for p in valid_predictions]
    latitudes = [p["latitude"] for p in valid_predictions]
    longitudes = [p["longitude"] for p in valid_predictions]
    precipitations = [p["precipitation"] for p in valid_predictions]
    
    # Créer la figure
    fig, ax = plt.subplots(figsize=(12, 10))
    
    # Dessiner la carte (simplifié, sans fond de carte)
    scatter = ax.scatter(
        longitudes, 
        latitudes, 
        c=precipitations,
        s=300,  # Taille des cercles
        cmap='Blues',
        alpha=0.8,
        edgecolors='black'
    )
    
    # Ajouter les noms des régions
    for i, region in enumerate(regions):
        ax.annotate(
            region, 
            (longitudes[i], latitudes[i]),
            xytext=(5, 5),
            textcoords='offset points',
            fontsize=9,
            fontweight='bold'
        )
    
    # Ajouter une barre de couleur
    cbar = plt.colorbar(scatter)
    cbar.set_label('Précipitations prévues (mm)')
    
    # Ajouter le titre
    plt.title(f'Prédictions pluviométriques pour le Sénégal - {date.strftime("%d/%m/%Y")}')
    
    # Ajouter des étiquettes aux axes
    plt.xlabel('Longitude')
    plt.ylabel('Latitude')
    
    # Ajuster les limites pour centrer sur le Sénégal
    plt.xlim(min(longitudes) - 0.5, max(longitudes) + 0.5)
    plt.ylim(min(latitudes) - 0.5, max(latitudes) + 0.5)
    
    # Créer le répertoire de sortie s'il n'existe pas
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    # Sauvegarder la carte
    map_filename = f'carte_precipitations_{date.strftime("%Y%m%d")}.png'
    map_path = os.path.join(output_dir, map_filename)
    plt.savefig(map_path)
    plt.close()
    
    return map_path

def main():
    parser = argparse.ArgumentParser(description='Prédiction pluviométrique pour le Sénégal')
    
    # Arguments en ligne de commande
    parser.add_argument('--region', type=str, default=None, 
                        help='Région spécifique à prédire (par défaut: toutes)')
    parser.add_argument('--date', type=str, default=None, 
                        help='Date de prédiction au format YYYY-MM-DD (par défaut: demain)')
    parser.add_argument('--jours', type=int, default=1, 
                        help='Nombre de jours dans le futur pour la prédiction (par défaut: 1)')
    parser.add_argument('--donnees', type=str, default='data/donnees_pluviometriques_senegal.csv',
                        help='Chemin vers le fichier de données (par défaut: data/donnees_pluviometriques_senegal.csv)')
    parser.add_argument('--output', type=str, default='output/predictions',
                        help='Répertoire de sortie pour les visualisations (par défaut: output/predictions)')
    
    args = parser.parse_args()
    
    try:
        # Charger les données
        df = load_data(args.donnees)
        
        # Déterminer la date de prédiction
        target_date = get_date(args.jours, args.date)
        print(f"Date de prédiction: {target_date.strftime('%d/%m/%Y')}")
        
        # Obtenir toutes les régions disponibles
        all_regions = sorted(df['region'].unique())
        
        # Déterminer les régions à traiter
        if args.region:
            if args.region not in all_regions:
                print(f"AVERTISSEMENT: La région '{args.region}' n'est pas dans le jeu de données.")
                print(f"Régions disponibles: {', '.join(all_regions)}")
                return
            regions_to_process = [args.region]
        else:
            regions_to_process = all_regions
            print(f"Traitement de toutes les régions: {', '.join(regions_to_process)}")
        
        # Faire des prédictions pour les régions sélectionnées
        all_predictions = []
        region_results = {}
        
        for region in regions_to_process:
            print(f"Traitement de la région: {region}")
            prediction = predict_for_region_and_date(df, region, target_date)
            all_predictions.append(prediction)
            
            # Créer une visualisation pour chaque région
            viz_path = visualize_prediction(prediction, args.output)
            
            # Stocker les résultats pour affichage
            region_results[region] = {
                "precipitation": prediction["precipitation"],
                "category": prediction["category"],
                "visualization": viz_path
            }
        
        # Créer une carte si on traite plusieurs régions
        map_path = None
        if len(regions_to_process) > 1:
            map_path = create_precipitation_map(all_predictions, target_date, args.output)
        
        # Afficher le tableau récapitulatif
        print("\n" + "=" * 80)
        print(f"PRÉDICTIONS PLUVIOMÉTRIQUES POUR LE {target_date.strftime('%d/%m/%Y')}")
        print("=" * 80)
        print(f"{'Région':<15} {'Précipitations (mm)':<20} {'Catégorie':<15}")
        print("-" * 80)
        
        for region in sorted(region_results.keys()):
            result = region_results[region]
            print(f"{region:<15} {result['precipitation']:<20.2f} {result['category']:<15}")
        
        print("=" * 80)
        
        if map_path:
            print(f"Carte des précipitations: {map_path}")
        
        for region, result in sorted(region_results.items()):
            print(f"Visualisation pour {region}: {result['visualization']}")
        
    except Exception as e:
        print(f"Erreur lors des prédictions: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()
