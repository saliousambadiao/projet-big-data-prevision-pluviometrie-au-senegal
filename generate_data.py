# Importation des bibliothèques
import pandas as pd
import numpy as np
import random
from datetime import datetime, timedelta
import matplotlib.pyplot as plt
import seaborn as sns

# Définition des régions du Sénégal
regions = [
    "Dakar",
    "Thiès",
    "Diourbel",
    "Fatick",
    "Kaolack",
    "Kaffrine",
    "Louga",
    "Saint-Louis",
    "Matam",
    "Tambacounda",
    "Kédougou",
    "Kolda",
    "Sédhiou",
    "Ziguinchor"
]

# Définition des zones climatiques du Sénégal en fonction de la pluviométrie moyenne annuelle
# Données basées sur les recherches (pluviométrie moyenne annuelle en mm)
region_pluie = {
    "Dakar": 450,         # Zone sahélienne côtière
    "Thiès": 500,         # Zone sahélienne côtière
    "Diourbel": 550,      # Zone sahélienne
    "Fatick": 650,        # Zone soudano-sahélienne
    "Kaolack": 700,       # Zone soudano-sahélienne
    "Kaffrine": 750,      # Zone soudano-sahélienne
    "Louga": 350,         # Zone sahélienne nord
    "Saint-Louis": 300,   # Zone sahélienne nord
    "Matam": 350,         # Zone sahélienne nord
    "Tambacounda": 800,   # Zone soudanienne
    "Kédougou": 1200,     # Zone soudano-guinéenne
    "Kolda": 1000,        # Zone soudano-guinéenne
    "Sédhiou": 1100,      # Zone soudano-guinéenne
    "Ziguinchor": 1300    # Zone soudano-guinéenne (Casamance)
}

# Définition des tendances mensuelles de précipitations (pourcentage de précipitations annuelles)
# Ces valeurs sont approximatives et varient selon les régions, mais suivent le schéma général
# de la saison des pluies au Sénégal (juillet à octobre principalement)
mois_pourcentage = {
    1: 0.1,    # Janvier - saison sèche
    2: 0.1,    # Février - saison sèche
    3: 0.2,    # Mars - saison sèche
    4: 0.5,    # Avril - début occasionnel de pluies dans le Sud
    5: 1.0,    # Mai - premières pluies possibles
    6: 5.0,    # Juin - début de la saison des pluies
    7: 20.0,   # Juillet - plein saison des pluies
    8: 35.0,   # Août - pic de la saison des pluies
    9: 25.0,   # Septembre - saison des pluies
    10: 10.0,  # Octobre - fin de la saison des pluies
    11: 2.0,   # Novembre - début de la saison sèche
    12: 1.0    # Décembre - saison sèche
}

# Fonction pour générer une valeur aléatoire qui suit une distribution normale
# avec des limites minimales et maximales
def random_bounded_normal(mean, std_dev, min_val, max_val):
    value = np.random.normal(mean, std_dev)
    return max(min_val, min(max_val, value))

# Fonction pour générer des données pluviométriques pour une région sur une période donnée
def generer_donnees_region(region, annee_debut, nb_annees):
    data = []
    
    # Pluviométrie moyenne annuelle pour la région
    pluviometrie_moyenne = region_pluie[region]
    
    # Pour chaque année
    for annee in range(annee_debut, annee_debut + nb_annees):
        # Générer une variation annuelle (certaines années sont plus sèches/humides)
        # Variation possible de ±30% pour simuler les années sèches et humides
        facteur_annuel = random_bounded_normal(1.0, 0.15, 0.7, 1.3)
        pluviometrie_annuelle = pluviometrie_moyenne * facteur_annuel
        
        # Pour chaque mois
        for mois in range(1, 13):
            # Calculer la pluviométrie mensuelle théorique
            pourcentage_mois = mois_pourcentage[mois]
            pluviometrie_mensuelle_theorique = pluviometrie_annuelle * pourcentage_mois / 100.0
            
            # Ajouter une variabilité mensuelle (±20%)
            facteur_mensuel = random_bounded_normal(1.0, 0.1, 0.8, 1.2)
            pluviometrie_mensuelle = pluviometrie_mensuelle_theorique * facteur_mensuel
            
            # Génération du nombre de jours de pluie dans le mois
            # Le nombre de jours de pluie est proportionnel à la pluviométrie
            # mais avec un cap selon le mois (max 30 jours en août, moins pour les autres mois)
            max_jours_pluie = min(30, int(15 * pourcentage_mois / 20))
            nb_jours_pluie = int(max(1, min(max_jours_pluie, pluviometrie_mensuelle / 15)))
            
            # Seulement pour les mois avec précipitations significatives
            if pluviometrie_mensuelle > 1.0:
                # Générer des données journalières
                date_debut = datetime(annee, mois, 1)
                nb_jours_dans_mois = (date_debut.replace(month=mois % 12 + 1, day=1) if mois < 12 
                                     else datetime(annee + 1, 1, 1)) - date_debut
                nb_jours_dans_mois = nb_jours_dans_mois.days
                
                # Sélectionner des jours aléatoires pour les précipitations
                jours_pluie = sorted(random.sample(range(1, nb_jours_dans_mois + 1), nb_jours_pluie))
                
                # Répartir la pluviométrie mensuelle sur les jours sélectionnés
                # en suivant approximativement une distribution log-normale
                # (quelques gros épisodes pluvieux et plusieurs petites pluies)
                pluies_journalieres = np.random.lognormal(mean=0.0, sigma=1.0, size=nb_jours_pluie)
                pluies_journalieres = pluies_journalieres / np.sum(pluies_journalieres) * pluviometrie_mensuelle
                
                for idx, jour in enumerate(jours_pluie):
                    date = date_debut + timedelta(days=jour-1)
                    pluie = round(pluies_journalieres[idx], 1)
                    
                    # Ajouter d'autres variables météorologiques corrélées avec la pluie
                    # Température moyenne : plus basse les jours de pluie
                    temp_min_base = {
                        1: 18, 2: 18, 3: 18, 4: 19, 5: 21, 6: 23, 
                        7: 25, 8: 25, 9: 25, 10: 25, 11: 23, 12: 20
                    }
                    temp_max_base = {
                        1: 26, 2: 26, 3: 26, 4: 25, 5: 26, 6: 29, 
                        7: 30, 8: 30, 9: 31, 10: 31, 11: 30, 12: 28
                    }
                    
                    # Ajustement selon la région (nord plus chaud, sud plus frais)
                    ajustement_temp = 0
                    if region in ["Saint-Louis", "Matam", "Louga"]:
                        ajustement_temp = 2  # Plus chaud dans le nord
                    elif region in ["Ziguinchor", "Kolda", "Sédhiou", "Kédougou"]:
                        ajustement_temp = -2  # Plus frais dans le sud
                    
                    # Réduction de température les jours de pluie (corrélation)
                    reduction_temp = min(5, pluie / 5)
                    
                    temp_min = temp_min_base[mois] + ajustement_temp - reduction_temp/2
                    temp_max = temp_max_base[mois] + ajustement_temp - reduction_temp
                    
                    # Humidité relative: plus élevée les jours de pluie
                    base_humidite = {
                        1: 40, 2: 40, 3: 45, 4: 50, 5: 60, 6: 65, 
                        7: 75, 8: 80, 9: 80, 10: 75, 11: 60, 12: 50
                    }
                    # Augmentation d'humidité selon la pluie
                    humidite = min(98, base_humidite[mois] + min(30, pluie * 2))
                    
                    # Vent (en km/h) - valeurs moyennes avec légère variabilité
                    vent_moyen = 15 + random.randint(-5, 5)
                    
                    # Pression atmosphérique (en hPa) - valeurs typiques pour les tropiques
                    # Baisse lors des épisodes pluvieux
                    pression = 1013 - (pluie / 10) + random.randint(-3, 3)
                    
                    data.append({
                        'region': region,
                        'date': date,
                        'annee': annee,
                        'mois': mois,
                        'jour': jour,
                        'precipitations': pluie,
                        'temperature_min': round(temp_min, 1),
                        'temperature_max': round(temp_max, 1),
                        'humidite': round(humidite, 1),
                        'vent': vent_moyen,
                        'pression': round(pression, 1)
                    })
                
                # Ajouter les jours sans pluie pour compléter le mois
                jours_sans_pluie = [j for j in range(1, nb_jours_dans_mois + 1) if j not in jours_pluie]
                for jour in jours_sans_pluie:
                    date = date_debut + timedelta(days=jour-1)
                    
                    # Variables météo pour les jours sans pluie
                    temp_min = temp_min_base[mois] + ajustement_temp + random.randint(-1, 1)
                    temp_max = temp_max_base[mois] + ajustement_temp + random.randint(-1, 3)
                    humidite = base_humidite[mois] + random.randint(-10, 10)
                    vent_moyen = 15 + random.randint(-5, 10)
                    pression = 1015 + random.randint(-2, 2)
                    
                    data.append({
                        'region': region,
                        'date': date,
                        'annee': annee,
                        'mois': mois,
                        'jour': jour,
                        'precipitations': 0.0,
                        'temperature_min': round(temp_min, 1),
                        'temperature_max': round(temp_max, 1),
                        'humidite': round(humidite, 1),
                        'vent': vent_moyen,
                        'pression': round(pression, 1)
                    })
            else:
                # Pour les mois sans précipitations significatives, générer des jours sans pluie
                date_debut = datetime(annee, mois, 1)
                nb_jours_dans_mois = (date_debut.replace(month=mois % 12 + 1, day=1) if mois < 12 
                                     else datetime(annee + 1, 1, 1)) - date_debut
                nb_jours_dans_mois = nb_jours_dans_mois.days
                
                for jour in range(1, nb_jours_dans_mois + 1):
                    date = date_debut + timedelta(days=jour-1)
                    
                    # Ajustement selon la région
                    ajustement_temp = 0
                    if region in ["Saint-Louis", "Matam", "Louga"]:
                        ajustement_temp = 2  # Plus chaud dans le nord
                    elif region in ["Ziguinchor", "Kolda", "Sédhiou", "Kédougou"]:
                        ajustement_temp = -2  # Plus frais dans le sud
                    
                    # Température: valeurs typiques pour le mois sans pluie
                    temp_min_base = {
                        1: 18, 2: 18, 3: 18, 4: 19, 5: 21, 6: 23, 
                        7: 25, 8: 25, 9: 25, 10: 25, 11: 23, 12: 20
                    }
                    temp_max_base = {
                        1: 26, 2: 26, 3: 26, 4: 25, 5: 26, 6: 29, 
                        7: 30, 8: 30, 9: 31, 10: 31, 11: 30, 12: 28
                    }
                    
                    temp_min = temp_min_base[mois] + ajustement_temp + random.randint(-1, 1)
                    temp_max = temp_max_base[mois] + ajustement_temp + random.randint(-1, 3)
                    
                    # Humidité relative: valeurs typiques pour le mois sans pluie
                    base_humidite = {
                        1: 40, 2: 40, 3: 45, 4: 50, 5: 60, 6: 65, 
                        7: 75, 8: 80, 9: 80, 10: 75, 11: 60, 12: 50
                    }
                    humidite = base_humidite[mois] + random.randint(-10, 10)
                    
                    # Vent (en km/h)
                    vent_moyen = 15 + random.randint(-5, 10)
                    
                    # Pression atmosphérique (en hPa)
                    pression = 1015 + random.randint(-2, 2)
                    
                    data.append({
                        'region': region,
                        'date': date,
                        'annee': annee,
                        'mois': mois,
                        'jour': jour,
                        'precipitations': 0.0,
                        'temperature_min': round(temp_min, 1),
                        'temperature_max': round(temp_max, 1),
                        'humidite': round(humidite, 1),
                        'vent': vent_moyen,
                        'pression': round(pression, 1)
                    })
    
    return pd.DataFrame(data)

# Fonction pour générer des données pour toutes les régions
def generer_toutes_regions(annee_debut, nb_annees):
    all_data = []
    
    for region in regions:
        print(f"Génération des données pour la région de {region}...")
        df_region = generer_donnees_region(region, annee_debut, nb_annees)
        all_data.append(df_region)
    
    # Combiner toutes les données en un seul DataFrame
    df_combined = pd.concat(all_data, ignore_index=True)
    return df_combined

# Ajouter des caractéristiques géographiques pour chaque région
def ajouter_caracteristiques_geographiques(df):
    # Dictionnaire des coordonnées approximatives pour chaque région (latitude, longitude)
    coords = {
        "Dakar": (14.7167, -17.4677),
        "Thiès": (14.7833, -16.9333),
        "Diourbel": (14.7167, -16.2333),
        "Fatick": (14.3333, -16.4167),
        "Kaolack": (14.1667, -16.0667),
        "Kaffrine": (14.1017, -15.5500),
        "Louga": (15.6167, -16.2333),
        "Saint-Louis": (16.0333, -16.5000),
        "Matam": (15.6500, -13.2500),
        "Tambacounda": (13.7667, -13.6667),
        "Kédougou": (12.5500, -12.1833),
        "Kolda": (12.9000, -14.9500),
        "Sédhiou": (12.7000, -15.7000),
        "Ziguinchor": (12.5667, -16.2667)
    }
    
    # Altitude moyenne approximative pour chaque région (en mètres)
    altitude = {
        "Dakar": 10,
        "Thiès": 70,
        "Diourbel": 15,
        "Fatick": 10,
        "Kaolack": 7,
        "Kaffrine": 18,
        "Louga": 40,
        "Saint-Louis": 4,
        "Matam": 15,
        "Tambacounda": 50,
        "Kédougou": 178,
        "Kolda": 10,
        "Sédhiou": 15,
        "Ziguinchor": 13
    }
    
    # Type de sol pour chaque région (simplifié)
    sol = {
        "Dakar": "Sableux",
        "Thiès": "Sableux-argileux",
        "Diourbel": "Sableux",
        "Fatick": "Sablo-argileux",
        "Kaolack": "Argileux",
        "Kaffrine": "Argileux",
        "Louga": "Sableux",
        "Saint-Louis": "Sableux-limoneux",
        "Matam": "Limoneux-argileux",
        "Tambacounda": "Argilo-sableux",
        "Kédougou": "Latéritique",
        "Kolda": "Argileux",
        "Sédhiou": "Argileux",
        "Ziguinchor": "Sablo-argileux"
    }
    
    # Distance à la côte (en km, approximatif)
    distance_cote = {
        "Dakar": 0,
        "Thiès": 20,
        "Diourbel": 80,
        "Fatick": 30,
        "Kaolack": 60,
        "Kaffrine": 120,
        "Louga": 50,
        "Saint-Louis": 0,
        "Matam": 400,
        "Tambacounda": 250,
        "Kédougou": 350,
        "Kolda": 120,
        "Sédhiou": 80,
        "Ziguinchor": 60
    }
    
    # Caractéristiques pour chaque région
    df['latitude'] = df['region'].map(lambda r: coords[r][0])
    df['longitude'] = df['region'].map(lambda r: coords[r][1])
    df['altitude'] = df['region'].map(altitude)
    df['type_sol'] = df['region'].map(sol)
    df['distance_cote'] = df['region'].map(distance_cote)
    
    return df

# Fonction principale pour générer et sauvegarder le jeu de données
def generer_dataset(annee_debut=2015, nb_annees=10, fichier_sortie="data/donnees_pluviometriques_senegal2.csv"):
    print(f"Génération de données pluviométriques pour {nb_annees} ans à partir de {annee_debut}...")
    
    # Générer les données de base
    df = generer_toutes_regions(annee_debut, nb_annees)
    
    # Ajouter des caractéristiques géographiques
    df = ajouter_caracteristiques_geographiques(df)
    
    # Convertir la date en format string pour faciliter l'écriture CSV
    df['date'] = df['date'].dt.strftime('%Y-%m-%d')
    
    # Sauvegarder en CSV
    df.to_csv(fichier_sortie, index=False)
    print(f"Données sauvegardées dans {fichier_sortie}")
    
    # Afficher quelques statistiques
    print("\nQuelques statistiques sur les données générées:")
    print(f"Nombre total d'enregistrements: {len(df)}")
    
    # Précipitations moyennes annuelles par région
    precipitations_par_region = df.groupby('region')['precipitations'].sum().reset_index()
    precipitations_par_region['precipitations_annuelles'] = precipitations_par_region['precipitations'] / nb_annees
    print("\nPrécipitations moyennes annuelles par région:")
    for _, row in precipitations_par_region.sort_values('precipitations_annuelles', ascending=False).iterrows():
        print(f"  {row['region']}: {row['precipitations_annuelles']:.1f} mm")
    
    return df

# Exécuter la génération de données
df = generer_dataset(annee_debut=2015, nb_annees=10, fichier_sortie="data/donnees_pluviometriques_senegal2.csv")

# Afficher les premières lignes du jeu de données
print("\nAperçu des données générées:")
print(df.head())

# Visualisations (facultatif - commentez si vous souhaitez seulement générer les données)
try:
    # Visualisation des précipitations annuelles moyennes par région
    precipitations_par_region = df.groupby('region')['precipitations'].sum().reset_index()
    precipitations_par_region['precipitations_annuelles'] = precipitations_par_region['precipitations'] / 10  # pour 10 ans

    plt.figure(figsize=(12, 6))
    sns.barplot(x='region', y='precipitations_annuelles', data=precipitations_par_region.sort_values('precipitations_annuelles', ascending=False))
    plt.title('Précipitations moyennes annuelles par région au Sénégal')
    plt.xlabel('Région')
    plt.ylabel('Pluviométrie moyenne (mm/an)')
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig('output/figures/precipitations_moyennes_annuelles.png')
    plt.show()
    
    # Visualisation de la distribution mensuelle des précipitations
    # Agrégation des données par mois
    df['date'] = pd.to_datetime(df['date'])
    precipitations_mensuelles = df.groupby(['region', 'mois'])['precipitations'].sum().reset_index()
    precipitations_mensuelles = precipitations_mensuelles.pivot(index='region', columns='mois', values='precipitations')
    precipitations_mensuelles = precipitations_mensuelles.div(10)  # Moyenne sur 10 ans

    # Sélection de quelques régions représentatives
    regions_selectionnees = ["Dakar", "Saint-Louis", "Kaolack", "Ziguinchor"]
    df_plot = precipitations_mensuelles.loc[regions_selectionnees]

    plt.figure(figsize=(14, 8))
    for idx, region in enumerate(regions_selectionnees):
        plt.subplot(2, 2, idx+1)
        plt.bar(range(1, 13), df_plot.loc[region])
        plt.title(f'Distribution mensuelle des précipitations - {region}')
        plt.xlabel('Mois')
        plt.ylabel('Pluviométrie moyenne (mm)')
        plt.xticks(range(1, 13), ['Jan', 'Fév', 'Mar', 'Avr', 'Mai', 'Juin', 'Juil', 'Août', 'Sep', 'Oct', 'Nov', 'Déc'])

    plt.tight_layout()
    plt.savefig('output/figures/distribution_mensuelle_precipitations.png')
    plt.show()
    
    print("Visualisations générées et sauvegardées.")
except Exception as e:
    print(f"Visualisations non générées. Erreur : {e}")
    print("Vous pouvez ignorer cette erreur si vous n'avez pas besoin des visualisations.")