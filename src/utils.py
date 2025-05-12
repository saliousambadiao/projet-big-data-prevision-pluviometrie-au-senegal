"""
Utilitaires pour le projet de prédiction pluviométrique au Sénégal
==================================================================
Ce module contient des fonctions utilitaires réutilisables.
"""

import sys
import subprocess
import os
from datetime import datetime, timedelta
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

def install_package(package):
    """
    Vérifie si un package est installé et l'installe si nécessaire.
    
    Args:
        package (str): Nom du package à installer
    """
    try:
        __import__(package)
        print(f"{package} est déjà installé.")
    except ImportError:
        print(f"Installation de {package}...")
        subprocess.check_call([sys.executable, "-m", "pip", "install", package])
        print(f"{package} a été installé avec succès.")

def ensure_directory_exists(path):
    """
    Crée un répertoire s'il n'existe pas déjà.
    
    Args:
        path (str): Chemin du répertoire à créer
    """
    if not os.path.exists(path):
        os.makedirs(path)
        print(f"Répertoire créé: {path}")

def format_date(date):
    """
    Formate une date pour affichage ou pour les noms de fichiers.
    
    Args:
        date (datetime): Date à formater
        
    Returns:
        str: Date formatée au format jj/mm/aaaa
    """
    return date.strftime('%d/%m/%Y')

def get_tomorrow_date():
    """
    Retourne la date de demain.
    
    Returns:
        datetime: Date de demain
    """
    return datetime.now() + timedelta(days=1)

def get_season(month):
    """
    Détermine la saison pour un mois donné au Sénégal.
    
    Args:
        month (int): Numéro du mois (1-12)
        
    Returns:
        str: Saison ('saison_pluies' ou 'saison_seche')
    """
    if 6 <= month <= 10:
        return "saison_pluies"
    else:
        return "saison_seche"

def save_visualization(fig, filename, title=None, output_dir='output/figures'):
    """
    Sauvegarde une visualisation matplotlib dans le répertoire de sortie.
    
    Args:
        fig (matplotlib.figure.Figure): Figure à sauvegarder
        filename (str): Nom du fichier
        title (str, optional): Titre de la figure
        output_dir (str, optional): Répertoire de sortie
    """
    ensure_directory_exists(output_dir)
    if title:
        plt.title(title)
    plt.tight_layout()
    full_path = os.path.join(output_dir, filename)
    fig.savefig(full_path)
    print(f"Visualisation sauvegardée: {full_path}")
    
def classify_rainfall(precipitation):
    """
    Classifie les précipitations selon leur intensité.
    
    Args:
        precipitation (float): Valeur des précipitations en mm
        
    Returns:
        str: Catégorie des précipitations
    """
    if precipitation < 0.1:
        return "Temps sec"
    elif precipitation < 5:
        return "Pluie légère"
    elif precipitation < 20:
        return "Pluie modérée"
    else:
        return "Forte pluie"
