"""
Traitement des données pluviométriques du Sénégal
=================================================
Ce module gère le chargement et le nettoyage des données pluviométriques.
"""

import os
import sys
import findspark
findspark.init()

from pyspark.sql import SparkSession
from pyspark.sql.types import StructType, StructField, StringType, IntegerType, DoubleType, DateType
from pyspark.sql.functions import col, count, isnan, when, sum as spark_sum

# Définition du schéma des données
SCHEMA = StructType([
    StructField("region", StringType(), False),
    StructField("date", DateType(), False),
    StructField("annee", IntegerType(), False),
    StructField("mois", IntegerType(), False),
    StructField("jour", IntegerType(), False),
    StructField("precipitations", DoubleType(), False),
    StructField("temperature_min", DoubleType(), False),
    StructField("temperature_max", DoubleType(), False),
    StructField("humidite", DoubleType(), False),
    StructField("vent", IntegerType(), False),
    StructField("pression", DoubleType(), False),
    StructField("latitude", DoubleType(), False),
    StructField("longitude", DoubleType(), False),
    StructField("altitude", IntegerType(), False),
    StructField("type_sol", StringType(), False),
    StructField("distance_cote", IntegerType(), False)
])

def create_spark_session(app_name="Traitement Données Pluviométriques Sénégal"):
    """
    Crée et retourne une session Spark.
    
    Args:
        app_name (str): Nom de l'application Spark
        
    Returns:
        SparkSession: Session Spark configurée
    """
    return SparkSession.builder \
        .appName(app_name) \
        .config("spark.executor.memory", "2g") \
        .config("spark.driver.memory", "2g") \
        .getOrCreate()

def load_data(spark, file_path="data/donnees_pluviometriques_senegal.csv"):
    """
    Charge les données pluviométriques depuis un fichier CSV.
    
    Args:
        spark (SparkSession): Session Spark active
        file_path (str): Chemin vers le fichier de données
        
    Returns:
        DataFrame: DataFrame Spark contenant les données
    """
    try:
        df = spark.read \
            .option("header", "true") \
            .option("dateFormat", "yyyy-MM-dd") \
            .schema(SCHEMA) \
            .csv(file_path)
        
        print(f"Données chargées avec succès depuis {file_path}")
        return df
    except Exception as e:
        print(f"Erreur lors du chargement des données: {e}")
        # Fallback: essayer d'inférer le schéma
        try:
            print("Tentative de chargement avec inférence de schéma...")
            df = spark.read \
                .option("header", "true") \
                .option("inferSchema", "true") \
                .csv(file_path)
            print("Données chargées avec inférence de schéma")
            return df
        except Exception as e2:
            print(f"Échec du chargement des données: {e2}")
            sys.exit(1)

def clean_dataframe(df, missing_threshold=0.05):
    """
    Nettoie le dataframe en traitant les valeurs manquantes selon leur proportion.
    
    Args:
        df: DataFrame PySpark à nettoyer
        missing_threshold: Seuil pour décider de supprimer ou d'imputer (par défaut 5%)
        
    Returns:
        DataFrame PySpark nettoyé
    """
    # Identifier les colonnes numériques et non-numériques
    numeric_cols = ["precipitations", "temperature_min", "temperature_max", 
                    "humidite", "pression", "latitude", "longitude", 
                    "altitude", "distance_cote", "vent"]
    non_numeric_cols = [c for c in df.columns if c not in numeric_cols]
    
    # Calculer le nombre de valeurs manquantes pour chaque colonne
    null_counts = df.select(
        # Pour les colonnes numériques, vérifier isNull() et isnan()
        [spark_sum(when(col(c).isNull() | isnan(col(c)), 1).otherwise(0)).alias(c) for c in numeric_cols] + 
        # Pour les colonnes non-numériques, vérifier seulement isNull()
        [spark_sum(when(col(c).isNull(), 1).otherwise(0)).alias(c) for c in non_numeric_cols]
    )
    
    print("Analyse des valeurs manquantes avant nettoyage:")
    null_counts.show()
    
    # Obtenir un dictionnaire des décomptes de valeurs nulles
    null_counts_dict = {column: null_counts.first()[column] for column in df.columns}
    
    # Nombre total d'enregistrements
    total_records = df.count()
    
    # Initialiser le DataFrame nettoyé
    cleaned_df = df
    
    print(f"Nettoyage des données avec un seuil de {missing_threshold*100}%...")
    
    # Pour chaque colonne avec des valeurs manquantes, décider de supprimer ou d'imputer
    for col_name in numeric_cols:
        if null_counts_dict.get(col_name, 0) > 0:
            null_ratio = null_counts_dict[col_name] / total_records
            
            if null_ratio > missing_threshold:
                # Si le pourcentage de valeurs manquantes est supérieur au seuil,
                # on remplace par la moyenne de la colonne
                avg_val = cleaned_df.select(col_name).agg({col_name: "avg"}).first()[f"avg({col_name})"]
                cleaned_df = cleaned_df.fillna(avg_val, subset=[col_name])
                print(f"Imputation de la colonne {col_name} ({null_ratio*100:.2f}% manquantes) avec la moyenne: {avg_val}")
            else:
                # Si le pourcentage est inférieur au seuil, on supprime les lignes
                cleaned_df = cleaned_df.filter(~(col(col_name).isNull() | isnan(col(col_name))))
                print(f"Suppression des lignes avec {col_name} manquant ({null_ratio*100:.2f}%)")
    
    # Pour les colonnes non numériques, on peut adopter d'autres stratégies
    # comme le remplacement par le mode (valeur la plus fréquente)
    for col_name in non_numeric_cols:
        if null_counts_dict.get(col_name, 0) > 0:
            if col_name == "type_sol" and df.schema[col_name].dataType == StringType():
                cleaned_df = cleaned_df.fillna("Standard", subset=[col_name])
                print(f"Imputation de la colonne {col_name} avec 'Standard'")
    
    # Vérifier les valeurs manquantes restantes
    final_null_counts = cleaned_df.select(
        [spark_sum(when(col(c).isNull() | isnan(col(c)), 1).otherwise(0)).alias(c) for c in numeric_cols] + 
        [spark_sum(when(col(c).isNull(), 1).otherwise(0)).alias(c) for c in non_numeric_cols]
    )
    
    print("\nVérification finale des valeurs manquantes après nettoyage:")
    final_null_counts.show()
    
    print(f"Données nettoyées: {cleaned_df.count()} lignes restantes sur {total_records} initiales")
    return cleaned_df

def analyze_data(df):
    """
    Réalise une analyse descriptive des données.
    
    Args:
        df: DataFrame PySpark à analyser
    """
    # Statistiques descriptives
    print("\nStatistiques descriptives des données:")
    df.describe().show()
    
    # Distribution par région
    print("\nDistribution des données par région:")
    df.groupBy("region").count().orderBy("count", ascending=False).show()
    
    # Moyenne des précipitations par région
    print("\nMoyenne des précipitations par région:")
    df.groupBy("region").agg({"precipitations": "mean"}) \
        .withColumnRenamed("avg(precipitations)", "precipitations_moyennes") \
        .orderBy("precipitations_moyennes", ascending=False) \
        .show()
    
    # Moyenne des précipitations par mois
    print("\nMoyenne des précipitations par mois:")
    df.groupBy("mois").agg({"precipitations": "mean"}) \
        .withColumnRenamed("avg(precipitations)", "precipitations_moyennes") \
        .orderBy("mois") \
        .show()

if __name__ == "__main__":
    # Code à exécuter si le script est lancé directement
    print("Lancement du traitement des données...")
    spark = create_spark_session()
    
    df = load_data(spark)
    cleaned_df = clean_dataframe(df)
    analyze_data(cleaned_df)
    
    # Sauvegarde des données nettoyées si nécessaire
    output_path = "data/donnees_pluviometriques_nettoyees.csv"
    cleaned_df.write.mode("overwrite").option("header", "true").csv(output_path)
    print(f"Données nettoyées sauvegardées dans {output_path}")
    
    spark.stop()
