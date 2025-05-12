"""
Ingénierie des features pour la prédiction pluviométrique au Sénégal
====================================================================
Ce module gère la création et la transformation des features pour les modèles prédictifs.
"""

import findspark
findspark.init()

from pyspark.sql.functions import col, dayofyear, when, lit
from pyspark.sql.types import IntegerType
from pyspark.ml.feature import VectorAssembler, StandardScaler, StringIndexer, OneHotEncoder

def add_temporal_features(df):
    """
    Ajoute des features temporelles au DataFrame.
    
    Args:
        df: DataFrame PySpark avec données pluviométriques
        
    Returns:
        DataFrame avec features temporelles additionnelles
    """
    print("Ajout des features temporelles...")
    
    # Jour de l'année (1-366)
    df = df.withColumn("jour_annee", dayofyear(col("date")))
    
    # Saison (saison des pluies: juin à octobre, saison sèche: le reste)
    df = df.withColumn("saison", 
                      when((col("mois") >= 6) & (col("mois") <= 10), "saison_pluies")
                      .otherwise("saison_seche"))
    
    # Trimestre (1-4)
    df = df.withColumn("trimestre", ((col("mois") - 1) / 3 + 1).cast(IntegerType()))
    
    print("Features temporelles ajoutées: jour_annee, saison, trimestre")
    return df

def encode_categorical_features(df):
    """
    Encode les variables catégorielles avec one-hot encoding.
    
    Args:
        df: DataFrame PySpark
        
    Returns:
        DataFrame avec variables catégorielles encodées
    """
    print("Encodage des variables catégorielles...")
    
    # Définir les colonnes catégorielles
    categorical_cols = ["region", "type_sol", "saison"]
    indexed_cols = [f"{col}_idx" for col in categorical_cols]
    encoded_cols = [f"{col}_enc" for col in categorical_cols]
    
    # Création des indexeurs et encodeurs
    indexers = [StringIndexer(inputCol=col, outputCol=f"{col}_idx", handleInvalid="keep") 
               for col in categorical_cols]
    
    # Appliquer les indexeurs
    for indexer in indexers:
        df = indexer.fit(df).transform(df)
    
    # Appliquer l'encodage one-hot
    encoder = OneHotEncoder(inputCols=indexed_cols, outputCols=encoded_cols)
    df = encoder.fit(df).transform(df)
    
    print(f"Variables catégorielles encodées: {', '.join(categorical_cols)}")
    return df, encoded_cols

def assemble_features(df, numeric_features, categorical_features, output_col="features"):
    """
    Assemble les features numériques et catégorielles dans un vecteur.
    
    Args:
        df: DataFrame PySpark
        numeric_features: Liste des colonnes numériques
        categorical_features: Liste des colonnes catégorielles encodées
        output_col: Nom de la colonne de sortie
        
    Returns:
        DataFrame avec les features assemblées
    """
    print("Assemblage des features...")
    
    # Combinaison des features numériques et catégorielles
    all_features = numeric_features + categorical_features
    
    # Création de l'assembleur de vecteurs
    assembler = VectorAssembler(
        inputCols=all_features, 
        outputCol=output_col,
        handleInvalid="skip"
    )
    
    # Appliquer l'assembleur
    df_assembled = assembler.transform(df)
    
    print(f"Features assemblées dans la colonne '{output_col}'")
    return df_assembled

def scale_features(df, input_col="features", output_col="scaled_features"):
    """
    Normalise les features pour améliorer les performances des modèles.
    
    Args:
        df: DataFrame PySpark avec features assemblées
        input_col: Nom de la colonne des features à normaliser
        output_col: Nom de la colonne de sortie
        
    Returns:
        DataFrame avec features normalisées et modèle de scaling
    """
    print("Normalisation des features...")
    
    # Création du scaler
    scaler = StandardScaler(
        inputCol=input_col,
        outputCol=output_col,
        withStd=True,
        withMean=True
    )
    
    # Ajustement et transformation
    scaler_model = scaler.fit(df)
    df_scaled = scaler_model.transform(df)
    
    print(f"Features normalisées dans la colonne '{output_col}'")
    return df_scaled, scaler_model

def prepare_features_pipeline(df):
    """
    Applique l'ensemble du pipeline de préparation des features.
    
    Args:
        df: DataFrame PySpark avec données brutes
        
    Returns:
        DataFrame prêt pour l'entraînement, listes des features, modèle de scaling
    """
    # Définir les features numériques d'intérêt
    numeric_features = ["temperature_min", "temperature_max", "humidite", 
                       "vent", "pression", "jour_annee", "latitude", 
                       "longitude", "altitude", "distance_cote"]
    
    # 1. Ajouter les features temporelles
    df = add_temporal_features(df)
    
    # 2. Encoder les variables catégorielles
    df, encoded_categorical_features = encode_categorical_features(df)
    
    # 3. Assembler les features
    df_assembled = assemble_features(df, numeric_features, encoded_categorical_features)
    
    # 4. Normaliser les features
    df_scaled, scaler_model = scale_features(df_assembled)
    
    return df_scaled, numeric_features, encoded_categorical_features, scaler_model

def split_train_test(df, test_year=2021):
    """
    Divise les données en ensembles d'entraînement et de test.
    
    Args:
        df: DataFrame PySpark
        test_year: Année à partir de laquelle commencent les données de test
        
    Returns:
        DataFrame d'entraînement, DataFrame de test
    """
    print(f"Division des données: entraînement (<{test_year}) et test (≥{test_year})...")
    
    # Diviser les données par année
    train_df = df.filter(col("annee") < test_year)
    test_df = df.filter(col("annee") >= test_year)
    
    # Vérifier les proportions
    train_count = train_df.count()
    test_count = test_df.count()
    total = train_count + test_count
    
    print(f"Ensemble d'entraînement: {train_count} enregistrements ({train_count/total*100:.1f}%)")
    print(f"Ensemble de test: {test_count} enregistrements ({test_count/total*100:.1f}%)")
    
    return train_df, test_df
