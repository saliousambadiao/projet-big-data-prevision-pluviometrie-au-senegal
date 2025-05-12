"""
Système de prédiction pluviométrique au Sénégal
===============================================
Point d'entrée principal du projet permettant de traiter les données, 
entraîner les modèles et réaliser des prédictions.
"""

import argparse
import sys
import os
from datetime import datetime

# Initialisation des arguments de ligne de commande
def parse_args():
    parser = argparse.ArgumentParser(description="Système de prédiction pluviométrique au Sénégal")
    
    # Sous-parseurs pour les différentes commandes
    subparsers = parser.add_subparsers(dest="command", help="Commande à exécuter")
    
    # Sous-parseur pour le traitement des données
    process_parser = subparsers.add_parser("process", help="Traiter les données pluviométriques")
    process_parser.add_argument("--input", default="data/donnees_pluviometriques_senegal.csv", 
                              help="Chemin du fichier de données d'entrée")
    process_parser.add_argument("--output", default="data/donnees_pluviometriques_nettoyees.csv", 
                              help="Chemin de sortie des données nettoyées")
    
    # Sous-parseur pour l'entraînement des modèles
    train_parser = subparsers.add_parser("train", help="Entraîner les modèles de prédiction")
    train_parser.add_argument("--data", default="data/donnees_pluviometriques_senegal.csv", 
                            help="Chemin du fichier de données")
    train_parser.add_argument("--models", "-m", default=["rf", "gbt"], nargs="+", 
                           choices=["rf", "gbt", "prophet"], 
                           help="Modèles à entraîner (rf=Random Forest, gbt=Gradient Boosting, prophet=Prophet)")
    train_parser.add_argument("--region", default="Dakar",
                            help="Région pour le modèle Prophet (utilisé uniquement avec --models prophet)")
    
    # Sous-parseur pour la prédiction
    predict_parser = subparsers.add_parser("predict", help="Prédire la pluviométrie")
    predict_parser.add_argument("--region", default="Dakar",
                              help="Région pour laquelle faire la prédiction")
    predict_parser.add_argument("--model", default="gbt", choices=["rf", "gbt"],
                             help="Modèle à utiliser pour la prédiction (rf=Random Forest, gbt=Gradient Boosting)")
    
    return parser.parse_args()

def main():
    # Analyser les arguments
    args = parse_args()
    
    # Exécuter la commande appropriée
    if args.command == "process":
        # Traiter les données
        print(f"Traitement des données à partir de {args.input}...")
        from src.data_processing import create_spark_session, load_data, clean_dataframe
        
        # Créer la session Spark
        spark = create_spark_session()
        
        try:
            # Charger et nettoyer les données
            df = load_data(spark, args.input)
            cleaned_df = clean_dataframe(df)
            
            # Sauvegarder les données nettoyées
            dir_path = os.path.dirname(args.output)
            if dir_path and not os.path.exists(dir_path):
                os.makedirs(dir_path)
                
            cleaned_df.write.mode("overwrite").option("header", "true").csv(args.output)
            print(f"Données nettoyées sauvegardées dans {args.output}")
            
        finally:
            # Arrêter la session Spark
            spark.stop()
            
    elif args.command == "train":
        # Entraîner les modèles
        print(f"Entraînement des modèles {', '.join(args.models)} avec les données de {args.data}...")
        
        from src.data_processing import create_spark_session, load_data, clean_dataframe
        from src.feature_engineering import prepare_features_pipeline, split_train_test
        from src.model_training import (train_random_forest, train_gradient_boosting, 
                                     evaluate_model, save_model, visualize_predictions,
                                     train_prophet_model)
        
        # Créer la session Spark
        spark = create_spark_session("Entraînement des modèles de prédiction pluviométrique")
        
        try:
            # Charger et nettoyer les données
            df = load_data(spark, args.data)
            cleaned_df = clean_dataframe(df)
            
            # Préparer les features
            prepared_df, numeric_features, categorical_features, scaler_model = prepare_features_pipeline(cleaned_df)
            
            # Diviser en ensembles d'entraînement et de test
            train_df, test_df = split_train_test(prepared_df)
            
            # Entraîner les modèles spécifiés
            for model_type in args.models:
                if model_type == "rf":
                    print("\n----- RANDOM FOREST -----")
                    rf_model = train_random_forest(train_df)
                    rf_results = evaluate_model(rf_model, test_df, "Random Forest")
                    save_model(rf_model, "rf_model")
                    
                    if "gbt" not in args.models:
                        # Si GBT n'est pas entraîné, on ne peut pas visualiser la comparaison
                        continue
                    
                elif model_type == "gbt":
                    print("\n----- GRADIENT BOOSTED TREES -----")
                    gbt_model = train_gradient_boosting(train_df)
                    gbt_results = evaluate_model(gbt_model, test_df, "Gradient Boosted Trees")
                    save_model(gbt_model, "gbt_model")
                    
                    if "rf" not in args.models:
                        # Si RF n'est pas entraîné, on ne peut pas visualiser la comparaison
                        continue
                    
                elif model_type == "prophet":
                    print("\n----- PROPHET (LONG TERME) -----")
                    # Prophet nécessite une région spécifique
                    train_prophet_model(cleaned_df, args.region)
            
            # Si RF et GBT ont été entraînés, visualiser la comparaison
            if "rf" in args.models and "gbt" in args.models:
                visualize_predictions(rf_results, gbt_results)
                
        finally:
            # Arrêter la session Spark
            spark.stop()
            
    elif args.command == "predict":
        # Prédire la pluviométrie
        print(f"Prédiction de la pluviométrie pour {args.region} avec le modèle {args.model}...")
        
        from src.prediction import predict_for_tomorrow
        
        try:
            # Effectuer la prédiction
            result = predict_for_tomorrow(args.region, args.model)
            
            # Afficher le chemin de la visualisation
            print(f"\nVisualisation disponible: {result['visualization']}")
            
        except Exception as e:
            print(f"Erreur lors de la prédiction: {e}")
            sys.exit(1)
            
    else:
        # Afficher l'aide si aucune commande n'est spécifiée
        print("Aucune commande spécifiée. Utilisez --help pour voir les options disponibles.")
        sys.exit(1)

if __name__ == "__main__":
    main()
