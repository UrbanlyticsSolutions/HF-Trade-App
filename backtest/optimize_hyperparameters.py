"""
Hyperparameter Optimization using Optuna
Finds optimal CatBoost parameters for trading classification
"""
import optuna
import numpy as np
import pandas as pd
from ml_trade_classifier import MLTradeClassifier
from sklearn.model_selection import cross_val_score
from sklearn.preprocessing import StandardScaler
import json
import os

def objective(trial):
    """Optuna objective function to maximize F1-score"""
    
    # Load training data
    if os.path.exists('training_data_real.csv'):
        data_file = 'training_data_real.csv'
    else:
        print("Error: training_data_real.csv not found")
        return 0.0
    
    # Suggest hyperparameters
    params = {
        'iterations': trial.suggest_int('iterations', 300, 1000, step=100),
        'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.15, log=True),
        'depth': trial.suggest_int('depth', 4, 10),
        'l2_leaf_reg': trial.suggest_float('l2_leaf_reg', 1.0, 10.0),
        'subsample': trial.suggest_float('subsample', 0.6, 1.0),
        'min_data_in_leaf': trial.suggest_int('min_data_in_leaf', 1, 20),
        'random_seed': 42,
        'verbose': False,
        'auto_class_weights': 'Balanced'
    }
    
    # Create temporary classifier with suggested params
    from catboost import CatBoostClassifier
    model = CatBoostClassifier(**params)
    
    # Load and prepare data
    df = pd.read_csv(data_file)
    if 'profitable' in df.columns:
        df['label'] = df['profitable'].astype(int)
    else:
        df['label'] = (df['PnL'] > 0).astype(int)
    
    # Extract features
    classifier = MLTradeClassifier()
    features_list = []
    labels = []
    
    for idx, row in df.iterrows():
        trade_data = row.to_dict()
        features = classifier.extract_features(trade_data)
        features_list.append(list(features.values()))
        labels.append(row['label'])
    
    X = np.array(features_list)
    y = np.array(labels)
    
    # Scale features
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    # Use cross-validation with F1-score
    from sklearn.metrics import make_scorer, f1_score
    f1_scorer = make_scorer(f1_score)
    
    scores = cross_val_score(model, X_scaled, y, cv=5, scoring=f1_scorer, n_jobs=-1)
    
    return scores.mean()

def optimize_hyperparameters(n_trials=50):
    """Run Optuna optimization"""
    
    print("="*80)
    print("CATBOOST HYPERPARAMETER OPTIMIZATION")
    print("="*80)
    print(f"Running {n_trials} trials...")
    print()
    
    # Create study
    study = optuna.create_study(
        direction='maximize',
        sampler=optuna.samplers.TPESampler(seed=42)
    )
    
    # Optimize
    study.optimize(objective, n_trials=n_trials, show_progress_bar=True)
    
    # Results
    print("\n" + "="*80)
    print("OPTIMIZATION RESULTS")
    print("="*80)
    print(f"Best F1-Score: {study.best_value:.4f}")
    print(f"Best Parameters:")
    for key, value in study.best_params.items():
        print(f"  {key}: {value}")
    print("="*80)
    
    # Save best parameters
    os.makedirs('models', exist_ok=True)
    with open('models/best_catboost_params.json', 'w') as f:
        json.dump(study.best_params, f, indent=2)
    
    print("\n✓ Best parameters saved to models/best_catboost_params.json")
    
    return study.best_params

if __name__ == "__main__":
    best_params = optimize_hyperparameters(n_trials=50)
