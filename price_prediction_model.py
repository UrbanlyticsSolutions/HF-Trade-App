"""
Intraday Price Prediction Model
Predicts next-bar price direction (UP/DOWN/NEUTRAL)
"""
import numpy as np
import pandas as pd
import joblib
import logging
import os
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report
from sklearn.preprocessing import StandardScaler

try:
    from catboost import CatBoostClassifier
    CATBOOST_AVAILABLE = True
except ImportError:
    CATBOOST_AVAILABLE = False

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger('PricePredictionModel')

class PricePredictionModel:
    """
    Predicts next-bar price movement:
    - 0: DOWN (price will decrease)
    - 1: NEUTRAL (price stays flat)
    - 2: UP (price will increase)
    """
    
    def __init__(self, model_path="models/price_predictor.pkl", model_type='catboost'):
        self.model_path = model_path
        self.model_type = model_type
        self.model = None
        self.scaler = StandardScaler()
        
        # Features for prediction
        self.feature_names = [
            # Price features
            'price_change_1', 'price_change_3', 'price_change_5',
            'price_volatility',
            
            # Moving averages
            'ma_5', 'ma_10', 'ma_20',
            'price_to_ma5', 'price_to_ma20',
            
            # Momentum
            'rsi', 'macd', 'macd_hist',
            
            # Volume
            'volume_ratio', 'volume_change',
            
            # Trend
            'adx', 'trend_strength',
            
            # Time
            'hour', 'minute'
        ]
        
        os.makedirs(os.path.dirname(model_path), exist_ok=True)
    
    def _initialize_model(self):
        """Initialize the ML model"""
        if self.model_type == 'catboost' and CATBOOST_AVAILABLE:
            self.model = CatBoostClassifier(
                iterations=500,
                learning_rate=0.1,
                depth=6,
                loss_function='MultiClass',
                verbose=False,
                random_seed=42
            )
        elif self.model_type == 'gradient_boosting':
            self.model = GradientBoostingClassifier(
                n_estimators=200,
                learning_rate=0.1,
                max_depth=5,
                random_state=42
            )
        else:  # random_forest
            self.model = RandomForestClassifier(
                n_estimators=200,
                max_depth=10,
                random_state=42
            )
    
    def extract_features(self, data_row):
        """Extract features from a data point"""
        features = {
            'price_change_1': data_row.get('price_change_1', 0),
            'price_change_3': data_row.get('price_change_3', 0),
            'price_change_5': data_row.get('price_change_5', 0),
            'price_volatility': data_row.get('price_volatility', 0),
            'ma_5': data_row.get('ma_5', 0),
            'ma_10': data_row.get('ma_10', 0),
            'ma_20': data_row.get('ma_20', 0),
            'price_to_ma5': data_row.get('price_to_ma5', 1.0),
            'price_to_ma20': data_row.get('price_to_ma20', 1.0),
            'rsi': data_row.get('rsi', 50),
            'macd': data_row.get('macd', 0),
            'macd_hist': data_row.get('macd_hist', 0),
            'volume_ratio': data_row.get('volume_ratio', 1.0),
            'volume_change': data_row.get('volume_change', 0),
            'adx': data_row.get('adx', 0),
            'trend_strength': data_row.get('trend_strength', 0),
            'hour': data_row.get('hour', 12),
            'minute': data_row.get('minute', 0)
        }
        
        return [features[k] for k in self.feature_names]
    
    def prepare_training_data(self, data_file):
        """Prepare training data from CSV"""
        df = pd.read_csv(data_file)
        
        # Extract features
        X = []
        y = []
        
        for idx, row in df.iterrows():
            features = self.extract_features(row.to_dict())
            X.append(features)
            y.append(row['target'])  # 0=DOWN, 1=NEUTRAL, 2=UP
        
        return np.array(X), np.array(y)
    
    def train(self, X, y, test_size=0.2):
        """Train the prediction model"""
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=42, stratify=y
        )
        
        # Scale features
        X_train_scaled = self.scaler.fit_transform(X_train)
        X_test_scaled = self.scaler.transform(X_test)
        
        # Train model
        self._initialize_model()
        logger.info(f"Training {self.model_type} model...")
        self.model.fit(X_train_scaled, y_train)
        
        # Evaluate
        train_acc = self.model.score(X_train_scaled, y_train)
        test_acc = self.model.score(X_test_scaled, y_test)
        
        y_pred = self.model.predict(X_test_scaled)
        
        logger.info(f"Train Accuracy: {train_acc:.2%}")
        logger.info(f"Test Accuracy: {test_acc:.2%}")
        logger.info("\nClassification Report:")
        logger.info(classification_report(y_test, y_pred, target_names=['DOWN', 'NEUTRAL', 'UP']))
        
        return {
            'train_accuracy': train_acc,
            'test_accuracy': test_acc,
            'classification_report': classification_report(y_test, y_pred, output_dict=True)
        }
    
    def predict(self, features):
        """
        Predict next price movement
        
        Returns:
            0: DOWN
            1: NEUTRAL  
            2: UP
        """
        if self.model is None:
            return 1  # Default to NEUTRAL
        
        X = np.array([features])
        X_scaled = self.scaler.transform(X)
        prediction = self.model.predict(X_scaled)[0]
        
        return int(prediction)
    
    def predict_proba(self, features):
        """Get prediction probabilities"""
        if self.model is None:
            return [0.33, 0.34, 0.33]  # Equal probabilities
        
        X = np.array([features])
        X_scaled = self.scaler.transform(X)
        probabilities = self.model.predict_proba(X_scaled)[0]
        
        return probabilities
    
    def save_model(self):
        """Save trained model"""
        model_data = {
            'model': self.model,
            'scaler': self.scaler,
            'feature_names': self.feature_names,
            'model_type': self.model_type
        }
        joblib.dump(model_data, self.model_path)
        logger.info(f"Model saved to {self.model_path}")
    
    def load_model(self):
        """Load trained model"""
        model_data = joblib.load(self.model_path)
        self.model = model_data['model']
        self.scaler = model_data['scaler']
        self.feature_names = model_data['feature_names']
        self.model_type = model_data['model_type']
        logger.info(f"Model loaded from {self.model_path}")

if __name__ == "__main__":
    print("Price Prediction Model")
    print("This model predicts next-bar price direction")
    print("Run 'python scripts/generate_prediction_data.py' first to create training data")
