"""
Market Analyzer
Centralizes logic for feature extraction, model prediction, and indicator calculation.
"""
import logging
import os
import pandas as pd
import numpy as np
from datetime import datetime

from core.features import build_feature_frame
from price_prediction_model import PricePredictionModel

logger = logging.getLogger('MarketAnalyzer')

class MarketAnalyzer:
    def __init__(self, model_path="models/price_predictor.pkl"):
        self.model_path = model_path
        self.model = PricePredictionModel(model_path=model_path)
        self.last_model_load_time = 0
        self.load_model()

    def load_model(self):
        """Load or reload the model if it exists."""
        try:
            if os.path.exists(self.model_path):
                mtime = os.path.getmtime(self.model_path)
                if mtime > self.last_model_load_time:
                    self.model.load_model()
                    self.last_model_load_time = mtime
                    logger.info(f"Model loaded/reloaded from {self.model_path}")
            else:
                logger.warning(f"Model not found at {self.model_path}")
        except Exception as e:
            logger.error(f"Error loading model: {e}")

    def analyze(self, df: pd.DataFrame):
        """
        Analyze market data:
        1. Calculate indicators/features
        2. Generate prediction
        3. Return structured analysis
        """
        # Ensure model is up to date
        self.load_model()

        if df is None or df.empty:
            return None

        # Calculate features
        # dropna=False to keep the last row even if some rolling windows aren't full (though features might be 0)
        # But for prediction we usually need full features. 
        # Let's use dropna=False and handle NaNs carefully or rely on build_feature_frame's logic.
        # The original API used dropna=False.
        df_features = build_feature_frame(df, dropna=False)
        
        if df_features.empty:
            return None

        latest = df_features.iloc[-1]
        
        # Extract features for model
        features = self.model.extract_features(latest.to_dict())
        
        # Predict
        prediction_class = self.model.predict(features)
        probabilities = self.model.predict_proba(features)
        
        # Map prediction
        direction_map = {0: 'DOWN', 1: 'NEUTRAL', 2: 'UP'}
        direction = direction_map.get(prediction_class, 'NEUTRAL')
        confidence = float(max(probabilities) * 100)
        
        # Trend Analysis
        close = float(latest['close'])
        ma20 = float(latest.get('ma_20', close))
        ma50_raw = df_features['close'].rolling(window=50).mean().iloc[-1] if len(df_features) >= 50 else close
        ma50 = float(ma50_raw) if not np.isnan(ma50_raw) else close
        
        trend = 'BULLISH' if ma20 > ma50 else 'BEARISH'
        adx = float(latest.get('adx', 0))
        
        # Target Price
        if direction == 'UP':
            target_price = close * 1.002
        elif direction == 'DOWN':
            target_price = close * 0.998
        else:
            target_price = close

        return {
            'timestamp': latest['timestamp'],
            'current_price': close,
            'prediction': {
                'direction': direction,
                'confidence': confidence,
                'probabilities': {
                    'down': float(probabilities[0] * 100),
                    'neutral': float(probabilities[1] * 100),
                    'up': float(probabilities[2] * 100)
                }
            },
            'trend': {
                'direction': trend,
                'adx': adx,
                'strength': 'Strong' if adx > 25 else 'Moderate' if adx > 15 else 'Weak'
            },
            'target_price': target_price,
            'indicators': {
                'rsi': float(latest.get('rsi', 50)),
                'macd': float(latest.get('macd', 0)),
                'ma_20': ma20,
                'ma_50': ma50
            }
        }
