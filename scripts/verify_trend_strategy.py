
import sys
import os
import unittest
from unittest.mock import MagicMock, patch
import pandas as pd
import numpy as np
from datetime import datetime

# Add project root to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from strategy.ml_trend_following_strategy import MLTrendFollowingStrategy

class TestMLTrendFollowingStrategy(unittest.TestCase):
    def setUp(self):
        self.risk_limits = MagicMock()
        
        # Mock ML models
        # Only PricePredictionModel is used now
        with patch('strategy.ml_trend_following_strategy.PricePredictionModel') as MockPP:
            self.strategy = MLTrendFollowingStrategy('AAPL', self.risk_limits)
            
            # Setup mocks
            self.strategy.price_predictor = MagicMock()
            
            # Mock feature extraction to return dummy data
            self.strategy.get_price_prediction_features = MagicMock(return_value=[1, 2, 3])

    def test_should_enter_bullish_trend(self):
        # Case 1: Bullish Trend, Price Predictor UP -> SHOULD ENTER
        self.strategy.price_predictor.predict.return_value = 2 # UP
        
        should_enter = self.strategy.should_enter(trend_direction=1)
        self.assertTrue(should_enter, "Should enter when price predictor confirms bullish trend")
        
        # Case 2: Bullish Trend, Price Predictor NEUTRAL -> SHOULD ENTER
        self.strategy.price_predictor.predict.return_value = 1 # NEUTRAL
        should_enter = self.strategy.should_enter(trend_direction=1)
        self.assertTrue(should_enter, "Should enter when price predictor is neutral")
        
        # Case 3: Bullish Trend, Price Predictor DOWN -> SHOULD NOT ENTER
        self.strategy.price_predictor.predict.return_value = 0 # DOWN
        should_enter = self.strategy.should_enter(trend_direction=1)
        self.assertFalse(should_enter, "Should NOT enter when price predictor contradicts trend")

    def test_should_enter_bearish_trend(self):
        # Case 1: Bearish Trend, Price Predictor DOWN -> SHOULD ENTER
        self.strategy.price_predictor.predict.return_value = 0 # DOWN
        
        should_enter = self.strategy.should_enter(trend_direction=-1)
        self.assertTrue(should_enter, "Should enter when price predictor confirms bearish trend")
        
        # Case 2: Bearish Trend, Price Predictor UP -> SHOULD NOT ENTER
        self.strategy.price_predictor.predict.return_value = 2 # UP
        should_enter = self.strategy.should_enter(trend_direction=-1)
        self.assertFalse(should_enter, "Should NOT enter when price predictor contradicts trend")

if __name__ == '__main__':
    unittest.main()
