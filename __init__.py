"""
Questrade API Client

Simple Python client for the Questrade trading API.

Usage:
    from questrade import Questrade
    
    qt = Questrade(refresh_token="YOUR_TOKEN")  # First time
    qt = Questrade()  # Uses saved token after first use
    
    qt.print_accounts()
"""

from .client import Questrade, QuestradeError, AuthenticationError

__version__ = "2.0.0"
__all__ = ["Questrade", "QuestradeError", "AuthenticationError"]
