from enum import Enum
from typing import List, Optional, Dict
from pydantic import BaseModel


class AnalystType(str, Enum):
    MARKET = "market"
    SOCIAL = "social"
    NEWS = "news"
    FUNDAMENTALS = "fundamentals"
    # Crypto-specific analysts
    ONCHAIN = "onchain"
    PREDICTION = "prediction"
    CRYPTO_SENTIMENT = "crypto_sentiment"
