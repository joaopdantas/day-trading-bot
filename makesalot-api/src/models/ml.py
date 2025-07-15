"""
Machine Learning Models Module for Day Trading Bot.

This module implements various ML models for price prediction and pattern recognition.
This is now a compatibility layer that imports from the modular structure.
"""

import logging
import warnings

# Import classes from modular structure
from .builder import ModelBuilder
from .trainer import ModelTrainer
from .optimizer import ModelOptimizer
from .prediction import PredictionModel
from .ensemble import EnsembleModel

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Show deprecation warning
warnings.warn(
    "Importing directly from ml.py is deprecated. "
    "Please import from src.models instead (e.g., from src.models import PredictionModel).",
    DeprecationWarning,
    stacklevel=2
)

# Re-export all classes to maintain backward compatibility
__all__ = [
    'ModelBuilder',
    'ModelTrainer', 
    'ModelOptimizer',
    'PredictionModel',
    'EnsembleModel'
]