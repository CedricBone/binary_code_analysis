"""
Data processing and generation for binary analysis.
"""

from .preprocessing import BinaryPreprocessor
from .generator import SyntheticDataGenerator
from .real_binary_loader import RealBinaryLoader
from .cross_architecture import CrossArchitectureProcessor

__all__ = [
    'BinaryPreprocessor', 
    'SyntheticDataGenerator',
    'RealBinaryLoader',
    'CrossArchitectureProcessor'
]