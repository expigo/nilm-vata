"""NILM Algorithms module"""
from .combinatorial_optimization import CombinatorialOptimization
from .fhmm import FHMM
from .baseline import MeanAlgorithm, ZeroAlgorithm, MedianAlgorithm
from .knn import KNNDisaggregator

# Deep learning algorithms (optional - require PyTorch)
DEEP_LEARNING_AVAILABLE = False
try:
    import torch
    # Only import if PyTorch is available
    from .seq2seq import Seq2SeqDisaggregator
    from .dae import DAEDisaggregator
    DEEP_LEARNING_AVAILABLE = True
except (ImportError, NameError):
    # PyTorch not available or import failed
    pass

__all__ = [
    'CombinatorialOptimization',
    'FHMM',
    'MeanAlgorithm',
    'ZeroAlgorithm',
    'MedianAlgorithm',
    'KNNDisaggregator',
]

if DEEP_LEARNING_AVAILABLE:
    __all__.extend(['Seq2SeqDisaggregator', 'DAEDisaggregator'])
