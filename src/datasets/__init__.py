"""Dataset loaders and utilities"""
from .redd_loader import REDDLoader
from .real_datasets import REDDDataset, UKDALEDataset, DatasetRegistry
from .refit_dataset import REFITDataset
from .ampds_dataset import AMPdsDataset
from .eco_dataset import ECODataset
from .more_datasets import (
    GREENDDataset,
    PLAIDDataset,
    BLUEDDataset,
    iAWEDataset,
    DREDDataset
)

__all__ = [
    # Synthetic data
    'REDDLoader',

    # Real datasets - Low/Medium frequency (good for disaggregation)
    'REDDDataset',        # 6 homes, USA
    'UKDALEDataset',      # 5 homes, UK
    'REFITDataset',       # 20 homes, UK
    'AMPdsDataset',       # 1 home, Canada
    'ECODataset',         # 6 homes, Switzerland
    'GREENDDataset',      # 9 homes, Italy/Austria
    'iAWEDataset',        # 1 home, India
    'DREDDataset',        # 1 home, Netherlands

    # High-frequency datasets (event detection/identification)
    'PLAIDDataset',       # Appliance signatures, 30kHz
    'BLUEDDataset',       # Event detection, 12kHz

    # Utilities
    'DatasetRegistry'
]
