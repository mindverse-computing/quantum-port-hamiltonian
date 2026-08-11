"""dissipative package."""
from .quantum_phnn import DynamicQpHNN, VectorFieldQpHNN
from .data_generator import (
    DampedHarmonicOscillator,
    VanDerPolOscillator,
    DissipativeVectorFieldDataset,
    TrajectoryDataset,
)
from .trainer import (
    train_dynamic_qphnn,
    train_vector_field_qphnn,
    DissipativeTrainingResult,
)

__all__ = [
    "DynamicQpHNN",
    "VectorFieldQpHNN",
    "DampedHarmonicOscillator",
    "VanDerPolOscillator",
    "DissipativeVectorFieldDataset",
    "TrajectoryDataset",
    "train_dynamic_qphnn",
    "train_vector_field_qphnn",
    "DissipativeTrainingResult",
]
