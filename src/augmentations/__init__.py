try:
    from src.augmentations.augmix import (
        AugMixDataset,
        AugMixGTSRB,
        jensen_shannon_loss,
    )
except ImportError:
    AugMixDataset = None
    AugMixGTSRB = None
    jensen_shannon_loss = None

try:
    from src.augmentations.frequency import (
        low_frequency_component,
        high_frequency_component,
        frequency_dropout,
        frequency_blend,
        FrequencyAugDataset,
    )
except ImportError:
    low_frequency_component = None
    high_frequency_component = None
    frequency_dropout = None
    frequency_blend = None
    FrequencyAugDataset = None

__all__ = [
    "AugMixDataset",
    "AugMixGTSRB",
    "jensen_shannon_loss",
    "low_frequency_component",
    "high_frequency_component",
    "frequency_dropout",
    "frequency_blend",
    "FrequencyAugDataset",
]
