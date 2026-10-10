# Models package
try:
    from .baseline_cnn import BaselineResNet18, get_parameter_groups
    from .cbm import ConceptBottleneckModel, CBMLoss, get_concept_lookup_tensors
    from .hybrid_loss import HybridCBMLoss, compute_jsd
except ImportError:
    pass  # torch not available (local diagnostic environment)


from .concept_encoder import (
    BOTTLENECK_DIM,
    SHAPE_VOCAB, COLOUR_VOCAB, BORDER_VOCAB, SYMCAT_VOCAB,
    encode_all_classes,
    get_concept_tensor_for_class,
    get_oracle_concept_matrix,
    decode_concept_indices,
    load_class_concepts,
)

