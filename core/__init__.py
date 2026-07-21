from .config import WideBandConfig
from .model import (
    WideBindStack, WideBindBlock, GroupedCognitiveMirror, GroupedMLP,
    PartitionedEmbedding, PartitionedHead,
    AdaptiveController, MirrorLRScheduler,
    dct_basis, zeckendorf_codes, sparse_block_codes,
    vsa_prefix_scan,
)
from .zeckendorf_readout import ZeckendorfReadout

CognitiveMirror = GroupedCognitiveMirror