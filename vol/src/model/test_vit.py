import torch
import torch.nn as nn
from tokenizers import PatchEmbedding, PositionalEncoding
from vit import TransformerEncoderBlock, TransformerDecoderBlock, VisionTransformer


# Test script
vit = VisionTransformer(
    img_size=224,
    patch_size=16,
    in_channels=3,
    embed_dim=768,
    num_heads=12,
    num_layers=12,
    mlp_dim=3072,
    num_classes=1000,
    dropout=0.1
)

dummy_data = torch.randn(8, 3, 224, 224)
output = vit(dummy_data)
print(f"Output shape: {output.shape}")  # Should output (8, 1000)