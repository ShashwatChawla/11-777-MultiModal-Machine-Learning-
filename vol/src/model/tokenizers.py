import torch
import torch.nn as nn
import math

class PatchEmbedding(nn.Module):
    '''
    Patch Embedding class to convert image into patches and then into embeddings
    '''
    def __init__(self, img_size=224, patch_size=16, in_channels=3, embed_dim=768):
        super(PatchEmbedding, self).__init__()
        self.img_size = img_size
        self.patch_size = patch_size
        self.n_patches = (img_size // patch_size) ** 2
        
        # Conv2d with patch size as kernel, and stride as patch size
        self.patch_embed = nn.Conv2d(in_channels, embed_dim, kernel_size=patch_size, stride=patch_size)

    def forward(self, x):
        # (batch_size, 3, img_size, img_size) -> (batch_size, embed_dim, n_patches ** 2)
        x = self.patch_embed(x)
        x = x.flatten(2)  # flatten to (batch_size, embed_dim, n_patches)
        x = x.transpose(1, 2)  # (batch_size, n_patches, embed_dim)
        return x


class PositionalEncoding(nn.Module):
    def __init__(self, embed_dim, n_patches, dropout=0.1):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)
        n_tokens = n_patches + 1  # Add 1 for the class token
        position = torch.arange(0, n_tokens, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, embed_dim, 2).float() * (-math.log(10000.0) / embed_dim))
        pe = torch.zeros(n_tokens, embed_dim)
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)  # Add batch dimension
        self.register_buffer('positional_encoding', pe)

    def forward(self, x):
        x = x + self.positional_encoding[:, :x.size(1), :]
        return x