import torch

import torch.nn as nn
import torch.nn.functional as F

class VOLNet(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(VOLNet, self).__init__()
        self.flow_model = None
        self.vit = None
        self.rotation_head = RotationHead(input_size, hidden_size, output_size)
        self.translation_head = TranslationHead(input_size, hidden_size, output_size)

    def forward(self, x):
        img1 = x["img1"]
        img2 = x["img2"]

        lidar1 = x["lidar1"]
        lidar2 = x["lidar2"]

        # Flow model
        flow = self.flow_model(img1, img2)

        # Append flow + lidar
        fused_data = fuseInputsBatch(lidar1, lidar2, flow)

        # Vision Transformer
        vit_output = self.vit(fused_data)

        # Output heads
        rotation = self.rotation_head(vit_output)
        translation = self.translation_head(vit_output)

        return rotation, translation

# Example usage
if __name__ == "__main__":
    net = VOLNet(input_size=768, hidden_size=512, output_size=3)
    dummy_data = {
        "img1": torch.randn(8, 3, 224, 224),
        "img2": torch.randn(8, 3, 224, 224),
        "lidar1": torch.randn(8, 3, 1024),
        "lidar2": torch.randn(8, 3, 1024)
    }
    rotation, translation = net(dummy_data)
    print(f"Rotation output shape: {rotation.shape}")  # Should output (8, 3)
    print(f"Translation output shape: {translation.shape}")  # Should output (8, 3)