import torch

from resnet_encoder import ResNetEncoder

model = ResNetEncoder()

dummy = torch.randn(2,3,224,224)

output = model(dummy)

print(output.shape)