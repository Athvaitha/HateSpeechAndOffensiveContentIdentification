import torch

from text_encoder import TextEncoder

model = TextEncoder()

input_ids = torch.randint(
    0,
    1000,
    (2,64)
)

attention = torch.ones((2,64))

output = model(input_ids,attention)

print(output.shape)