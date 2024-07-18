from torchvision.models import efficientnet_v2_m, EfficientNet_V2_M_Weights, ResNet101_Weights

from torchvision import transforms

import torch

# weights = EfficientNet_V2_M_Weights.IMAGENET1K_V1
I_model = efficientnet_v2_m(weights=weights, progress=True).cuda()
I_model.train()

x = torch.randn(16, 3, 480, 480).cuda()

out = I_model(x)

print(out.shape)