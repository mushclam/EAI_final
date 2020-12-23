import torch
import torch.nn as nn
import torch.nn.functional as F

class CustomizedModel(nn.Module):
    def __init__(self, mask):
        super(CustomizedModel, self).__init__()

        if isinstance(mask, torch.Tensor):
            self.mask = mask.t()
        else:
            if mask is not None:
                self.mask = torch.tensor(mask, dtype=torch.float).t()
            else:
                self.mask = None

        # self.flatten = nn.Flatten()
        # self.linear = nn.Linear(784, 1000)
        # self.sigmoid = nn.Sigmoid()

        self.layer = nn.Sequential(
            nn.Flatten(),
            nn.Linear(784, 1000),
            nn.Sigmoid()
        )

        self.maskedLinear = nn.Linear(1000, 10, bias=False)
        self.maskedLinear.mask_params = nn.Parameter(torch.Tensor(10, 1000))
        self.maskedLinear.original_weight = nn.Parameter(torch.Tensor(10, 1000))
        # if self.mask is not None:
        #     self.maskedLinear.mask_params = nn.Parameter(self.mask)
        #     self.maskedLinear.original_weight = self.maskedLinear.weight
        #     # del self.maskedLinear.weight
        #     def repopulate_weight(maskedLinear, _):
        #         maskedLinear.weight.data = nn.Parameter(maskedLinear.original_weight.data * maskedLinear.mask_params.data)
        #     self.maskedLinear.register_forward_pre_hook(repopulate_weight)

        # if mask is not None:
        #     self.maskedLinear.weight.data = self.maskedLinear.weight.data * self.mask

    def forward(self, x):
        # x = self.flatten(x)
        # x = self.linear(x)
        # x = self.sigmoid(x)
        x = self.layer(x)
        x = self.maskedLinear(x)

        return x

    def setMask(self):
        if self.mask is not None:
            self.maskedLinear.mask_params = nn.Parameter(self.mask)
            self.maskedLinear.original_weight = self.maskedLinear.weight
            # del self.maskedLinear.weight
            def repopulate_weight(maskedLinear, _):
                maskedLinear.weight.data = nn.Parameter(
                    maskedLinear.original_weight.data * maskedLinear.mask_params.data)
            self.maskedLinear.register_forward_pre_hook(repopulate_weight)