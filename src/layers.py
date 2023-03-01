import torch
import torch.nn as nn
import torch.nn.functional as F


# increased momentum
class WobblyBatchNorm(nn.Module):
    def __init__(self, h_in):
        super().__init__()
        self.bn = nn.BatchNorm2d(h_in, eps=1e-6, momentum=0.2)

    def forward(self, inputs):
        return self.bn(inputs)


Act = nn.Hardswish  # more efficient than Swish
Norm = WobblyBatchNorm


class View(nn.Module):

    def __init__(self, *shape):
        super().__init__()
        self.shape = shape

    def forward(self, inputs):
        return inputs.view(self.shape)


class SqueezeExcite(nn.Module):
    def __init__(self, h_dim, reduction_ratio=0.1):
        super().__init__()
        h_reduced = max(1, int(h_dim * reduction_ratio))
        self.fc_reduce = nn.Linear(h_dim, h_reduced)
        self.fc_augment = nn.Linear(h_reduced, h_dim)

    def forward(self, inputs):
        # global average pooling
        x = inputs.mean((2, 3))
        x = F.relu(self.fc_reduce(x))
        x = torch.sigmoid(self.fc_augment(x))
        x = x.unsqueeze(-1).unsqueeze(-1)
        return inputs.mul(x)


class ConvBlock(nn.Module):
    def __init__(self, h_in, h_out, deconv=False, kernel_size=3, stride=1, padding=1, wn=lambda x: x):
        super().__init__()
        self.stride = stride
        Conv = nn.ConvTranspose2d if deconv else nn.Conv2d
        self.block = nn.Sequential(Norm(h_in),
                                   Act(),
                                   wn(Conv(h_in, h_out, kernel_size=kernel_size, stride=stride, padding=padding,
                                           bias=False)),
                                   )
        nn.init.xavier_uniform_(self.block[2].weight)

    def forward(self, x):
        return self.block(x)


class ResLayer(nn.Module):
    def __init__(self, h_in):
        super().__init__()
        self.block = ConvBlock(h_in, h_in)
        self.se = SqueezeExcite(h_in)

    def forward(self, x):
        return x + self.se(self.block(x))


class ResBlock(nn.Module):
    def __init__(self, h_in, h_out, kernel_size, stride, padding, deconv=False, depth=0, interpolate=False):
        super().__init__()
        modules = []
        self.stride = stride
        self.deconv = deconv
        self.interpolate = interpolate
        self.h_in = h_in
        self.h_out = h_out
        if interpolate:
            self.se = SqueezeExcite(h_out)
        for _ in range(depth):
            modules.append(ResLayer(h_in))
        modules.append(ConvBlock(h_in, h_out, deconv=deconv, kernel_size=kernel_size, stride=stride, padding=padding))
        self.outer_block = nn.Sequential(*modules)

    def forward(self, x):
        y = self.outer_block(x)
        if not self.interpolate:
            return y
        output_size = y.size()
        if self.stride != 1:
            x = F.interpolate(x, size=output_size[2:], mode='bilinear', align_corners=False)
        if self.h_out < self.h_in:  # input channels are twice as much
            x = x.view(output_size[0], output_size[1], 2, output_size[2], output_size[3]).mean(2)
        elif self.h_out > self.h_in:  # output channels are twice as much
            x = x.repeat(1, 2, 1, 1)
        return x + self.se(y)


class FromLatentConv(nn.Module):
    def __init__(self, h_dim, z_dim, l_dim, drop_layer_prob=0.1):
        super().__init__()
        self.res = (l_dim + 1) // 2
        self.z_dim = z_dim
        self.to_image = nn.ConvTranspose2d(z_dim, h_dim, stride=2, kernel_size=3, padding=1)
        self.drop_layer_prob = drop_layer_prob
        # nn.init.xavier_uniform_(self.to_image[1].weight)

    def forward(self, z, x, condition=None):
        ones = torch.ones((z.shape[0], 1, self.res, self.res), device=z.device)
        condition = condition if condition is not None else 0
        z = z.view(z.shape[0], -1, self.res, self.res)
        z = torch.cat([z, condition * ones], dim=1)
        z = self.to_image(z)
        # if self.training and self.drop_layer_prob:
        #     drop = torch.rand(z.shape[0], device=z.device).view(-1, 1, 1, 1) > self.drop_layer_prob
        #     z = z * drop
        return x + z

# uses ADAIN
class ToLatentConvADAIN(nn.Module):
    def __init__(self, h_dim, z_dim, l_dim, aux_dim=0):
        super().__init__()
        if aux_dim:
            self.aux_conv = nn.Sequential(
                nn.Linear(aux_dim * ((l_dim + 3) // 4) ** 2, 2 * h_dim),
                View(-1, 2 * h_dim, 1, 1),
            )
        self.to_latent = ConvBlock(h_dim, 2 * z_dim, stride=2, wn=torch.nn.utils.weight_norm)

    def forward(self, x, aux=None):
        if aux is not None:
            mu, std = self.aux_conv(aux.flatten(1)).chunk(2, 1)
            x = F.instance_norm(x) * torch.sigmoid(std) + mu
        x = self.to_latent(x)
        return x.chunk(2, 1)
