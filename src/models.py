import torch
import torch.nn as nn
from abc import abstractmethod
from src.layers import FromLatentConv, ResBlock, View, ConvBlock, ToLatentConvADAIN
from src.utils import reversed_zip, ABCHookAfterInit


class MLP(nn.Module):

    def __init__(self, x_dim=98346):
        super().__init__()
        layers = [x_dim, 512, 256, 32]
        hidden_layers = []
        for in_features, out_features in zip(layers[:-1], layers[1:]):
            hidden_layers.append(nn.Linear(in_features, out_features))
            hidden_layers.append(nn.ReLU())
        self.encode = nn.Sequential(*hidden_layers)
        self.classifier = nn.Linear(layers[-1], 1)
        self.settings = dict(hidden_layers=layers)

    def forward(self, x):
        x = self.encode(x)
        x = self.classifier(x)
        return {'logits': x}


class AE(nn.Module):

    def __init__(self, x_dim=98346, z_dim=32):
        super().__init__()
        layers = [x_dim, 512, 256, z_dim]
        hidden_layers = []
        for in_features, out_features in zip(layers[:-1], layers[1:-1]):
            hidden_layers.append(nn.Linear(in_features, out_features))
            hidden_layers.append(nn.ReLU())
        self.encode = nn.Sequential(*hidden_layers, nn.Linear(layers[-2], z_dim))

        hidden_layers = []
        rev_layers = layers[::-1]
        for in_features, out_features in zip(rev_layers[:-1], rev_layers[1:-1]):
            hidden_layers.append(nn.Linear(in_features, out_features))
            hidden_layers.append(nn.ReLU())

        self.decode = nn.Sequential(*hidden_layers, nn.Linear(rev_layers[-2], x_dim))
        self.settings = dict(hidden_layers=layers)

    def forward(self, x):
        data = {'z': self.encoder(x)}
        return self.decoder(data)

    def encoder(self, x):
        return self.encode(x)

    def decoder(self, data):
        z = data['z']
        data['recon'] = self.decode(z)
        return data


class AbstractVAE(nn.Module, metaclass=ABCHookAfterInit):
    in_channels = 2
    out_channel = 1
    dim_target = 0

    def __init__(self, **settings):
        super().__init__()
        # Dimensions
        self.x_dim = settings['x_dim']
        self.h_dims = settings['h_dims']
        self.k_dims = settings['k_dims']
        self.strides = settings['strides']
        self.paddings = settings['paddings']
        self.z_dims = settings['z_dims']
        self.depths = settings['depths']
        self.l_dims_encoder, self.l_dims_decoder, self.final_z_res = self.size_calc()
        self.print_z_dims()
        self.print_sizes()
        self.hook_after_init = self.print_total_parameters

    def forward(self, x, condition=None, slice_n=None):

        # Rescaling condition between -1 and 1 to have 0 as neutral value
        condition = 2 * condition.view(-1, 1, 1, 1) - 1 if condition is not None else 0

        # Rescaling slice
        slice_n = (slice_n / 100 - 1).view(-1, 1, 1, 1) if slice_n is not None else 0

        inference_data = self.encoder(x, condition, slice_n)
        return self.decoder(inference_data, condition, slice_n)

    def sampling(self, mu, log_var):
        std = torch.exp(0.5 * log_var)
        eps = torch.randn_like(std)
        z = eps.mul(std).add_(mu) if self.training else eps.mul(std / 3).add_(mu)
        return z

    @abstractmethod
    def encoder(self, x, condition, slice_n):
        pass

    @abstractmethod
    def decoder(self, x, condition, slice_n):
        pass

    def print_total_parameters(self):
        num_params = 0
        for param in self.parameters():
            num_params += param.numel()
        print('Total Parameters: {}'.format(num_params))
        return

    def print_z_dims(self):
        print('Latent dimensions:', end=' ')
        total = 0
        for z_dim, l_dim in zip(self.z_dims, self.l_dims_encoder[1:]):
            new_dims = z_dim * l_dim ** 2
            print(new_dims, end=' + ')
            total += new_dims
        new_dims = self.z_dims[-1] * self.final_z_res ** 2
        total += new_dims
        print(new_dims, '=', total)
        return

    def size_calc(self):
        l_dims = self.x_dim
        l_dims_encoder = [l_dims]
        for k_dim, stride, padding in zip(self.k_dims, self.strides, self.paddings):
            l_dims = self.conv_size_calc(L_in=l_dims, kernel_size=k_dim, stride=stride, padding=padding)
            l_dims_encoder.append(l_dims)
        final_z_res = self.conv_size_calc(L_in=l_dims, kernel_size=3, stride=2, padding=1)
        l_dims_decoder = [l_dims]
        for k_dim, stride, padding in reversed_zip(self.k_dims, self.strides, self.paddings):
            l_dims = self.transposed_conv_size_calc(L_in=l_dims, kernel_size=k_dim, stride=stride, padding=padding)
            l_dims_decoder.append(l_dims)
        assert l_dims == self.x_dim
        return l_dims_encoder, l_dims_decoder, final_z_res

    def print_sizes(self):
        print('Block Sizes:')
        for i, (h, l) in enumerate(zip(self.h_dims, self.l_dims_encoder)):
            print('Block {} size: None x {} x {} x {}'.format(i + 1, h, l, l))

        for i, (h, l) in enumerate(zip(self.h_dims[::-1], self.l_dims_decoder)):
            print('Block {} size: None x {} x {} x {}'.format(len(self.h_dims) + 1 + i, h, l, l))

    @staticmethod
    def conv_size_calc(L_in, kernel_size, stride=1, dilation=1, padding=0):
        return int((L_in + 2 * padding - dilation * (kernel_size - 1) - 1) // stride) + 1

    @staticmethod
    def transposed_conv_size_calc(L_in, kernel_size, stride=1, padding=0, dilation=1, output_padding=0):
        return (L_in - 1) * stride - 2 * padding + dilation * (kernel_size - 1) + output_padding + 1


class VAEX(AbstractVAE):

    def __new__(cls, **settings):
        self = super().__new__(cls)
        self.dim_target = 1
        self.in_channels += self.dim_target
        return self

    def __init__(self, **settings):
        super().__init__(**settings)  # calculates final layer.

        # settings will be saved
        *c_z_dims, final_z_dim = self.z_dims
        interpolate = settings['interpolate']
        self.settings = settings

        # encoder
        self.conv_init = nn.Conv2d(self.in_channels, self.h_dims[0], 1, bias=False)
        self.encode = nn.ModuleList([])
        *resolutions, final_res = self.l_dims_encoder
        for i, (k_dim, stride, padding, z_dim, depth) in enumerate(
                zip(self.k_dims, self.strides, self.paddings, c_z_dims, self.depths)):
            encode = ResBlock(self.h_dims[i], self.h_dims[i + 1], k_dim, stride, padding, deconv=False,
                              depth=depth, interpolate=interpolate)
            self.encode.append(encode)

        self.final_inference = ToLatentConvADAIN(self.h_dims[-1], final_z_dim, final_res, 0)
        self.h = torch.nn.parameter.Parameter(torch.ones(1, self.h_dims[-1], final_res, final_res))
        self.back_to_img = FromLatentConv(self.h_dims[-1], final_z_dim + self.dim_target, final_res, drop_layer_prob=0)

        # decoder
        aux_dim = final_z_dim
        self.inference = nn.ModuleList([])
        self.decode = nn.ModuleList([])
        self.generate = nn.ModuleList([])
        self.from_latent = nn.ModuleList([])
        self.merge = nn.ModuleList([])
        for i, (k_dim, stride, padding, z_dim, res, depth) in enumerate(
                reversed_zip(self.k_dims, self.strides, self.paddings, c_z_dims, resolutions, self.depths)):
            decode = ResBlock(self.h_dims[-i - 1], self.h_dims[-i - 2], k_dim, stride, padding, deconv=True,
                              depth=depth, interpolate=interpolate)
            from_latent = FromLatentConv(self.h_dims[-i - 2], z_dim + self.dim_target, res) if z_dim else None
            generate = ToLatentConvADAIN(self.h_dims[-i - 2], z_dim, res, aux_dim) if z_dim else None
            merge = ConvBlock(2 * self.h_dims[-i - 2], self.h_dims[-i - 2]) if z_dim else None
            inference = ToLatentConvADAIN(self.h_dims[-i - 2], z_dim, res) if z_dim else None
            self.decode.append(decode)
            self.inference.append(inference)
            self.generate.append(generate)
            self.from_latent.append(from_latent)
            self.merge.append(merge)
            aux_dim = z_dim or 0
        self.conv_final = ConvBlock(self.h_dims[0], self.out_channel)

    def encoder(self, x, condition, slice_n: torch.Tensor):
        data = {'hidden': [],
                'mu': [],
                'log_var': [],
                'z': []}
        ones = torch.ones(x.shape[0], 1, self.x_dim, self.x_dim, device=x.device)
        x = torch.cat([x, slice_n * ones, condition * ones], dim=1)

        x = self.conv_init(x)
        for encode in self.encode:
            data['hidden'].append(x)
            x = encode(x)

        mu, log_var = self.final_inference(x)
        z = self.sampling(mu, log_var)

        data['mu'].append(mu)
        data['log_var'].append(log_var)
        data['z'].append(z)
        return data

    def decoder(self, data, condition, slice_n: torch.Tensor, sample=None, s=1):
        data['prior_mu'] = []
        data['prior_log_var'] = []
        data['condition'] = condition
        z = data['z'].pop() if sample is None else sample
        z = z.view(-1, self.z_dims[-1], self.final_z_res, self.final_z_res)
        h = self.h.expand(z.size()[0], -1, -1, -1)
        h = slice_n * h if slice_n is not None else h
        x = self.back_to_img(z, h, condition)
        for decode, from_latent, generate, inference, merge in zip(self.decode, self.from_latent, self.generate,
                                                                   self.inference, self.merge):
            x = decode(x)
            h = data['hidden'].pop() if sample is None else None
            data['hidden'].insert(0, h)
            if from_latent is not None:
                p_mu, p_logvar = generate(x, z)
                data['prior_mu'].append(p_mu)
                data['prior_log_var'].append(p_logvar)
                if sample is None:
                    h = merge(torch.cat([x, h], dim=1))
                    mu, log_var = inference(h)
                    z = self.sampling(s * mu + p_mu, s * log_var + p_logvar)
                    data['mu'].append(mu)
                    data['log_var'].append(log_var)
                else:
                    z = self.sampling(p_mu, p_logvar)
                x = from_latent(z, x, condition)
                s *= s
        data['recon'] = torch.sigmoid(self.conv_final(x))
        return data


def get_mlp(**model_settings):
    return MLP()


def get_vae(**model_settings):
    return VAEX(**model_settings)


def get_ae(**model_settings):
    return AE()
