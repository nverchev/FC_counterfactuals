import torch
import torch.nn as nn


def reversed_zip(*args):
    return zip(*map(reversed, args))


def get_linear_layer(in_features, out_features, act=nn.ReLU(), drop_out=0.):
    return nn.Sequential(nn.Linear(in_features, out_features), act, nn.Dropout(drop_out))


class MLP(nn.Module):

    def __init__(self, x_dim, z_dim):
        super().__init__()
        self.h_dims = [x_dim, 512, 256, z_dim]
        self.encode = nn.ModuleList([])
        for in_features, out_features in zip(self.h_dims[:-2], self.h_dims[1:-1]):
            self.encode.append(get_linear_layer(in_features, out_features, drop_out=0.2))
        self.encode.append(nn.Linear(self.h_dims[-2], self.h_dims[-1]))
        self.classifier = nn.Sequential(nn.ReLU(), nn.Linear(self.h_dims[-1], 1))
        self.settings = dict(hidden_layers=self.h_dims)

    def forward(self, x):
        for encode in self.encode:
            x = encode(x)
        x = self.classifier(x)
        return {'logits': x}


class AE(MLP):

    def __init__(self, x_dim, z_dim, vae=False):
        super().__init__(x_dim, 2 * z_dim if vae else z_dim)
        self.h_dims[-1] = z_dim  # overwritten when calling the vae
        self.decode = nn.ModuleList([])
        for in_features, out_features in reversed_zip(self.h_dims[2:], self.h_dims[1:-1]):
            self.decode.append(get_linear_layer(in_features, out_features))

        self.decode.append(nn.Linear(self.h_dims[1], x_dim))
        self.settings = dict(hidden_layers=self.h_dims)

    def forward(self, x):
        data = self.encoder(x)
        return self.decoder(data)

    def encoder(self, x):
        for encode in self.encode:
            x = encode(x)
        return {'z': [x]}

    def decoder(self, data):
        x = data['z'][0]
        for decode in self.decode:
            x = decode(x)
        data['recon'] = x
        return data


class VAE(AE):

    def __init__(self, x_dim, z_dim):
        super().__init__(x_dim, z_dim, vae=True)

    def sampling(self, mu, log_var):
        std = torch.exp(0.5 * log_var)
        eps = torch.randn_like(std)
        z = eps.mul(std).add_(mu) if self.training else eps.mul(std / 3).add_(mu)
        return z

    def encoder(self, x):
        mu, log_var = self.encode(x).chunk(2, 1)
        return {'z': [self.sampling(mu, log_var)], 'mu': [mu], 'log_var': [log_var], 'prior_log_var': []}


class HVAE(VAE):

    def __init__(self, x_dim, z_dim, dim_condition=0):
        super().__init__(x_dim, z_dim)

        # decoder
        self.inference = nn.ModuleList([])
        self.generate = nn.ModuleList([])
        self.from_latent = nn.ModuleList([])
        for features in reversed(self.h_dims[1:-1]):
            self.from_latent.append(nn.Linear(features + z_dim + dim_condition, features))
            self.inference.append(nn.Linear(2 * features, 2 * z_dim))
            self.generate.append(nn.Linear(features + z_dim, 2 * z_dim))

    def encoder(self, x):
        data = {'hidden': [],
                'mu': [],
                'log_var': [],
                'z': []}

        for encode in self.encode:
            x = encode(x)
            data['hidden'].append(x)

        mu, log_var = data['hidden'].pop().chunk(2, 1)
        z = self.sampling(mu, log_var)

        data['mu'].append(mu)
        data['log_var'].append(log_var)
        data['z'].append(z)
        return data

    def decoder(self, data, sample=None, s=1):
        data['prior_mu'] = []
        data['prior_log_var'] = []
        z = data['z'].pop() if sample is None else sample
        x = z
        for decode, from_latent, generate, inference in \
                zip(self.decode, self.from_latent, self.generate, self.inference):
            x = decode(x)
            p_mu, p_logvar = generate(torch.cat([x, z], dim=1)).chunk(2, 1)
            data['prior_mu'].append(p_mu)
            data['prior_log_var'].append(p_logvar)
            if sample is None:
                h = data['hidden'].pop()
                h = torch.cat([x, h], dim=1)
                mu, log_var = inference(h).chunk(2, 1)
                z = self.sampling(s * mu + p_mu, s * log_var + p_logvar)
                data['mu'].append(mu)
                data['log_var'].append(log_var)
            else:
                z = self.sampling(p_mu, p_logvar)
            x = from_latent(torch.cat([x, z], dim=1))
            s *= s
        data['recon'] = self.decode[-1](x)
        return data


class VAEX(VAE):

    def __init__(self, x_dim, z_dim):
        super().__init__(x_dim, z_dim)

    def forward(self, x, condition=None):
        inference_data = self.encoder(x, condition)
        return self.decoder(inference_data, condition)

    def sampling(self, mu, log_var):
        std = torch.exp(0.5 * log_var)
        eps = torch.randn_like(std)
        z = eps.mul(std).add_(mu) if self.training else eps.mul(std / 3).add_(mu)
        return z

    def encoder(self, x, condition=None):
        return self.encode(x)

    def decoder(self, data, condition=None):
        z = data['z']
        data['recon'] = self.decode(z)
        return data


def get_model(name, res, z_dim, **other_settings):
    # num_feature_above_diagonal:
    x_dim = res * (res - 1) // 2

    model_settings = dict(
        x_dim=x_dim,
        z_dim=z_dim,
    )

    model_mapping = {
        'AE': AE,
        'MLP': MLP,
        'Counterfactual VAE': HVAE
    }

    return model_mapping[name](**model_settings)
