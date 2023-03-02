import torch
import torch.nn as nn


def reversed_zip(*args):
    return zip(*map(reversed, args))


class MLP(nn.Module):

    def __init__(self, x_dim, z_dim):
        super().__init__()
        layers = [x_dim, 512, 256, z_dim]
        hidden_layers = []
        for in_features, out_features in zip(layers[:-1], layers[1:]):
            hidden_layers.append(nn.Linear(in_features, out_features))
            hidden_layers.append(nn.ReLU())
            hidden_layers.append(nn.Dropout(0.2))
        self.encode = nn.Sequential(*hidden_layers)
        self.classifier = nn.Linear(layers[-1], 1)
        self.settings = dict(hidden_layers=layers)

    def forward(self, x):
        x = self.encode(x)
        x = self.classifier(x)
        return {'logits': x}


class AE(nn.Module):

    def __init__(self, x_dim, z_dim, vae=False):
        super().__init__()
        layers = [x_dim, 512, 256, z_dim]
        hidden_layers = []
        for in_features, out_features in zip(layers[:-1], layers[1:-1]):
            hidden_layers.append(nn.Linear(in_features, out_features))
            hidden_layers.append(nn.ReLU())
            hidden_layers.append(nn.Dropout(0.2))
        self.encode = nn.Sequential(*hidden_layers, nn.Linear(layers[-2], 2 * z_dim if vae else z_dim))

        hidden_layers = []
        for in_features, out_features in reversed_zip(layers[1:], layers[1:-1]):
            hidden_layers.append(nn.Linear(in_features, out_features))
            hidden_layers.append(nn.ReLU())

        self.decode = nn.Sequential(*hidden_layers, nn.Linear(layers[1], x_dim))
        self.settings = dict(hidden_layers=layers)

    def forward(self, x):
        data = self.encoder(x)
        return self.decoder(data)

    def encoder(self, x):
        return {'z': [self.encode(x)]}

    def decoder(self, data):
        z = data['z'][0]
        data['recon'] = self.decode(z)
        return data


class VAE(AE):

    def __init__(self, x_dim, z_dim):
        super().__init__(x_dim, z_dim, vae=True)

    def sampling(self, mu, log_var):
        std = torch.exp(0.5 * log_var)
        eps = torch.randn_like(std)
        z = eps.mul(std).add_(mu) if self.training else eps.mul(std / 3).add_(mu)
        return z

    def encoder(self, x, condition=None):
        mu, log_var = self.encode(x).chunk(2, 1)
        return {'z': [self.sampling(mu, log_var)], 'mu': [mu], 'log_var': [log_var], 'prior_log_var': []}



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
        'Counterfactual VAE': VAE
    }

    return model_mapping[name](**model_settings)
