import torch

def init_weights(m):
    if isinstance(m, torch.nn.Linear):
        torch.nn.init.xavier_uniform(m.weight),torch.nn.init.calculate_gain("leaky_relu")
        torch.nn.init.constant_(m.bias, 0)

# deep autoencoder - DAE
class DAE(torch.nn.Module):
    def __init__(self):
        super(DAE, self).__init__()
        self.dimensions = [784, 300, 500, 1000, 10]
        self.encoder = torch.nn.Sequential(
            torch.nn.Linear(784, 300),
            torch.nn.LeakyReLU(),
            torch.nn.Linear(300, 500),
            torch.nn.LeakyReLU(),
            torch.nn.Linear(500, 1000),
            torch.nn.LeakyReLU(),
            torch.nn.Linear(1000, 10),
        )

        self.decoder = torch.nn.Sequential(
            torch.nn.Linear(10, 1000),
            torch.nn.LeakyReLU(),
            torch.nn.Linear(1000, 500),
            torch.nn.LeakyReLU(),
            torch.nn.Linear(500, 300),
            torch.nn.LeakyReLU(),
            torch.nn.Linear(300, 784),
        )

    def forward(self, x):
        z = self.encoder(x)
        return self.decoder(z)
