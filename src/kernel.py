import torch
import torch.nn as nn


class Thinplate(nn.Module):

    def __init__(self, state_dim, hidden_dim):
        super().__init__()

        self.centers = nn.Parameter(torch.zeros(hidden_dim - state_dim, state_dim), requires_grad=True) # 
        self.K = hidden_dim - state_dim

    def forward(self, x):
        
        x_lifted = []
        for i in range(self.K):
            r_square = torch.sum( (x - self.centers[i]) ** 2, dim=-1, keepdims=True)
            r_square = torch.clip(r_square, 1e-6)

            y = torch.sqrt(r_square)
            x_lifted.append(y)
        
        x_lifted = torch.cat(x_lifted, dim=-1)

        return x_lifted


class Gauss(nn.Module):

    def __init__(self, state_dim, hidden_dim):
        super().__init__()

        self.centers = nn.Parameter(torch.zeros(hidden_dim - state_dim, state_dim), requires_grad=True) # 
        self.K = hidden_dim - state_dim

    def forward(self, x):
        
        x_lifted = []
        for i in range(self.K):
            r_square = torch.mean( (x - self.centers[i]) ** 2, dim=-1, keepdims=True)

            y = torch.exp(r_square / 1000)
            x_lifted.append(y)
        
        x_lifted = torch.cat(x_lifted, dim=-1)

        return x_lifted


class Invquad(nn.Module):

    def __init__(self, state_dim, hidden_dim):
        super().__init__()

        self.centers = nn.Parameter(torch.zeros(hidden_dim - state_dim, state_dim), requires_grad=True) # 
        self.K = hidden_dim - state_dim

    def forward(self, x):
        
        x_lifted = []
        for i in range(self.K):
            r_square = torch.sum( (x - self.centers[i]) ** 2, dim=-1, keepdims=True)

            y = 1 / (1 + r_square)
            x_lifted.append(y)
        
        x_lifted = torch.cat(x_lifted, dim=-1)

        return x_lifted


class Invmultquad(nn.Module):

    def __init__(self, state_dim, hidden_dim):
        super().__init__()

        self.centers = nn.Parameter(torch.zeros(hidden_dim - state_dim, state_dim), requires_grad=True) # 
        self.K = hidden_dim - state_dim

    def forward(self, x):
        
        x_lifted = []
        for i in range(self.K):
            r_square = torch.sum( (x - self.centers[i]) ** 2, dim=-1, keepdims=True)

            y = 1 / torch.sqrt(1 + r_square)
            x_lifted.append(y)
        
        x_lifted = torch.cat(x_lifted, dim=-1)

        return x_lifted


class Polyharmonic(nn.Module):

    def __init__(self, state_dim, hidden_dim):
        super().__init__()

        self.centers = nn.Parameter(torch.zeros(hidden_dim - state_dim, state_dim), requires_grad=True) # 
        self.K = hidden_dim - state_dim

    def forward(self, x):
        
        x_lifted = []
        for i in range(self.K):
            r_square = torch.sum( (x - self.centers[i]) ** 2, dim=-1, keepdims=True)
            r_square = torch.clip(r_square, 1e-6)

            y = torch.sqrt(r_square) * torch.log(torch.sqrt(r_square))
            x_lifted.append(y)
        
        x_lifted = torch.cat(x_lifted, dim=-1)

        return x_lifted


if __name__ == "__main__":

    model = Thinplate(6, 16)
    x = torch.randn(512, 100, 6)
    y = model(x)
    print(y.shape, torch.isnan(y).any())

    model = Gauss(6, 16)
    x = torch.randn(512, 100, 6)
    y = model(x)
    print(y.shape, torch.isnan(y).any())

    model = Invquad(6, 16)
    x = torch.randn(512, 100, 6)
    y = model(x)
    print(y.shape, torch.isnan(y).any())

    model = Invmultquad(6, 16)
    x = torch.randn(512, 100, 6)
    y = model(x)
    print(y.shape, torch.isnan(y).any())

    model = Polyharmonic(6, 16)
    x = torch.randn(512, 100, 6)
    y = model(x)
    print(y.shape, torch.isnan(y).any())