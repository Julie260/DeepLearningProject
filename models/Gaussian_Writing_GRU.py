from torch import nn
import torch


class Gaussian_Writing_GRU(nn.Module):
    def __init__(self, n_gaussian=20, dropout=0, rnn_size=256, rnn_layers=2, input_size=3):
        super(Gaussian_Writing_GRU, self).__init__()
        self.n_gaussian = n_gaussian
        self.rnn_size = rnn_size
        if (input_size == 3):
            self.n_output = 1 + n_gaussian * 6
        else:
            self.n_output = 2 + n_gaussian * 6
        self.rnn = nn.GRU(input_size, self.rnn_size, rnn_layers, dropout=dropout)
        self.linear = nn.Linear(self.rnn_size, self.n_output)
        self.input_size = input_size

    def forward(self, input, hidden=None):
        output, hidden = self.rnn(input, hidden)
        output = output.view(-1, self.rnn_size)
        output = self.linear(output)
        if (self.input_size == 3):
            mu1, mu2, log_sigma1, log_sigma2, rho, pi_logits, z0_logits = output.split(self.n_gaussian, dim=1)
            d0_logits = torch.tensor([])
        else:
            mu1, mu2, log_sigma1, log_sigma2, rho, pi_logits, logits= output.split(self.n_gaussian, dim=1)
            z0_logits, d0_logits = logits.split(1, dim=1)
        rho = torch.tanh(rho)
        return mu1, mu2, log_sigma1, log_sigma2, rho, pi_logits, z0_logits, d0_logits, hidden
