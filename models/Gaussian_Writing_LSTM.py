from torch import nn
import torch


class Gaussian_Writing_LSTM(nn.Module):
    def __init__(self, n_gaussian=20, dropout=0, rnn_size=256):
        super(Gaussian_Writing_LSTM, self).__init__()
        self.n_gaussian = n_gaussian
        self.rnn_size = rnn_size
        self.n_output = 1 + n_gaussian * 6
        self.rnn = nn.LSTM(3, self.rnn_size, 2, dropout=dropout)
        self.linear = nn.Linear(self.rnn_size, self.n_output)

    def forward(self, input, hidden=None):
        output, hidden = self.rnn(input, hidden)
        output = output.view(-1, self.rnn_size)
        output = self.linear(output)
        mu1, mu2, log_sigma1, log_sigma2, rho, pi_logits, z0_logits = \
            output.split(self.n_gaussian, dim=1)
        rho = torch.tanh(rho)
        return mu1, mu2, log_sigma1, log_sigma2, rho, pi_logits, z0_logits, hidden
