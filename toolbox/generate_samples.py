from torch import nn
import random
import torch


def draw_one_sample(model, mu1, mu2, log_sigma1, log_sigma2, rho, pi_logits, z0_logits, d0_logits, label):
    # draw Gaussian mixture
    pi = nn.functional.softmax(pi_logits)
    idx, = random.choices(range(model.n_gaussian), weights=pi.data.tolist()[0])
    sigma1, sigma2 = log_sigma1.exp(), log_sigma2.exp()
    x1 = torch.normal(mu1[:, idx], sigma1[:, idx])  # size = 1,
    mu2 = mu2 + rho * (log_sigma2 - log_sigma1).exp() * (x1 - mu1)
    sigma2 = (1 - rho ** 2) ** 0.5 * sigma2
    x2 = torch.normal(mu2[:, idx], sigma2[:, idx])  # $\Delta
    p_bernoulli = torch.sigmoid(z0_logits)
    eos = torch.bernoulli(p_bernoulli).view(1)
    if(model.input_size == 3):
        return torch.cat([x1, x2, eos]).view(1, 1, 3)
    else:
        p_bernoulli2 = torch.sigmoid(d0_logits)
        eod = torch.bernoulli(p_bernoulli2).view(1)
        return torch.cat([x1, x2, eos, eod, torch.tensor([label])]).view(1, 1, 5)


def generate(model, x0, hidden=None, n=100):
    res = []
    sample = x0
    label = x0.data.tolist()[0][0][-1]
    print(label)
    for i in range(n):
        mu1, mu2, log_sigma1, log_sigma2, rho, pi_logits, z0_logits, d0_logits, hidden = model.forward(sample, hidden)
        sample = draw_one_sample(model, mu1, mu2, log_sigma1, log_sigma2, rho, pi_logits, z0_logits, d0_logits, label)
        res.append(sample.data.tolist()[0][0])
    return res
