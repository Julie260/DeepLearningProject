import torch
from torch import nn
import math


def calculate_loss(model, x, xNext, data, hidden=None):
  batch_size = x.size(1)
  mu1, mu2, log_sigma1, log_sigma2, rho, pi_logits, z0_logits, d0_logits, hidden = model.forward(x, hidden)
  if data == "IAM":
    xNext = xNext.view(-1,3)
    x1, x2, x_eos = xNext.split(1,dim=1)
  if (data == "MNIST" or data == "split_MNIST"):
    xNext = xNext.view(-1,5)
    x1, x2, x_eos, x_eod, label = xNext.split(1,dim=1)
  loss1 = - logP_gaussian(model, x1, x2, mu1, mu2, log_sigma1, log_sigma2, rho, pi_logits)
  loss2 = nn.functional.binary_cross_entropy_with_logits(z0_logits,x_eos,size_average=False)
  if data == "IAM":
    loss = (loss1 + loss2)/batch_size
  if data == "MNIST" or data == "split_MNIST":
    loss3 = nn.functional.binary_cross_entropy_with_logits(d0_logits,x_eod,size_average=False)
    loss = (loss1 + 0.5*loss2 + 0.5*loss3)/batch_size # average over mini-batch
  return loss, hidden

def logsumexp(x):
  x_max, _ = x.max(dim=1,keepdim=True)
  x_max_expand = x_max.expand(x.size())
  res =  x_max + torch.log((x-x_max_expand).exp().sum(dim=1, keepdim=True))
  return res

def logP_gaussian(model,x1, x2, mu1, mu2, log_sigma1, log_sigma2, rho, pi_logits):
  x1, x2 = x1.repeat(1,model.n_gaussian), x2.repeat(1,model.n_gaussian)
  sigma1, sigma2 = log_sigma1.exp(), log_sigma2.exp()
  log_pi = nn.functional.log_softmax(pi_logits)
  z_tmp1, z_tmp2 = (x1-mu1)/sigma1, (x2-mu2)/sigma2
  z = z_tmp1**2 + z_tmp2**2 - 2*rho*z_tmp1*z_tmp2
  # part one
  log_gaussian = - math.log(math.pi*2)-log_sigma1 - log_sigma2 - 0.5*(1-rho**2).log()
  # part two
  log_gaussian += - z/2/(1-rho**2)
  # part three
  log_gaussian = logsumexp(log_gaussian + log_pi)
  return log_gaussian.sum()