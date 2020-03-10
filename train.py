from models import model
from loaders import loader
from toolbox import losses

import torch
from torch import optim

#model name can be either Gaussian_Writing_LSTM or Gaussian_Writing_GRU
model_name = 'Gaussian_Writing_LSTM'
#parameter to create the model
parameters = {'n_gaussian': 20, 'dropout': 0.2, 'rnn_size': 256, 'rnn_layers':2, 'input_size':3}
net = model.get_model(model_name, parameters)

#data can be either MNIST or IAM
# be careful the choice of the data impact the input size you give to your net
#for IAM the input size is 3 (dx, dy, eos)
#for MNIST the input size is 5 (dx, dy, eos, eod, label)
data = "IAM"

#path to save the model
dir = "C://Users//Julie//Documents//GitHub//DeepLearningProject//trained_models//Gaussian_Writing_GRUIAM_epoch_700_1_0,005_Adam_256_2.pt"

optimizer = optim.Adam(net.parameters(), lr=0.005)
max_epoch = 700
info_freq = 10
max_norm = 10  # for gradient clipping

batch = loader.get_loader(data)
loss_log = []
hidden = None
for epoch in range(max_epoch):
    x = next(batch)
    optimizer.zero_grad()
    x1 = x[:-1]
    x2 = x[1:]
    loss, hidden = losses.calculate_loss(net, x1, x2, data, hidden)
    loss.backward()
    torch.nn.utils.clip_grad_norm_(net.parameters(), max_norm)
    optimizer.step()
    if model_name == 'Gaussian_Writing_GRU':
        hidden.detach_()
    if model_name == 'Gaussian_Writing_LSTM':
        hidden[0].detach_()
        hidden[1].detach_()
    loss_log += [loss.item()]
    if epoch % info_freq == 0:
        print('epoch {}, loss {}'.format(epoch, loss.item()))

torch.save(net.state_dict(), dir)