from models import model
from loaders import loader
from toolbox import losses

import torch
from torch import optim

model_name = 'Gaussian_Writing_GRU'
parameters = {'n_gaussian': 20, 'dropout': 0.2, 'rnn_size': 256, 'rnn_layers':3, 'input_size':3}
net = model.get_model(model_name, parameters)

data = "IAM"

#path to save the model
dir = "C://Users//Julie//Documents//GitHub//DL-test//trained_models//"+model_name+"epoch_2000.pt"

optimizer = optim.Adam(net.parameters(), lr=0.005)
max_epoch = 2000
info_freq = 10
max_norm = 10  # for gradient clipping

batch = loader.get_loader(data)
loss_log = []
hidden = None
for epoch in range(max_epoch):
    x = next(batch)
    optimizer.zero_grad()
    loss, hidden = losses.calculate_loss(net, x[:-1], x[1:], hidden)
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