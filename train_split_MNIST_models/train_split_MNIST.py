from models import model
from loaders import loader
from toolbox import losses

import torch
from torch import optim

model_name = 'Gaussian_Writing_LSTM'
parameters = {'n_gaussian': 20, 'dropout': 0.2, 'rnn_size': 256, 'rnn_layers':2, 'input_size':5}

data = "split_MNIST"

#path to save the model
dir = "C://Users//Julie//Documents//GitHub//DeepLearningProject//train_split_MNIST_models//"

max_epoch = 600
info_freq = 10
max_norm = 10  # for gradient clipping

for i in range(10):
    net = model.get_model(model_name, parameters)
    optimizer = optim.Adam(net.parameters(), lr=0.005)
    batch = loader.get_loader(data, i)
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
    file = dir+"LSTM_label="+str(i)+".pt"
    torch.save(net.state_dict(), file)