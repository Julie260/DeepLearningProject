import torch
from models import model
from toolbox import generate_samples, draw_samples
from torch.autograd import Variable

file_name = 'Gaussian_Writing_LSTMepoch_2000.pt'
path = 'C://Users//Julie//Documents//GitHub//DL-test//trained_models//'

model_name = 'Gaussian_Writing_LSTM'
parameters = {'n_gaussian': 20, 'dropout': 0.2, 'rnn_size': 256}
net = model.get_model(model_name, parameters)

net.load_state_dict(torch.load(path+file_name))

x0 = Variable(torch.Tensor([0,0,1]).view(1,1,3))
data = generate_samples.generate(net, x0, n=500)
draw_samples.plot_points(data)