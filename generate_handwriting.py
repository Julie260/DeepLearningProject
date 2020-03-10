import torch
from models import model
from toolbox import generate_samples, draw_samples
from torch.autograd import Variable

#name and path of the file where the model was saved
file_name = 'Gaussian_Writing_GRUIAM_epoch_700_1_0,005_Adam_256_2.pt'
path = 'C://Users//Julie//Documents//GitHub//DeepLearningProject//trained_models//'

#model name can be either Gaussian_Writing_LSTM or Gaussian_Writing_GRU
model_name = 'Gaussian_Writing_GRU'

#parameter to create the model
parameters = {'n_gaussian': 1, 'dropout': 0.2, 'rnn_size': 256, 'rnn_layers': 2, "input_size": 3}
net = model.get_model(model_name, parameters)
net.load_state_dict(torch.load(path+file_name))

for i in range(10):
    x0 = Variable(torch.Tensor([0,0,1]).view(1,1,3))
    data = generate_samples.generate(net, x0, n=60)
    draw_samples.plot_points(data)