import torch
from models import model
from toolbox import generate_samples, draw_samples
from torch.autograd import Variable

path = 'C://Users//Julie//Documents//GitHub//DeepLearningProject//train_split_MNIST_models//'

model_name = 'Gaussian_Writing_LSTM'
parameters = {'n_gaussian': 20, 'dropout': 0.2, 'rnn_size': 256, 'rnn_layers': 2, "input_size": 5}


for i in range(10):
    file_name = 'LSTM_label='+str(i)+'.pt'
    net = model.get_model(model_name, parameters)
    net.load_state_dict(torch.load(path + file_name))
    x0 = Variable(torch.Tensor([0,0,0,0,i]).view(1,1,5))
    data = generate_samples.generate(net, x0, n=50)
    draw_samples.plot_points(data)