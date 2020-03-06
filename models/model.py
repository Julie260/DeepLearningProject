from models.Gaussian_Writing_GRU import Gaussian_Writing_GRU
from models.Gaussian_Writing_LSTM import Gaussian_Writing_LSTM
from models.Gaussian_Writing_GRU_LSTM import Gaussian_Writing_GRU_LSTM


def get_model(name, parameters):
    if name == 'Gaussian_Writing_GRU':
        model = Gaussian_Writing_GRU(parameters["n_gaussian"], parameters["dropout"], parameters["rnn_size"], parameters["rnn_layers"], parameters["input_size"])
    elif name == 'Gaussian_Writing_LSTM':
        model = Gaussian_Writing_LSTM(parameters["n_gaussian"], parameters["dropout"], parameters["rnn_size"], parameters["rnn_layers"], parmaeters["input_size"])
    else:
        print('Model {} not available')
    return model
