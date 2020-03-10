import os
import torch
from torch.autograd import Variable
from random import randint

def sample_xml(label, batch_size=1):
    file=[]
    f2 = open("C:\\Users\\Julie\\Documents\\Polytechnique\\3A\\DeepLearning\\sequences\\trainlabels.txt", "r")
    labels = f2.readlines()
    for k in range (batch_size):
        file.append(randint(1, 59999))
    res = []
    for i in file:
        if (int(labels[i]) == int(label)):
            tab =parse_numbers(i, labels)
            for pts in tab:
                res.append(pts)
    return res

def parse_numbers(i, labels) :
    n = 60000
    tab=[]
    label = int(labels[i])
    filename = "C:\\Users\\Julie\\Documents\\Polytechnique\\3A\\DeepLearning\\sequences\\trainimg-"+str(i)+"-inputdata.txt"
    if os.path.exists(filename):
        f = open(filename, "r")
        #tab.append([100,0,1,1,label])
        for line in f:
            if(len(tab) == 0):
                tab.append([0, 0, 0, 0, label])
            else:
                tmp = line.split(' ')
                delta_x = int(tmp[0])
                delta_y = int(tmp[1])
                eos = int(tmp[2])
                eod = int(tmp[3])
                tab_intermediaire = [delta_x, delta_y, eos, eod, label]
                tab.append(tab_intermediaire)
        f.close()
        if(len(tab)<100):
            tab.append([0,0,0,1,label])
        return tab


def batch_generator(label, seq_size=100, batch_size=50):
    cache = []
    data_size = seq_size * batch_size
    while True:
        if len(cache) < data_size:
            cache += sample_xml(label, 30)
        else:
            x = torch.Tensor(cache[:data_size]) \
                .view(seq_size, batch_size, 5) \
                .transpose(0, 1) \
                .contiguous()
            cache = cache[data_size:]
            yield Variable(x)