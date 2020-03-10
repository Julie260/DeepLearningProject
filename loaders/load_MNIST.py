import numpy as np
import os
from matplotlib import pyplot as plt
from scipy import interpolate as itp
import torch
from torch.autograd import Variable
from random import randint

def sample_xml(batch_size=1):
    file=[]
    f2 = open("C:\\Users\\Julie\\Documents\\Polytechnique\\3A\\DeepLearning\\sequences\\trainlabels.txt", "r")
    labels = f2.readlines()
    for k in range (batch_size):
        file.append(randint(1, 59999))
    res = []
    for i in file:
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

def plot_points(data):
    plt.figure(figsize=[16,4])
    plt.gca().invert_yaxis()
    plt.axis('equal')
    pts = np.array(data).cumsum(axis=0)
    data[-1][-1] = 1
    idx = [i for i, v in enumerate(data) if data[i][-1]==1]
    start = 0
    for end in idx:
        tmp = pts[start:end+1]
        plt.plot(tmp[:,0], tmp[:,1], linewidth=2)
        start = end+1
    plt.show()

def plot_points_interpolate(data):
    plt.figure(figsize=[16,4])
    plt.gca().invert_yaxis()
    plt.axis('equal')
    pts = np.array(data).cumsum(axis=0)
    data[-1][-1] = 1
    idx = [i for i, v in enumerate(data) if data[i][-1]==1]
    print(idx)
    start = 0
    for end in idx:
        tmp = pts[start:end+1]
        print(tmp[:,0])
        print(tmp[:,1])
        xp=tmp[:,0]
        yp=tmp[:,1]
        okay = np.where(np.abs(np.diff(xp)) + np.abs(np.diff(yp)) > 0)
        xp = np.r_[xp[okay], xp[-1], xp[0]]
        yp = np.r_[yp[okay], yp[-1], yp[0]]
        try :
            mytck, myu = itp.splprep([xp, yp])
            xnew,ynew= itp.splev(np.linspace(0,1,1000),mytck)
            plt.plot(xnew,ynew)
        except TypeError :
            print("not enough points to interpolate...")
        start = end+1
    plt.show()


def batch_generator(seq_size=100, batch_size=50):
    cache = []
    data_size = seq_size * batch_size
    while True:
        if len(cache) < data_size:
            cache += sample_xml(30)
        else:
            x = torch.Tensor(cache[:data_size]) \
                .view(seq_size, batch_size, 5) \
                .transpose(0, 1) \
                .contiguous()
            cache = cache[data_size:]
            yield Variable(x)
#  #test
# f2 = open("C:\\Users\\Julie\\Documents\\Polytechnique\\3A\\DeepLearning\\sequences\\trainlabels.txt", "r")
# labels = f2.readlines()
# for i in range(15):
#     digit= parse_numbers(i, labels)
#     if digit != None and len(digit)>=4 :
#         plot_points(digit)
#         plot_points_interpolate(digit)
