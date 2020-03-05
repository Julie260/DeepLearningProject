from toolbox import draw_samples
from bs4 import BeautifulSoup as Soup
import numpy as np
import glob
import torch
from torch.autograd import Variable

def extract_points(filename):
    with open(filename, 'r') as f:
        soup = Soup(f.read(), 'lxml')
    pts = [[float(pt['x']), float(pt['y'])] for pt in soup.find_all('point')]
    pen_lifts = []
    for stroke in soup.find_all('stroke'):
        pen_lifts += [0] * (len(stroke.find_all('point')) - 1) + [1]
    return pts, pen_lifts


path = 'C:\\Users\\Julie\\Documents\\Polytechnique\\3A\\DeepLearning\\original\\**\\*.xml'
all_files = list(glob.iglob(path, recursive=True))


def sample_xml(batch_size=1, factor=20):
    pts = []
    pen_lifts = []
    for filename in np.random.choice(all_files, batch_size, replace=False):
        _pts, _pen_lifts = extract_points(filename)
        _pts, _pen_lifts = np.array(_pts), np.array(_pen_lifts)
        _pts, _pen_lifts = _pts - np.roll(_pts, 1, axis=0), _pen_lifts
        _pts[0][0] = 100
        _pts[0][1] = 0
        _pen_lifts[0] = 1
        _pts = np.minimum(np.maximum(_pts, -500), 500) / factor
        pts += _pts.tolist()
        pen_lifts += _pen_lifts.tolist()
    res = [v1 + [v2, ] for v1, v2 in zip(pts, pen_lifts)]
    return res


def batch_generator(seq_size=300, batch_size=50):
    cache = []
    data_size = seq_size * batch_size
    while True:
        if len(cache) < data_size:
            cache += sample_xml(1000)
        else:
            x = torch.Tensor(cache[:data_size]) \
                .view(batch_size, seq_size, 3) \
                .transpose(0, 1) \
                .contiguous()
            cache = cache[data_size:]
            yield Variable(x)

def plot_IAM_samples(data):
    draw_samples.plot_points(data[:1000])
