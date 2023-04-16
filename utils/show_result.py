import os
import numpy as np
import matplotlib.pyplot as plt
import time
import torch
from .meshgen import meshgen

def plot_map(mapping):
    """
    Inputs:
        mapping: (2, h, w)
    """
    x = mapping[0].reshape((-1, 1))
    y = mapping[1].reshape((-1, 1))
    plt.plot(x, y, 'r.')
    plt.show()
    
def print_error(pred, Ax, Ay):
    """
    Inputs:
        pred: (1, 2, h, w)
        Ax, Ay: (h*w, h*w)
    """
    pred = pred[0]
    x_pred = pred[0].reshape((-1, 1))
    x = Ax.dot(x_pred)
    y_pred = pred[1].reshape((-1, 1))
    y = Ay.dot(y_pred)
    print(np.mean(np.abs(x)) + np.mean(np.abs(y)))

def print_mu_error(pred_map, org_mu):
    """
    Inputs:
        pred_map: (1, 2, h, w), numpy array
        org_mu: (1, 2, h, w), numpy array
    """
    h, w = pred_map.shape[2:]
    pred_mu = bc_metric(torch.from_numpy(pred_map))
    pred_mu = pred_mu[0].numpy().reshape((2, h-1, (w-1)*2))
    pred_mu = pred_mu[0] + pred_mu[1] * 1j
    mu = org_mu[0][:, :-1, :-1].numpy()
    mu = mu[0] + 1j * mu[1]
    mu_reshape = np.zeros((mu.shape[0], mu.shape[1]*2), dtype = np.complex)
    mu_reshape[:, ::2] = mu
    mu_reshape[:-1, 1:-1:2] = (mu[:-1, :-1] + mu[:-1, 1:] + mu[1:, :-1]) / 3
    mu_reshape[:-1, -1] = (mu[:-1, -1] + mu[1:, -1]) / 2
    mu_reshape[-1, 1:-1:2] = (mu[-1, :-1] + mu[-1, 1:]) / 2
    mu_reshape[-1, -1] = mu[-1, -1]
    mu = mu_reshape
    err = np.abs(mu - pred_mu).reshape(-1)
    print('mu_err:', np.mean(err))

def plot_pred_map(pred, mapping, save=False):
    """
    Inputs:
        pred, mapping: (2, h, w)
        save: True or False
    """
    #x_pred = pred[0].reshape((-1, 1))
    #y_pred = pred[1].reshape((-1, 1))
    #x_true = mapping[0].reshape((-1, 1))
    #y_true = mapping[1].reshape((-1, 1))

    #fig = plt.figure()
    #fig.add_subplot(1, 2, 1)
    #plt.plot(x_pred, y_pred, 'r.', markersize=1)
    #plt.title('Network output')
    #fig.add_subplot(1, 2, 2)
    #plt.plot(x_true, y_true, 'r.', markersize=1)
    #plt.title('Target image')

    h, w = pred.shape[1], pred.shape[2]

    x_pred = pred[0]
    y_pred = pred[1]
    x_true = mapping[0]
    y_true = mapping[1]

    fig = plt.figure()

    fig.add_subplot(1, 2, 1)
    for i in range(0, h, 2):
        plt.plot(x_pred[i, :].reshape(-1), y_pred[i, :].reshape(-1), 'b-', linewidth=0.5)
    for i in range(0, w, 2):
        plt.plot(x_pred[:, i].reshape(-1), y_pred[:, i].reshape(-1), 'b-', linewidth=0.5)
    plt.axis('off')
    plt.title('Network output')

    fig.add_subplot(1, 2, 2)
    for i in range(0, h, 2):
        plt.plot(x_true[i, :].reshape(-1), y_true[i, :].reshape(-1), 'b-', linewidth=0.5)
    for i in range(0, w, 2):
        plt.plot(x_true[:, i].reshape(-1), y_true[:, i].reshape(-1), 'b-', linewidth=0.5)
    plt.axis('off')
    plt.title('Target image')

    if save:
        if not os.path.exists('save_plot'):
            os.mkdir('save_plot')
        plt.savefig('./save_plot/'+str(time.time())+'.png',dpi=1200, bbox_inches = 'tight')
    else:
        plt.show()
    #plt.savefig("")

def triplot_pred_map(pred, mapping, save=False):
    """
    Inputs:
        pred, mapping: (2, h, w)
        save: True or False
    """
    h, w = pred.shape[1:]
    face, _ = meshgen(h, w)
    x_pred = pred[0].reshape(-1)#.reshape((-1, 1))
    y_pred = pred[1].reshape(-1)#.reshape((-1, 1))
    x_true = mapping[0].reshape(-1)#.reshape((-1, 1))
    y_true = mapping[1].reshape(-1)#.reshape((-1, 1))

    fig = plt.figure()
    fig.add_subplot(1, 2, 1)
    plt.triplot(x_pred, y_pred, face, 'r.-')#, markersize=3)
    plt.title('Network output')
    fig.add_subplot(1, 2, 2)
    plt.triplot(x_true, y_true, face, 'r.-')#, markersize=3)
    plt.title('Target image')
    if save:
        if not os.path.exists('save_plot'):
            os.mkdir('save_plot')
        plt.savefig('./save_plot/'+str(time.time())+'.png',dpi=1200, bbox_inches = 'tight')
    else:
        plt.show()
    #plt.savefig("")
if __name__ == '__main__':
    pass
