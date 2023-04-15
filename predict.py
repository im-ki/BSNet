import argparse
import logging
import os
import time

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from torch.utils.data import DataLoader

from model.bsnet import BSNet
from data import QCM_Gen
from utils.show_result import plot_pred_map, triplot_pred_map, print_mu_error, print_error
from utils.qc_cpu import mu2map

def get_args():
    parser = argparse.ArgumentParser(description='Predict masks from input images',
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--model', '-f', default='MODEL.pth',
                        metavar='FILE',
                        help="Specify the file in which the model is stored")
    parser.add_argument('--input', '-i', metavar='INPUT', default = '../imagenet.pkl',
                        help='filenames of input images')

    return parser.parse_args()

if __name__ == "__main__":
    args = get_args()
    data_path = args.input

    size = [112, 112]

    dataset = QCM_Gen(data_path, size)
    test_loader = DataLoader(dataset, batch_size=1, shuffle=False, num_workers=1)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logging.info('Using device {}'.format(device))

    net = BSNet(size, n_channels=2, n_classes=2, bilinear = True)
    logging.info("Loading model {}".format(args.model))

    net.to(device=device)
    net.load_state_dict(torch.load(args.model, map_location=device))

    logging.info("Model loaded !")

    current = 0
    for batch in test_loader:
        current += 1

        mu = batch
        mu = mu.to(device=device, dtype=torch.float32)
        h, w = mu.shape[2], mu.shape[3]

        logging.info("\nPredicting image {}".format(current))

        net.eval()
        with torch.no_grad():
            pred_map = net(mu).cpu().numpy()

        mu = mu[0].cpu().numpy()
        plot_pred_map(pred_map[0], mu2map(mu[0] + 1j * mu[1])[0], save=False)
        #triplot_pred_map(pred, mapping, size[0], size[1], save=True)
