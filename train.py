import argparse
import logging
import os
import sys

import torch
import torch.nn as nn
from torch import optim
from tqdm import tqdm
import torch.nn.functional as F
from torch.utils.tensorboard import SummaryWriter
from torch.utils.data import DataLoader

from model.bsnet import BSNet
from data import QCM_Gen
from utils.loss import BSLossFunc, detDFunc
from utils.qc import mu2A_torch


data_path = '../imagenet.pkl'
#data_path = '../clefi.pkl'
size = [112, 112]
dir_checkpoint = 'checkpoints/'
def train_net(net,
              device,
              epochs=5,
              batch_size=1,
              weight = 0,
              lr=0.001,
              save_interval=5,
              save_cp=True):

    dataset = QCM_Gen(data_path, size)
    n_train = len(dataset)
    train_loader = DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=2, pin_memory=True, drop_last=True)

    writer = SummaryWriter(comment='LR_{}_BS_{}'.format(lr, batch_size))
    global_step = 0

    logging.info('''Starting training:
        Epochs:          {}
        Batch size:      {}
        Learning rate:   {}
        Checkpoints:     {}
        Device:          {}
        Weight:          {}
    '''.format(epochs, batch_size, lr, save_cp, device.type, weight))

    optimizer = optim.RMSprop(net.parameters(), lr=lr, weight_decay=1e-8, momentum=0.5)
    #scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min' if net.n_classes > 1 else 'max', patience=2)
    lambda1 = lambda epoch: 0.95 ** (epoch/80.)
    scheduler = optim.lr_scheduler.LambdaLR(optimizer, lambda1)
    criterion1 = BSLossFunc()
    criterion2 = detDFunc(device, size)
    mu2A = mu2A_torch(*size, device, batch_size)

    for epoch in range(epochs):
        net.train()

        epoch_loss = 0
        sample_cnt = 0
        det_sum = 0
        with tqdm(total=n_train, desc='Epoch {}/{}'.format(epoch+1, epochs), unit='imgs') as pbar:
            for batch in train_loader:
                mu = batch
                h, w = mu.shape[2], mu.shape[3]

                assert mu.shape[1] == 2, 'Network has been defined with {} input channels, but loaded images have {} channels. Please check that the images are loaded correctly.'.format(net.n_channels, mu.shape[1])

                mu = mu.to(device=device, dtype=torch.float32)
                A, vmu_weight = mu2A(mu)

                map_pred = net(mu)

                loss1 = criterion1(map_pred, A)
                detDloss = criterion2(map_pred)
                det_sum += detDloss.item()

                loss = loss1 #+ 0.1 * detDloss
                epoch_loss += loss1.item()# + 0.1 * detDloss.item()

                writer.add_scalar('Loss/Beltrami_loss', loss1.item(), global_step)
                writer.add_scalar('Loss/det_loss', detDloss.item(), global_step)

                sample_cnt += 1
                pbar.set_postfix(**{'loss(batch)': loss.item(), 'loss1(batch)': loss1.item(), 'avg det': det_sum / sample_cnt, 'epoch avg loss:': epoch_loss / sample_cnt})

                optimizer.zero_grad()
                loss.backward()
                #for name, param in net.named_parameters():
                #    print(sample_cnt, name, torch.isfinite(param.grad).all())
                #for name, param in criterion2.named_parameters():
                #    print(sample_cnt, name, torch.isfinite(param.grad).all())
                nn.utils.clip_grad_value_(net.parameters(), 0.1)
                optimizer.step()

                pbar.update(batch_size)
                global_step += 1
                #if global_step % (n_train // (10 * batch_size)) == 0:
                #    for tag, value in net.named_parameters():
                #        tag = tag.replace('.', '/')
                #        writer.add_histogram('weights/' + tag, value.data.cpu().numpy(), global_step)
                #        writer.add_histogram('grads/' + tag, value.grad.data.cpu().numpy(), global_step)

                #    # scheduler.step(val_score)
                #    # writer.add_scalar('learning_rate', optimizer.param_groups[0]['lr'], global_step)

                #    #writer.add_images('images', mu, global_step)
        if save_cp:
            try:
                os.mkdir(dir_checkpoint)
                logging.info('Created checkpoint directory')
            except OSError:
                pass
            if (epoch+1) % save_interval == 0 or epoch == 0:
                torch.save(net.state_dict(),
                           dir_checkpoint + 'bsnet{}.pth'.format(epoch + 1))
                logging.info('Checkpoint {} saved !'.format(epoch + 1))

    writer.close()


def get_args():
    parser = argparse.ArgumentParser(description='Train the BSNet on images and target masks',
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('-e', '--epochs', metavar='E', type=int, default=4000,
                        help='Number of epochs', dest='epochs')
    parser.add_argument('-b', '--batch-size', metavar='B', type=int, nargs='?', default=10,
                        help='Batch size', dest='batchsize')
    parser.add_argument('-l', '--learning-rate', metavar='LR', type=float, nargs='?', default=0.0002,
                        help='Learning rate', dest='lr')
    parser.add_argument('-f', '--load', dest='load', type=str, default=False,
                        help='Load model from a .pth file')
    parser.add_argument('-w', '--weight', dest='weight', type=float, default=1,
                        help='The weight of the custom loss')

    return parser.parse_args()


if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
    args = get_args()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    torch.backends.cudnn.benchmark = True
    logging.info('Using device {}'.format(device))

    net = BSNet(size, n_channels=2, n_classes=2, bilinear=True)
    #logging.info('Network:\n\t{} input channels\n\t{} output channels (classes)\n\tnet.bilinear = {} upscaling'.format(net.n_channels, net.n_classes, net.bilinear))

    if args.load:
        net.load_state_dict(torch.load(args.load, map_location=device))
        logging.info('Model loaded from {}'.format(args.load))

    net.to(device=device)

    try:
        train_net(net=net,
                  epochs=args.epochs,
                  weight = args.weight,
                  batch_size=args.batchsize,
                  lr=args.lr,
                  device=device)
    except KeyboardInterrupt:
        torch.save(net.state_dict(), 'INTERRUPTED.pth')
        logging.info('Saved interrupt')
        try:
            sys.exit(0)
        except SystemExit:
            os._exit(0)
