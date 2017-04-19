import argparse

import chainer
import chainer.links as L
from chainer import training
from chainer.training import extensions

from net import SiameseNetwork

import random
import numpy as np

def create_pairs(data, digit_indices):
    x0_data = []
    x1_data = []
    label = []

    n = min([len(digit_indices[d]) for d in range(10)]) - 1
    for d in range(10):
        for i in range(n):  # 各数字でn個のペアを作る
            # 同じクラスのペアを作る
            # ラベルは1
            z1, z2 = digit_indices[d][i], digit_indices[d][i+1]
            x0_data.append(data[z1])
            x1_data.append(data[z2])
            label.append(1)

            # 異なるクラスのペアを作る
            # 足すのが最低値が1なので同じクラスにはならない
            # ラベルは0
            inc = random.randrange(1, 10)
            dn = (d + inc) % 10
            z1, z2 = digit_indices[d][i], digit_indices[dn][i]
            x0_data.append(data[z1])
            x1_data.append(data[z2])
            label.append(0)

    x0_data = np.array(x0_data)
    x1_data = np.array(x1_data)
    label = np.array(label, dtype=np.int32)

    return x0_data, x1_data, label


def create_iterator(datasets, batchsize, test=False):
    data, label = datasets._datasets[0], datasets._datasets[1]
    digit_indices = [np.where(label == i)[0] for i in range(10)]
    x0_data, x1_data, label = create_pairs(data, digit_indices)

    dd = chainer.datasets.DictDataset(x0_data=x0_data, x1_data=x1_data, label=label)
    if test:
        data_iter = chainer.iterators.SerialIterator(dd, batchsize, repeat=False)
    else:
        data_iter = chainer.iterators.SerialIterator(dd, batchsize)

    return data_iter


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--gpu', '-g', type=int, default=-1,
                        help='GPU ID (negative value indicates CPU)')
    parser.add_argument('--epoch', '-e', type=int, default=32,
                        help='Number of sweeps over the dataset to train')
    parser.add_argument('--batchsize', '-b', type=int, default=128,
                        help='Number of images in each mini-batch')
    parser.add_argument('--out', '-o', default='result',
                        help='Directory to output the result')
    args = parser.parse_args()

    # create pair dataset iterator
    train, test = chainer.datasets.get_mnist(ndim=3)
    train_iter = create_iterator(train, args.batchsize)
    test_iter = create_iterator(test, args.batchsize, test=True)

    # create siamese network and train it
    model = SiameseNetwork()
    if args.gpu >= 0:
        chainer.cuda.get_device(args.gpu).use()
        model.to_gpu()

    optimizer = chainer.optimizers.MomentumSGD()
    optimizer.setup(model)

    updater = training.StandardUpdater(train_iter, optimizer, device=args.gpu)
    trainer = training.Trainer(updater, (args.epoch, 'epoch'), out=args.out)

    trainer.extend(extensions.Evaluator(test_iter, model, device=args.gpu))
    trainer.extend(extensions.snapshot_object(model, 'model.npz'), trigger=(args.epoch, 'epoch'))
    trainer.extend(extensions.LogReport())
    trainer.extend(extensions.PrintReport(
        ['epoch', 'main/loss', 'validation/main/loss', 'elapsed_time']
    ))
    trainer.extend(extensions.ProgressBar())

    trainer.run()


if __name__ == '__main__':
    main()
