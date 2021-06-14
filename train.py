#!/usr/bin/env python

import matplotlib as mpl
mpl.use('Agg')
import argparse
import random
import re
from pathlib import Path
import numpy as np

import chainer
from chainer import dataset
from chainer import training
from chainer.training import extensions
from visualize import out_generated_image, burn_in
# import chainerx

# import dali_util
from model import InvertibleGray


class PreprocessedDataset(chainer.dataset.DatasetMixin):
    def __init__(self, path, root, crop_size):
        self.base = chainer.datasets.ImageDataset(path, root)

        # self.base =
        self.gray_weight = np.array([[[0.299]], [[0.587]], [[0.114]]])
        self.crop_size = crop_size

    def __len__(self):
        return len(self.base)

    def get_example(self, i):
        crop_size = self.crop_size

        image = self.base[i]

        _, h, w = image.shape
        if h < 224:
            image = np.resize(image, (3, 224, w))
        if w < 224:
            image = np.resize(image, (3, h, 224))
        _, h, w = image.shape
        top = (h - crop_size) // 2
        left = (w - crop_size) // 2
        bottom = top + crop_size
        right = left + crop_size
        image = image[:, top:bottom, left:right]
        # image -= self.mean[:, top:bottom, left:right]

        image = image * (1.0 / 127.5) - 1 # [-1, 1]
        gray = (image * self.gray_weight).sum(0)
        return image, gray

def converter(x, device, padding=None):
    rgb_img = np.array([i[0] for i in x], dtype="f")
    gray_img = np.array([[i[1]] for i in x], dtype="f")
    # print(rgb_img.min(), rgb_img.max(), gray_img.min(), gray_img.max())
    input_imgages = [
        chainer.Variable(rgb_img),
        chainer.Variable(gray_img),
    ]
    if device >= 0:
        for i in input_imgages:
            i.to_gpu(device)
    return input_imgages


# def main():
if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Learning convnet from ILSVRC2012 dataset')
    # parser.add_argument('train', help='Path to training image-label list file')
    # parser.add_argument('val', help='Path to validation image-label list file')
    parser.add_argument('--seed', '-s', type=int, default=0,
                        help='Seed Number')
    parser.add_argument('--batchsize', '-B', type=int, default=8,
                        help='Learning minibatch size')
    parser.add_argument('--epoch', '-E', type=int, default=120,
                        help='Number of epochs to train')
    parser.add_argument('--device', '-d', type=str, default='0',
                        help='Device specifier. Either ChainerX device '
                        'specifier or an integer. If non-negative integer, '
                        'CuPy arrays with specified device id are used. If '
                        'negative integer, NumPy arrays are used')
    parser.add_argument('--initmodel',
                        help='Initialize the model from given file')
    parser.add_argument('--loaderjob', '-j', type=int,
                        help='Number of parallel data loading processes')
    parser.add_argument('--resume', '-r', default='',
                        help='Initialize the trainer from given file')
    parser.add_argument('--out', '-o', default='result',
                        help='Output directory')
    parser.add_argument('--root', '-R', default='.',
                        help='Root directory path of image files')
    parser.add_argument('--val_batchsize', '-b', type=int, default=8,
                        help='Validation minibatch size')
    parser.add_argument('--test', action='store_true')
    parser.set_defaults(test=False)
    parser.add_argument('--dali', action='store_true')
    parser.set_defaults(dali=False)
    parser.add_argument('--stage_two', action='store_true')
    parser.set_defaults(stage_two=False)
    group = parser.add_argument_group('deprecated arguments')
    group.add_argument('--gpu', '-g', type=int, nargs='?', const=0,
                        help='GPU ID (negative value indicates CPU)')
    args = parser.parse_args()

    # device = parse_device(args)
    device = int(args.device)
    np.random.seed(args.seed)
    random.seed(args.seed)
    print('Device: {}'.format(device))
    print('# Minibatch-size: {}'.format(args.batchsize))
    print('# epoch: {}'.format(args.epoch))
    print(f"# seed : {args.seed}")
    print(f"# stage_two : {args.stage_two}")
    print('')

    # Initialize the model to train
    model = InvertibleGray(args.stage_two)
    if args.initmodel:
        print('Load model from {}'.format(args.initmodel))
        chainer.serializers.load_npz(args.initmodel, model)
    model.to_gpu(device)

    # device.use()
    chainer.cuda.get_device_from_id(args.gpu).use()
    dataset_path = Path("./VOCdevkit/VOC2012/JPEGImages")
    images = sorted(dataset_path.iterdir())
    random.shuffle(images)
    train_img_path = images[:13758]
    val_img_path = images[13758:]
    train = PreprocessedDataset(train_img_path, ".", model.insize)
    val = PreprocessedDataset(val_img_path, ".", model.insize)

    # These iterators load the images with subprocesses running in parallel
    # to the training/validation.
    train_iter = chainer.iterators.MultiprocessIterator(
        train, args.batchsize, n_processes=args.loaderjob)
    val_iter = chainer.iterators.MultiprocessIterator(
        val, args.val_batchsize, repeat=False, n_processes=args.loaderjob)
    # converter = dataset.concat_examples
    test_batch = converter(chainer.iterators.SerialIterator(val, 9, repeat=False).next(), device)

    # Set up an optimizer
    # optimizer = chainer.optimizers.MomentumSGD(lr=0.01, momentum=0.9)
    optimizer = chainer.optimizers.Adam(alpha=0.0002)
    optimizer.setup(model)

    # Set up a trainer
    updater = training.updaters.StandardUpdater(
        train_iter, optimizer, converter=converter, device=device)
    trainer = training.Trainer(updater, (args.epoch, 'epoch'), args.out)

    val_interval = (1, "epoch")
    log_interval = (100, 'iteration')


    trainer.extend(extensions.Evaluator(val_iter, model, converter=converter,
                                        device=device), trigger=val_interval)
    # trainer.extend(extensions.snapshot(), trigger=val_interval)
    trainer.extend(extensions.snapshot_object(model, 'model_iter_{.updater.epoch}'), trigger=val_interval)
    trainer.extend(extensions.LogReport(trigger=log_interval))
    trainer.extend(extensions.observe_lr(), trigger=log_interval)
    trainer.extend(out_generated_image(model, test_batch, 10, 10, args.out), trigger=log_interval)
    trainer.extend(burn_in(model, epoch=30), trigger=val_interval)
    report = ['epoch', 'iteration']
    for loss_name in ["invertible", "lightness", "contrast", "local_structure", "quantization"]:
        for v in ["", "validation/"]:
            report.append(f"{v}main/{loss_name}")
    report.append("lr")
    trainer.extend(extensions.PrintReport(report), trigger=log_interval)
    trainer.extend(extensions.ProgressBar(update_interval=10))
    trainer.extend(extensions.PlotReport(['main/invertible', 'validation/main/invertible'], trigger=log_interval,  file_name='loss.png'))



    if args.resume:
        chainer.serializers.load_npz(args.resume, trainer)

    trainer.run()


# if __name__ == '__main__':
    # main()