#!/usr/bin/env python

import os

import numpy as np
from PIL import Image

import chainer
import chainer.backends.cuda
from chainer import Variable


def out_generated_image(model, test_batch, rows=3, cols=9, dst="result_img"):
    @chainer.training.make_extension()
    def make_image(trainer):
        t_rgb, _ = test_batch
        with chainer.using_config("train", False), chainer.using_config("type_check", False), chainer.using_config("enable_backprop", False):
            gray = model.encoder(t_rgb)
            y_rgb = model.decoder(gray)
        gray = chainer.cuda.to_cpu(gray.data)
        y_rgb = chainer.cuda.to_cpu(y_rgb.data)
        img = np.empty((3, 3 * 224, 9 * 224))
        t_rgb = chainer.cuda.to_cpu(t_rgb.data)
        for i in range(9):
            img[:, 0:224, i * 224: (i+1)*224] = t_rgb[i]
        gray = np.broadcast_to(gray, (9, 3, 224, 224))
        for i in range(9):
            img[:, 224:448, i * 224: (i+1)*224] = gray[i]
        for i in range(9):
            img[:, 448:672, i * 224: (i+1)*224] = y_rgb[i]
        preview_dir = f'{dst}/preview'
        preview_path = preview_dir +f'/image{trainer.updater.iteration:0>8}.png'
        if not os.path.exists(preview_dir):
            os.makedirs(preview_dir)
        img += 1
        img *= 122.5
        img = np.asarray(np.clip(img, 0.0, 255.0), dtype=np.uint8)
        Image.fromarray(img.transpose(1, 2, 0)).save(preview_path)
    return make_image


def burn_in(model, epoch=90):
    @chainer.training.make_extension()
    def stage_two(trainer):
        if trainer.updater.epoch == epoch:
            model.stage_two = True
    return stage_two