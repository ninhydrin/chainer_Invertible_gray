import numpy as np

import chainer
import chainer.functions as F
from chainer import initializers, Variable as V
import chainer.links as L
from chainer.functions.loss.vae import gaussian_kl_divergence
from chainer.backends import cuda

class Block(chainer.Chain):
    def __init__(self, out_ch):
        super(Block, self).__init__()
        with self.init_scope():
            self.conv1 = L.Convolution2D(None, out_ch, 3, pad=1)
            self.conv2 = L.Convolution2D(None, out_ch, 3, pad=1)

    def __call__(self, x):
        h = F.relu(self.conv1(x))
        h = self.conv2(h)
        return h + x


class IGEncoder(chainer.Chain):
    def __init__(self):
        super(IGEncoder, self).__init__()
        with self.init_scope():
            self.conv1 = L.Convolution2D(None, 64, 3, pad=1)
            self.block2 = Block(64)
            self.block3 = Block(64)
            self.conv4 = L.Convolution2D(None, 128, 3, stride=2, pad=1)
            self.conv5 = L.Convolution2D(None, 128, 3, stride=1, pad=1)

            self.conv6 = L.Convolution2D(None, 256, 3, stride=2, pad=1)
            self.conv7 = L.Convolution2D(None, 256, 3, stride=1, pad=1)

            self.block8 = Block(256)
            self.block9 = Block(256)
            self.block10 = Block(256)
            self.block11 = Block(256)
            self.block12 = Block(256)
            self.block13 = Block(256)

            self.conv14 = L.Convolution2D(None, 128, 3, stride=1, pad=1)
            self.conv14_2 = L.Convolution2D(None, 128, 3, stride=1, pad=1)
            self.conv15 = L.Convolution2D(None, 64, 3, stride=1, pad=1)
            self.conv15_2 = L.Convolution2D(None, 64, 3, stride=1, pad=1)

            self.block16 = Block(64)
            self.block17 = Block(64)

            self.conv18 = L.Convolution2D(None, 1, 3, stride=1, pad=1)

    def __call__(self, x):
        h = self.conv1(x)
        h = self.block2(h)
        h1 = self.block3(h)
        h = self.conv4(h1)
        h2 = F.relu(self.conv5(h))

        h = self.conv6(h2)
        h = F.relu(self.conv7(h))

        h = self.block8(h)
        h = self.block9(h)
        h = self.block10(h)
        h = self.block11(h)
        # h = self.block12(h)
        # h = self.block13(h)

        # _, index1 = F.max_pooling_2d(h2, ksize=2)
        h = F.resize_images(h, h2.shape[2:])
        h = self.conv14(h)
        h = F.relu(self.conv14_2(h)) + h2

        # _, index2 = F.max_pooling_2d(h1, ksize=2)
        # h = F.upsampling_2d(h, index2)
        h = F.resize_images(h, h1.shape[2:])
        h = self.conv15(h)
        h = F.relu(self.conv15_2(h)) + h1

        h = self.block16(h)
        h = self.block17(h)
        h = F.tanh(self.conv18(h))

        return h


class IGDecoder(chainer.Chain):
    def __init__(self):
        super(IGDecoder, self).__init__()
        with self.init_scope():
            self.conv = L.Convolution2D(None, 64, 3, pad=1)
            for i in range(8):
                setattr(self, f"block{i}", Block(64))
            self.conv8 = L.Convolution2D(None, 256, 3, pad=1)
            self.conv8_2 = L.Convolution2D(None, 3, 1)

    def __call__(self, x):
        h = self.conv(x)
        for i in range(8):
            h = getattr(self, f"block{i}")(h)
        h = self.conv8(h)
        h = F.tanh(self.conv8_2(h))
        return h


class InvertibleGray(chainer.Chain):
    def __init__(self, stage_two=False):
        super(InvertibleGray, self).__init__()
        self.insize = 224
        self.alpha = 1e-17
        self.beta = 0.5
        self.stage_two = stage_two
        with self.init_scope():
            self.encoder = IGEncoder()
            self.decoder = IGDecoder()
        self.vgg = L.VGG19Layers()
        self.wh = self.xp.array([[[[1], [-1]]]], dtype="f")
        self.ww = self.xp.array([[[[1, -1]]]], dtype="f")
        self.mean = np.array([103.939, 116.779, 123.68], dtype="f").reshape(1, 3, 1, 1)

    def to_gpu(self, device=None):
        super(InvertibleGray, self).to_gpu(device)
        with cuda._get_device(device):
            self.vgg = L.VGG19Layers().to_gpu()
            self.wh = self.xp.array([[[[1], [-1]]]], dtype="f")
            self.ww = self.xp.array([[[[1, -1]]]], dtype="f")
            self.mean = chainer.backends.cuda.to_gpu(self.mean)
        return self


    def __call__(self, x):
        t_color, t_gray = x
        with chainer.using_config("cudnn_deterministic", True):
            y_gray = self.encoder(t_color)
            y_color = self.decoder(y_gray)
        # t_color, t_gray = t["color"], t["gray"]
        invertible_loss = F.mean_squared_error(y_color, t_color)
        lightness_loss = self.calc_lightness_loss(y_gray, t_gray)
        contrast_loss = self.calc_contrast_loss(y_gray, t_color)
        local_structure_loss = self.calc_local_structure_loss(y_gray, t_gray)
        combind_loss = local_structure_loss + contrast_loss * self.alpha + lightness_loss * self.beta

        report = {
            "invertible": invertible_loss,
            "lightness": lightness_loss,
            "contrast": contrast_loss,
            "local_structure": local_structure_loss,
        }
        if not self.stage_two:
            loss = invertible_loss * 3 + combind_loss
        else:
            quantization_loss = self.calc_quantization_loss(y_gray)
            report["quantization"] = quantization_loss
            # loss = invertible_loss * 3 + combind_loss * 0.5 + quantization_loss * 10
            loss = invertible_loss * 3 + combind_loss * 1 + quantization_loss * 0.5
        chainer.report(report, self)
        return loss

    def calc_lightness_loss(self, x, gray):
        diff = F.absolute_error((x + 1) / 2, (gray + 1) / 2)
        loss = F.mean(F.maximum(diff - 70/127, self.xp.zeros(x.shape).astype("f")))
        return loss

    def calc_contrast_loss(self, x, t):
        y_rgb = F.broadcast_to(x, (len(x), 3, x.shape[2], x.shape[3]))
        y_conv4_feature = self.vgg((y_rgb + 1) / 2 * 255 - self.mean, ["conv4_4"])["conv4_4"]
        t_rgb = t
        t_conv4_feature = self.vgg((t_rgb + 1) / 2 * 255 - self.mean, ["conv4_4"])["conv4_4"]
        loss = F.mean_absolute_error(y_conv4_feature, t_conv4_feature)
        return loss

    def calc_local_structure_loss(self, x, t):
        tv_loss_h = F.mean_absolute_error(F.depthwise_convolution_2d(x, self.wh), F.depthwise_convolution_2d(t, self.wh))
        tv_loss_w = F.mean_absolute_error(F.depthwise_convolution_2d(x, self.ww), F.depthwise_convolution_2d(t, self.ww))
        loss = tv_loss_h + tv_loss_w
        return loss

    def calc_quantization_loss(self, x):
        img_255 = (x + 1) / 2 * 255
        quantized_grayscale = self.xp.clip(img_255.data.round(), 0, 255)
        return F.mean_absolute_error(img_255, quantized_grayscale)
