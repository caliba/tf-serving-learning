#-*- encoding = utf-8 -*-
#@Time :2021/12/22 15:04
#@Author : Agonsle
#@File :image_show.py
#@Software :PyCharm

# coding: utf-8
import numpy as np
from mnist_data_save import load_mnist #MNISTData是上一段代码的文件名
from PIL import Image


def img_show(img):
    pil_img = Image.fromarray(np.uint8(img))
    pil_img.show()

# 加载测试集合训练集
(x_train, t_train), (x_test, t_test) = load_mnist(flatten=True, normalize=False)

img = x_train[0] # 第一张图片
label = t_train[0] # 第一张图片的标签
print(label)  # 5
#
print(img.shape)  # (784,)
img = img.reshape(28, 28)  # 把图像的形状变为原来的尺寸
print(img.shape)  # (28, 28)
#
img_show(img)