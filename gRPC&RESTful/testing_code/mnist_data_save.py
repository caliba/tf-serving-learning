#-*- encoding = utf-8 -*-
#@Time :2021/12/22 15:01
#@Author : Agonsle
#@File :mnist_data_save.py
#@Software :PyCharm

# coding: utf-8
import urllib.request
import os.path
import gzip
import pickle
import os
import numpy as np

#  数据集位置
url_base = 'http://yann.lecun.com/exdb/mnist/'
key_file = {
    'train_img': 'train-images-idx3-ubyte.gz',
    'train_label': 'train-labels-idx1-ubyte.gz',
    'test_img': 't10k-images-idx3-ubyte.gz',
    'test_label': 't10k-labels-idx1-ubyte.gz'
}

# 将处理后的数据，保存到本地存储位置
dataset_dir = os.path.dirname(os.path.abspath(__file__)) # 当前文件夹下
save_file = dataset_dir + "/mnist.pkl"

train_num = 60000 # 训练集个数
test_num = 10000 # 测试集个数
img_dim = (1, 28, 28) # 每张图片的数据维度
img_size = 784 # 每张图片大小(28*28)

# 将单个具体的原始数据文件下载到本地
def _download(file_name):
    file_path = dataset_dir + "/" + file_name
    if os.path.exists(file_path):
        return

    print("Downloading " + file_name + " ... ")
    urllib.request.urlretrieve(url_base + file_name, file_path)
    print("Done")

# 下载4个数据文件
def download_mnist():
    for v in key_file.values():
        _download(v)

# 加载label数据，转化为Numpy数据格式
def _load_label(file_name):
    file_path = dataset_dir + "/" + file_name

    print("Converting " + file_name + " to NumPy Array ...")
    with gzip.open(file_path, 'rb') as f:
        labels = np.frombuffer(f.read(), np.uint8, offset=8)
    print("Done")

    return labels

# 加载图片数据，转化为numpy格式（n行，每行784个数字）
def _load_img(file_name):
    file_path = dataset_dir + "/" + file_name

    print("Converting " + file_name + " to NumPy Array ...")
    with gzip.open(file_path, 'rb') as f:
        data = np.frombuffer(f.read(), np.uint8, offset=16)
    data = data.reshape(-1, img_size)
    print("Done")

    return data

# 将下载的数据文件转化为numpy格式
def _convert_numpy():
    dataset = {}
    dataset['train_img'] = _load_img(key_file['train_img'])
    dataset['train_label'] = _load_label(key_file['train_label'])
    dataset['test_img'] = _load_img(key_file['test_img'])
    dataset['test_label'] = _load_label(key_file['test_label'])

    return dataset

# 从目标地址下载数据文件，转化为numpy格式，并保存为pickle文件
def init_mnist():
    download_mnist()
    dataset = _convert_numpy()
    print("Creating pickle file ...")
    with open(save_file, 'wb') as f:
        pickle.dump(dataset, f, -1)
    print("Done!")

# 将标签数据，转化为one_hot格式（正确数字对应下标为1，其他为0）
# 例如，原标签为[2,3],转化后为：
# [[0,0,1,0,0,0,0,0,0,0,0],[0,0,0,1,0,0,0,0,0,0,0]]
def _change_one_hot_label(X):
    T = np.zeros((X.size, 10))
    for idx, row in enumerate(T):
        row[X[idx]] = 1

    return T

#  主函数，如果第一次调用，从目标地址下载数据文件并处理为numpy后存为本地pickle文件
#  下次调用直接读取pickle文件，并根据参数，处理为相应的格式
def load_mnist(normalize=True, flatten=True, one_hot_label=False):
    """读入MNIST数据集

    Parameters
    ----------
    normalize : 将图像的像素值正规化为0.0~1.0
    one_hot_label :
        one_hot_label为True的情况下，标签作为one-hot数组返回
        one-hot数组是指[0,0,1,0,0,0,0,0,0,0]这样的数组
    flatten : 是否将图像展开为一维数组

    Returns
    -------
    (训练图像, 训练标签), (测试图像, 测试标签)
    """
    if not os.path.exists(save_file):
        init_mnist()

    with open(save_file, 'rb') as f:
        dataset = pickle.load(f)

    if normalize:
        for key in ('train_img', 'test_img'):
            dataset[key] = dataset[key].astype(np.float32)
            dataset[key] /= 255.0

    if one_hot_label:
        dataset['train_label'] = _change_one_hot_label(dataset['train_label'])
        dataset['test_label'] = _change_one_hot_label(dataset['test_label'])

    if not flatten:
        for key in ('train_img', 'test_img'):
            dataset[key] = dataset[key].reshape(-1, 1, 28, 28)

    return (dataset['train_img'], dataset['train_label']), (dataset['test_img'], dataset['test_label'])


if __name__ == '__main__':
    init_mnist()