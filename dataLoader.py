# coding:utf-8
import numpy as  np
import gzip
from struct import unpack

#读取图像
def __read_image(path):
    with gzip.open(path, 'rb') as f:
        magic, num, rows, cols = unpack('>4I', f.read(16))
        img=np.frombuffer(f.read(), dtype=np.uint8).reshape(num, 28*28)
    return img

#读取标签
def __read_label(path):
    with gzip.open(path, 'rb') as f:
        magic, num = unpack('>2I', f.read(8))
        lab = np.frombuffer(f.read(), dtype=np.uint8)
        # print(lab[1])
    return lab

#将图像信息正则化，即0-255 --> 0-1
def __normalize_image(image):
    img = image.astype(np.float32) / 255.0
    return img

#将标签进行one-hot编码，如数字标签5转换为[0,0,0,0,0,1,0,0,0,0,0,0]
def __one_hot_label(label):
    lab = np.zeros((label.size, 10))
    for i, row in enumerate(lab):
        row[label[i]] = 1
    return lab

def loadMnist(train_data_path, train_labels_path, test_data_path, test_labels_path, normalize = 1, one_hot = 1):
    image = {
        'train' : __read_image(train_data_path),
        'test' : __read_image(test_data_path)
    }
    label = {
        'train' : __read_label(train_labels_path),
        'test' : __read_label(test_labels_path)
    }
    if normalize:
        for type in ('train', 'test'):
            image[type] = __normalize_image(image[type])
    if one_hot:
        for type in ('train', 'test'):
            label[type] = __one_hot_label(label[type])
    return (image['train'], label['train']), (image['test'], label['test'])