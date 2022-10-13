import numpy as np
#import matplotlib.pyplot as plt
from dataLoader import loadMnist
import matplotlib.pyplot as plt
import MLP
#导入数据
train_images_path = './dataset/train-images-idx3-ubyte.gz'
train_labels_path = './dataset/train-labels-idx1-ubyte.gz'
test_images_path = './dataset/t10k-images-idx3-ubyte.gz'
test_labels_path = './dataset/t10k-labels-idx1-ubyte.gz'
(train_image, train_label), (test_image, test_label) = loadMnist(train_images_path, train_labels_path, test_images_path, test_labels_path)
# train_num = 60000
# test_num = 10000
#验证数据正确性
def show_train(index):
    plt.imshow(train_image[index].reshape(28,28), cmap = 'gray')
    print('label:{}'.format(train_label[index].argmax()))
show_train(np.random.randint(60000))
#建立MLP网络模型
mlp = MLP.MLP_model(layer_size=[784,256,256,10],learning_rate=0.002, batch_size=64, max_epoch=40)
mlp.train(train_image, train_label)#使用训练集数据进行测试
mlp.predict(test_image, test_label)#使用测试集数据测试网络对于手写体数字的识别能力