# coding:utf-8
import numpy as np
#import matplotlib.pyplot as plt

def ReLU(X):
    #激活函数
    return(np.maximum(0,X))

def deReLU(X):
    #激活函数的导数
    X[X<=0] = 0
    X[X>0] = 1
    return X

def softmax(x):
    e = np.exp(x)
    return e / np.sum(e, axis=1).reshape(x.shape[0], 1)

#交叉熵损失函数
def cross_entropy_loss(output, label):
    delta = 1e-7 #防止出现log（0）的情况
    return -np.sum(label * np.log(output + delta))

def accuracy(Y, y_hat):
    #计算预测的准确度
    n = Y.shape[0]
    Y_label = np.argwhere(Y == 1)[:, 1:]
    y_hat_label = np.argwhere(y_hat == np.max(y_hat, axis=1).reshape(n, 1))[:, 1:]
    mask = np.zeros(Y_label.shape)
    mask[Y_label == y_hat_label] = 1
    return np.sum(mask) / n

class MLP_model:
    def __init__(self, layer_size, learning_rate=0.001, batch_size=64, max_epoch=40):
        self.layer_size = layer_size #各层神经元个数，如：[784, 256, 256, 10]
        self.learning_rate = learning_rate
        self.batch_size = batch_size
        self.max_epoch = max_epoch #训练的轮数

        self.layer_num = len(layer_size) #隐藏层数量+输入层+输出层
        self.weights = []#各层（输入层、隐层）输出的权重
        self.bias = []#各层（输入层、隐层）输出的偏
        for i in range(1, self.layer_num):
            #由于使用relu激活函数，因此使用He初始化
            self.weights.append(
                np.random.randn(self.layer_size[i-1], self.layer_size[i]) * np.sqrt(2 / self.layer_size[i-1])
            )
            self.bias.append(np.random.randn(self.layer_size[i])) 
        
        self.delta_w = []#损失函数关于各层（输入层、隐层）输出的权重weight的偏导
        self.delta_b = []#损失函数关于各层（输入层、隐层）输出的偏移bias的偏导
        self.input_net = []#各层的输入
        self.output_net = []#各层的输出
        for i in range(self.layer_num):
            self.input_net.append(np.zeros(self.layer_size[i]))
            self.output_net.append(np.zeros(self.layer_size[i]))

    def forward(self, data):
        #前向传输，data为输入层的输入数据
        self.output_net[0] = data
        for j in range(1, self.layer_num-1):
            #求隐藏层的输入与输出
            self.input_net[j] = np.dot(self.output_net[j-1], self.weights[j-1]) + self.bias[j-1]
            self.output_net[j] = ReLU(self.input_net[j])
        #求输出层的输入与输出
        self.input_net[-1] = np.dot(self.output_net[-2], self.weights[-1]) + self.bias[-1]
        self.output_net[-1] = softmax(self.input_net[-1])
        y_hat = self.output_net[-1]
        return y_hat
    
    def backward(self, y_hat, y):
        self.delta_b = []
        self.delta_w = []
        delta_output = y_hat - y #输出层的梯度，即损失函数关于输出层输入的导数
        layer_index = self.layer_num - 1
        while layer_index > 0:
            self.delta_b.append(np.sum(delta_output, axis=0))
            self.delta_w.append(np.dot(self.output_net[layer_index-1].T, delta_output))
            if layer_index > 1:
                #计算前一层的梯度
                delta_output = np.dot(delta_output, self.weights[layer_index-1].T) * deReLU(self.input_net[layer_index-1])
            layer_index -= 1

    def train(self, inputs, y):
        if self.weights == []:
            for i in range(1, self.layer_num):
                #由于使用relu激活函数，因此使用He初始化
                self.weights.append(
                    np.random.randn(self.layer_size[i-1], self.layer_size[i]) * np.sqrt(2 / self.layer_size[i-1])
                )
        if self.bias == []:
            for i in range(1, self.layer_num):
                self.bias.append(np.random.randn(self.layer_size[i])) 
        n_sample = inputs.shape[0] #输入样本数
        iter_num = n_sample // self.batch_size #训练一轮所需迭代次数
        if n_sample % self.batch_size: iter_num += 1
        for epoch_index in range(self.max_epoch):
            for iter_index in range(iter_num):
                sample_index_array = []
                for i in range(self.batch_size):
                    sample_index = np.random.randint((n_sample))
                    sample_index_array.append(sample_index)
                input_batch = inputs[sample_index_array]
                y_batch = y[sample_index_array]
                # input_batch = inputs[sample_index * self.batch_size: min((sample_index + 1) * self.batch_size, n_sample)] #输入样本batch
                # y_batch = y[sample_index * self.batch_size: min((sample_index + 1) * self.batch_size, n_sample)]#label batch
                yhat_batch = self.forward(input_batch)#前向传播
                self.backward(yhat_batch, y_batch)#反向传播

                for i in range(len(self.weights)):
                    #调整参数
                    self.weights[i] -= self.learning_rate * self.delta_w[self.layer_num-2-i]
                    self.bias[i] -= self.learning_rate * self.delta_b[self.layer_num-2-i]
            if epoch_index < 10:
                self.learning_rate -= 0.0001 * self.learning_rate
            #计算损失与准确度
            y_hat = self.forward(inputs)
            loss = cross_entropy_loss(y_hat, y)
            accur = accuracy(y, y_hat)
            print('Epoch: {} Loss: {} accuracy: {}'.format(epoch_index, loss, accur))
        
    def predict(self, inputs, y):
        #对测试集进行预测
        y_hat = self.forward(inputs)
        loss = cross_entropy_loss(y_hat, y)
        accur = accuracy(y, y_hat)
        print('Loss in test set: {}\nAccuracy in test set: {}'.format(loss, accur))