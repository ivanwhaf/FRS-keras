import os
import time
import cv2  # install
import numpy as np  # install
from PIL import Image  # install
import keras  # install
from keras import backend as K
from keras.models import Model
from keras.utils import np_utils
from keras.utils import plot_model
from keras.optimizers import SGD
from keras.models import Sequential
from keras.models import load_model
from keras.layers import Conv2D, MaxPooling2D
from keras.layers import Dense, Dropout, Activation, Flatten
import matplotlib.pyplot as plt  # install
from image_preprocessing import load_img

np.random.seed(1337)
os.environ["PATH"] += os.pathsep + \
    'C:/Program Files (x86)/Graphviz2.38/bin'  # 添加graphviz至环境变量 用于输出网络结构图

path = 'data'  # 数据集路径
epochs = 1000  # 轮数
nb_classes = 5  # 图片种类
number_per_category = 100  # 每类图片数量
batch_size = 32  # 一次训练的样本数
lr = 0.0001  # 学习率
activation = 'tanh'  # 激活函数
width, height, depth = 200, 200, 3  # 图片的宽、高、深度
nb_filters1, nb_filters2 = 5, 10  # 卷积核的数目（即输出的维度）
train_per_category = int(number_per_category*0.8)  # 每个类别训练数量
valid_per_category = int(number_per_category*0.1)  # 每个类别验证数量
test_per_category = int(number_per_category*0.1)  # 每个类别测试数量


def set_model(lr=lr, decay=1e-6, momentum=0.9):
    # 模型初始化设置
    model = Sequential()
    if K.image_data_format() == 'channels_first':
        model.add(Conv2D(nb_filters1, kernel_size=(3, 3),
                         input_shape=(depth, height, width), name='conv1'))
    else:
        model.add(Conv2D(nb_filters1, kernel_size=(3, 3),
                         input_shape=(height, width, depth), name='conv1'))
    model.add(Activation(activation))
    model.add(MaxPooling2D(pool_size=(2, 2)))

    model.add(Conv2D(nb_filters2, kernel_size=(3, 3), name='conv2'))
    model.add(Activation(activation))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.6))

    model.add(Flatten())
    model.add(Dense(128, name='dense1'))  # Full connection
    model.add(Activation(activation))
    model.add(Dropout(0.6))

    model.add(Dense(nb_classes, name='dense2'))  # output
    model.add(Activation('softmax'))

    sgd = SGD(lr=lr, decay=decay, momentum=momentum, nesterov=True)
    model.compile(loss='categorical_crossentropy',
                  optimizer=sgd, metrics=['accuracy'])

    plot_model(model, to_file='model.png')  # 保存模型结构图
    model.summary()
    return model


def train_model(model, X_train, Y_train, X_val, Y_val):
    model.fit(X_train, Y_train, batch_size=batch_size, epochs=epochs, shuffle=True,
              verbose=1, validation_data=(X_val, Y_val), callbacks=[history])
    model.save('model.h5')
    return model


def test_model(X_test, Y_test):
    model = load_model('model.h5')
    score = model.evaluate(X_test, Y_test, verbose=0)
    return score


def deprocess_image(x):
    if K.image_data_format() == 'channels_first':
        x = x.transpose(1, 2, 3, 0)
    elif K.image_data_format() == 'channels_last':
        x = x.transpose(3, 1, 2, 0)
    x *= 255.
    x = np.clip(x, 0, 255).astype('uint8')
    return x


def conv_output(model, layer_name, img):
    """Get the output of conv layer.
    Args:
           model: keras model.
           layer_name: name of layer in the model.
           img: processed input image.

    Returns:
           intermediate_output: feature map.
    """
    # this is the placeholder for the input images
    input_img = model.input

    try:
        # this is the placeholder for the conv output
        out_conv = model.get_layer(layer_name).output
    except:
        raise Exception('Not layer named {}!'.format(layer_name))

    # get the intermediate layer model
    intermediate_layer_model = Model(inputs=input_img, outputs=out_conv)

    # get the output of intermediate layer model
    intermediate_output = intermediate_layer_model.predict(img)
    return intermediate_output[0]


def show_intermediate_output(model):
    # 显示中间层输出
    image = cv2.imread('test.jpg')
    image = cv2.resize(image, (width, height))
    image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    img_ndarray = np.asarray(image, dtype='float64')/255
    image = np.ndarray.flatten(img_ndarray)
    test_data = image
    test_data = test_data.astype('float32')

    if K.image_data_format() == 'channels_first':
        test_data = test_data.reshape(1, 1, height, width)
    else:
        test_data = test_data.reshape(1, height, width, 1)
    o = conv_output(model, 'conv1', test_data)

    # cv2.imshow('i',image_array)
    # cv2.imshow('o',o)


class LossHistory(keras.callbacks.Callback):
    # 损失历史记录 输出参数变化图像
    def on_train_begin(self, logs={}):
        self.losses = {'batch': [], 'epoch': []}
        self.accuracy = {'batch': [], 'epoch': []}
        self.val_loss = {'batch': [], 'epoch': []}
        self.val_acc = {'batch': [], 'epoch': []}

    def on_batch_end(self, batch, logs={}):
        self.losses['batch'].append(logs.get('loss'))
        self.accuracy['batch'].append(logs.get('acc'))
        self.val_loss['batch'].append(logs.get('val_loss'))
        self.val_acc['batch'].append(logs.get('val_acc'))

    def on_epoch_end(self, batch, logs={}):
        self.losses['epoch'].append(logs.get('loss'))
        self.accuracy['epoch'].append(logs.get('acc'))
        self.val_loss['epoch'].append(logs.get('val_loss'))
        self.val_acc['epoch'].append(logs.get('val_acc'))

    def loss_plot(self, loss_type):
        iters = range(len(self.losses[loss_type]))
        plt.figure(num='Change of parameters')
        # acc
        plt.plot(iters, self.accuracy[loss_type], 'r', label='train acc')
        # loss
        plt.plot(iters, self.losses[loss_type], 'g', label='train loss')

        if loss_type == 'epoch':
            # val_acc
            plt.plot(iters, self.val_acc[loss_type], 'b', label='val acc')
            # val_loss
            plt.plot(iters, self.val_loss[loss_type], 'k', label='val loss')

        plt.title('epoch='+str(epochs)+',lr='+str(lr)+',batch_size='+str(batch_size)+'\nactivation=' +
                  activation+',nb_classes='+str(nb_classes)+',nb_per_class='+str(number_per_category))
        plt.grid(True)
        plt.xlabel(loss_type)
        plt.ylabel('acc-loss')
        plt.legend(loc="upper right")
        now = time.strftime('%Y-%m-%d@%H-%M-%S', time.localtime(time.time()))
        plt.savefig('./parameter/'+now+'.jpg')
        plt.show()


def main():
    (X_train, y_train), (X_val, y_val), (X_test, y_test), category = load_img(path, nb_classes, number_per_category,
                                                                              width, height, depth, train_per_category, valid_per_category, test_per_category)  # 加载图片训练集

    if K.image_data_format() == 'channels_first':
        X_train = X_train.reshape(X_train.shape[0], depth, height, width)
        X_val = X_val.reshape(X_val.shape[0], depth, height, width)
        X_test = X_test.reshape(X_test.shape[0], depth, height, width)
        input_shape = (depth, height, width)
    else:
        X_train = X_train.reshape(X_train.shape[0], height, width, depth)
        X_val = X_val.reshape(X_val.shape[0], height, width, depth)
        X_test = X_test.reshape(X_test.shape[0], height, width, depth)
        input_shape = (height, width, depth)

    print('X_train shape:', X_train.shape)
    print('Class number:', nb_classes)
    print(X_train.shape[0], 'train samples')
    print(X_val.shape[0], 'validate samples')
    print(X_test.shape[0], 'test samples')

    # convert class vectors to binary class matrices
    Y_train = np_utils.to_categorical(y_train, nb_classes)
    Y_val = np_utils.to_categorical(y_val, nb_classes)
    Y_test = np_utils.to_categorical(y_test, nb_classes)

    model = set_model()  # 加载神经网络模型

    # 生成含有所有类别的txt文件
    with open('classes.txt', 'w') as f:
        for c in category:
            f.write(c+'\n')

    start = time.clock()
    model = train_model(model, X_train, Y_train, X_val, Y_val)  # 训练模型
    end = time.clock()

    score = test_model(X_test, Y_test)  # 评价得分

    classes = model.predict_classes(X_test, verbose=0)  # 预测类别
    test_accuracy = np.mean(np.equal(y_test, classes))

    wrong = 0
    for i in range(0, nb_classes*test_per_category):
        if y_test[i] != classes[i]:
            wrong = wrong+1
            category_test = category[int(y_test[i % number_per_category])]
            category_class = category[int(classes[i % number_per_category])]
            print(category_test, 'was wrongly classified as', category_class)

    print('Total training time:'+str(end-start)+'s')
    print('Score:', score)
    print('Test number:'+str(len(classes)))
    print('Wrong:'+str(wrong))
    print('Test accuarcy:', test_accuracy)
    history = LossHistory()
    history.loss_plot('epoch')


if __name__ == '__main__':
    main()
