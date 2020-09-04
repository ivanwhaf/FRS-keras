# @Author: Ivan
# @LastEdit: 2020/9/3
import os
import time
import cv2  # install
import numpy as np  # install
import keras  # install
from keras.regularizers import l2
from keras import backend as K
from keras.utils import np_utils
from keras.utils import plot_model
from keras.optimizers import SGD
from keras.models import Sequential
from keras.models import load_model
from keras.layers import MaxPooling2D, SeparableConv2D
from keras.layers import Dense, Dropout, Activation, Flatten
import matplotlib.pyplot as plt  # install
from image_preprocess import load_img
from image_util import show_intermediate_output, show_heatmap

np.random.seed(1337)
os.environ["PATH"] += os.pathsep + \
                      'D:/Graphviz2.38/bin'  # add graphviz to environment variable,for plotting network's structure

path = './dataset'  # root path of dataset
epochs = 800  # number of training
nb_classes = 5  # number of class
nb_per_class = 200  # number of each class
batch_size = 64
learning_rate = 0.001
activation = 'relu'
width, height, depth = 100, 100, 3
nb_filters1, nb_filters2 = 5, 10  # number of conv kernel(output dimension)
train_proportion = 0.8  # proportion of train set
valid_proportion = 0.1  # proportion of valid set
test_proportion = 0.1  # proportion of test set


def set_model(lr=learning_rate, decay=1e-6, momentum=0.9):
    model = Sequential()
    if K.image_data_format() == 'channels_first':
        model.add(SeparableConv2D(nb_filters1, kernel_size=(3, 3), kernel_regularizer=l2(0.01),
                                  input_shape=(depth, height, width), name='conv1'))
    else:
        model.add(SeparableConv2D(nb_filters1, kernel_size=(3, 3),
                                  input_shape=(height, width, depth), name='conv1'))
    model.add(Activation(activation))
    model.add(MaxPooling2D(pool_size=(2, 2), name='maxpooling1'))
    model.add(Dropout(0.5))

    model.add(SeparableConv2D(nb_filters2, kernel_size=(3, 3),
                              kernel_regularizer=l2(0.01), name='conv2'))
    model.add(Activation(activation))
    model.add(MaxPooling2D(pool_size=(2, 2), name='maxpooling2'))
    model.add(Dropout(0.5))

    model.add(Flatten())
    model.add(Dense(128, kernel_regularizer=l2(
        0.01), name='dense1'))  # Full connection
    model.add(Activation(activation))
    model.add(Dropout(0.5))

    model.add(Dense(nb_classes, name='dense2'))  # Output
    model.add(Activation('softmax'))

    sgd = SGD(lr=lr, decay=decay, momentum=momentum,
              nesterov=True)  # optimizer

    model.compile(loss='categorical_crossentropy',
                  optimizer=sgd, metrics=['accuracy'])
    model.summary()  # output each layer's parameter of the model
    return model


class LossHistory(keras.callbacks.Callback):
    # record loss history,output parameters' changing
    def on_train_begin(self, logs={}):
        self.losses = {'batch': [], 'epoch': []}
        self.accuracy = {'batch': [], 'epoch': []}
        self.val_loss = {'batch': [], 'epoch': []}
        self.val_acc = {'batch': [], 'epoch': []}

    def on_batch_end(self, batch, logs={}):
        self.losses['batch'].append(logs.get('loss'))
        self.accuracy['batch'].append(logs.get('accuracy'))
        self.val_loss['batch'].append(logs.get('val_loss'))
        self.val_acc['batch'].append(logs.get('val_accuracy'))

    def on_epoch_end(self, batch, logs={}):
        self.losses['epoch'].append(logs.get('loss'))  # train loss
        self.accuracy['epoch'].append(logs.get('accuracy'))  # train acc
        self.val_loss['epoch'].append(logs.get('val_loss'))  # val loss
        self.val_acc['epoch'].append(logs.get('val_accuracy'))  # val acc

    def loss_plot(self, loss_type):
        iters = range(len(self.losses[loss_type]))
        plt.figure('Change of accuracy and loss')
        # train acc
        plt.plot(iters, self.accuracy[loss_type], 'r', label='train acc')
        # train loss
        plt.plot(iters, self.losses[loss_type], 'g', label='train loss')

        if loss_type == 'epoch':
            # val_acc
            plt.plot(iters, self.val_acc[loss_type], 'b', label='val acc')
            # val_loss
            plt.plot(iters, self.val_loss[loss_type], 'k', label='val loss')

        plt.title('epoch:' + str(epochs) + ',lr:' + str(learning_rate) + ',batch_size:' + str(batch_size) +
                  '\nactivation:' + activation + ',nb_classes:' + str(nb_classes) + ',nb_per_class:' + str(
            nb_per_class))

        plt.grid(True)
        plt.xlabel(loss_type)
        plt.ylabel('acc-loss')
        plt.legend(loc="upper right")
        now = time.strftime('%Y-%m-%d@%H-%M-%S', time.localtime(time.time()))
        plt.savefig('./parameter/' + now + '.jpg')
        plt.show()


history = LossHistory()


def train_model(model, X_train, Y_train, X_val, Y_val):
    # tensorboard = keras.callbacks.TensorBoard(
    #     log_dir='F:/Log/', histogram_freq=1)

    model.fit(X_train, Y_train, batch_size=batch_size, epochs=epochs, shuffle=True,
              verbose=1, validation_data=(X_val, Y_val), callbacks=[history])
    model.save('model.h5')
    return model


def test_model(X_test, Y_test):
    model = load_model('model.h5')
    score = model.evaluate(X_test, Y_test, verbose=0)
    return score


def main():
    (X_train, y_train), (X_val, y_val), (X_test, y_test) = load_img(path, nb_classes, nb_per_class,
                                                                    width, height, depth, train_proportion,
                                                                    valid_proportion,
                                                                    test_proportion)  # load dataset

    if K.image_data_format() == 'channels_first':
        X_train = X_train.reshape(X_train.shape[0], depth, height, width)
        X_val = X_val.reshape(X_val.shape[0], depth, height, width)
        X_test = X_test.reshape(X_test.shape[0], depth, height, width)
    else:
        X_train = X_train.reshape(X_train.shape[0], height, width, depth)
        X_val = X_val.reshape(X_val.shape[0], height, width, depth)
        X_test = X_test.reshape(X_test.shape[0], height, width, depth)

    print('X_train shape:', X_train.shape)
    print('Class number:', nb_classes)
    print(X_train.shape[0], 'train samples')
    print(X_val.shape[0], 'validate samples')
    print(X_test.shape[0], 'test samples')

    # convert class vectors to binary class matrices
    Y_train = np_utils.to_categorical(y_train, nb_classes)
    Y_val = np_utils.to_categorical(y_val, nb_classes)
    Y_test = np_utils.to_categorical(y_test, nb_classes)

    model = set_model()  # load network model

    plot_model(model, to_file='model.png', show_shapes=True, expand_nested=True)  # save network's structure picture

    classes = []
    with open('classes.txt', 'r', encoding='utf-8') as f:
        for line in f.readlines():
            class_ = line[:-1]
            classes.append(class_)

    start = time.clock()
    model = train_model(model, X_train, Y_train, X_val, Y_val)  # train model
    end = time.clock()

    # visualize intermedia layers' output,visualize CAM
    image = cv2.imread('test.jpg')
    show_intermediate_output(model, 'conv1', image)
    image = cv2.imread('test.jpg')
    show_intermediate_output(model, 'maxpooling1', image)
    image = cv2.imread('test.jpg')
    show_intermediate_output(model, 'conv2', image)
    image = cv2.imread('test.jpg')
    show_intermediate_output(model, 'maxpooling2', image)
    image = cv2.imread('test.jpg')
    show_heatmap(model, 'conv2', image)

    score = test_model(X_test, Y_test)  # evaluate model's score

    pred_classes = model.predict_classes(X_test, verbose=0)  # predict class

    # test_accuracy = np.mean(np.equal(y_test, pred_classes))
    right = np.sum(np.equal(y_test, pred_classes))
    for i in range(0, nb_classes * int(test_proportion * nb_per_class)):
        if y_test[i] != pred_classes[i]:
            actual_class_name = classes[int(y_test[i % nb_per_class])]
            pred_class_name = classes[int(pred_classes[i % nb_per_class])]
            print(actual_class_name, 'was wrongly classified as', pred_class_name)

    print('Total training time:', end - start)
    print('Test number:', len(Y_test))
    print('Test right:', right)
    print('Test wrong:', len(Y_test) - right)
    print('Test loss:', score[0])
    print('Test accuracy:', score[1])
    history.loss_plot('epoch')  # plot parameter changing diagram


if __name__ == '__main__':
    main()
