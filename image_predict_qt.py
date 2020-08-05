import os
import sys
import threading
import cv2  # install
import keras  # install
from keras.models import load_model
from keras import backend as K
import h5py  # install
import numpy as np  # install
from PIL import Image, ImageDraw, ImageFont  # install
from PyQt5.QtWidgets import QWidget, QPushButton, QApplication, QLabel, QMainWindow  # install
from PyQt5.QtGui import QPixmap, QImage, QFont
from PyQt5.QtCore import Qt, QTimer

np.random.seed(1337)

# 图像高宽
width, height, depth = 200, 200, 3

# 摄像头高宽
cam_width, cam_height = 800, 600

# 窗口大小
window_width, window_height = 1600, 1200


def get_class_and_confidence(img, model):
    # 根据训练好的模型获取预测的类别和置信度
    # test_data = np.empty((1, width*height*3))
    if depth == 1:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    try:
        img = cv2.resize(img, (width, height))
    except:
        print('resize error!')
        return -1, -1
    img_ndarray = np.asarray(img, dtype='float64') / 255
    test_data = np.ndarray.flatten(img_ndarray)
    test_data = test_data.astype('float32')

    if K.image_data_format() == 'channels_first':
        test_data = test_data.reshape(1, depth, height, width)
    else:
        test_data = test_data.reshape(1, height, width, depth)

    preds = model.predict(test_data)
    class_ = np.argmax(preds[0], axis=1)
    confidence = float(preds[0][class_])
    confidence = '%.3f' % (confidence * 100)  # 置信度转化为百分比，保留3位小数
    return class_, confidence


def predict_one_img(img_path):
    # 预测单张图片
    category = []
    with open('classes.txt', 'r') as f:
        lines = f.readlines()
        for line in lines:
            class_ = line[:-1]
            print(class_)
            category.append(class_)

    model = load_model('model.h5')
    img = cv2.imread(img_path)
    class_, confidence = get_class_and_confidence(img, model)
    category_name = category[int(class_)]

    img = Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    draw = ImageDraw.Draw(img)
    font_text = ImageFont.truetype("yy.ttf", 60, encoding="utf-8")
    draw.text((5, 5), category_name + ' %' +
              str(confidence), (0, 255, 0), font=font_text)
    img = cv2.cvtColor(np.asarray(img), cv2.COLOR_RGB2BGR)

    print(category_name, '%', str(confidence))
    # cv2.namedWindow('img', 0)
    # cv2.resizeWindow('img',window_width,window_height)

    cv2.imshow('img', img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


class MyWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setGeometry(500, 300, cam_width, cam_height + 150)
        self.setFixedSize(cam_width, cam_height + 150)
        self.setWindowTitle('菜品识别系统')

        self.img_label = QLabel(self)
        self.img_label.setGeometry(0, 0, cam_width, cam_height)

        self.dish_label = QLabel(self)
        self.dish_label.move(70, cam_height + 25)
        self.dish_label.resize(250, 25)
        self.dish_label.setText("菜品名称：")
        self.dish_label.setFont(QFont("Roman times", 18, QFont.Bold))

        self.price_label = QLabel(self)
        self.price_label.move(70, cam_height + 70)
        self.price_label.resize(250, 25)
        self.price_label.setText("金额：")
        self.price_label.setFont(QFont("Roman times", 18, QFont.Bold))

        self.isChecking = False

        check_button = QPushButton("结算", self)
        check_button.move(450, cam_height + 50)
        check_button.resize(130, 40)
        # check_button.clicked.connect(self.predict_one)
        check_button.clicked.connect(self.check)

        ok_button = QPushButton("确定", self)
        ok_button.move(600, cam_height + 50)
        ok_button.resize(130, 40)
        ok_button.clicked.connect(self.ok)

        self.category = []  # 类别列表
        self.price = {}  # 价格字典

        with open('classes.txt', 'r') as f:
            lines = f.readlines()
            for line in lines:
                class_ = line[:-1]
                self.category.append(class_)

        with open('price.txt', 'r') as f:
            lines = f.readlines()
            for line in lines:
                line = line[:-1].split(' ')
                class_, price = line[0], line[1]
                self.price[class_] = price

        print(self.category)
        print(self.price)

        self.model = load_model('model.h5')
        print('Model loading complete!')

        self.cap = cv2.VideoCapture(0)
        self.cap.set(3, cam_width)
        self.cap.set(4, cam_height)

        self._timer = QTimer(self)
        self._timer.timeout.connect(self.update)
        # self._timer.start(33)

        # self.setCentralWidget(self.img_label)
        self.show()

    def predict_one(self):
        img = cv2.imread('fqcd.jpg')
        img = cv2.resize(img, (cam_width, cam_height))
        # img=cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        class_, confidence = get_class_and_confidence(img, self.model)
        category_name = self.category[int(class_)]

        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = Image.fromarray(img)
        draw = ImageDraw.Draw(img)
        font_text = ImageFont.truetype("yy.ttf", 30, encoding="utf-8")
        draw.text((5, 5), category_name + str(confidence) +
                  '%', (0, 255, 0), font=font_text)
        img = cv2.cvtColor(np.asarray(img), cv2.COLOR_RGB2BGR)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        print(category_name, '%', str(confidence))

        h, w = img.shape[:2]
        img = QImage(img, w, h, QImage.Format_RGB888)
        img = QPixmap.fromImage(img)
        self.check_pixmap = img
        self.isChecking = True
        self.dish_label.setText("菜品名称：" + category_name)
        self.price_label.setText("金额：" + self.price[category_name] + "元")
        self.img_label.setPixmap(img)
        self.img_label.setScaledContents(True)  # 自适应大小

    def update(self):
        # 定时器获取摄像头帧并转化成pixmap再显示在label上
        if self.isChecking:
            self.img_label.setPixmap(self.check_pixmap)
            # self.img_label.setScaledContents(True) #自适应大小
            return
        ret, self.fram = self.cap.read()  # 读取摄像头帧
        print('fram shape', self.fram.shape)
        fram = cv2.cvtColor(self.fram, cv2.COLOR_BGR2RGB)
        h, w = fram.shape[:2]
        img = QImage(fram, w, h, QImage.Format_RGB888)
        img = QPixmap.fromImage(img)
        self.img_label.setPixmap(img)  # img label显示图像
        self.img_label.setScaledContents(True)  # 自适应大小

    def check(self):
        # 结算按钮
        if self.isChecking:
            return
        fram = self.fram
        class_, confidence = get_class_and_confidence(fram, self.model)
        category_name = self.category[int(class_)]

        img = cv2.cvtColor(fram, cv2.COLOR_BGR2RGB)
        img = Image.fromarray(img)

        draw = ImageDraw.Draw(img)
        font_text = ImageFont.truetype("yy.ttf", 30, encoding="utf-8")
        draw.text((5, 5), category_name + ' %' +
                  str(confidence), (0, 255, 0), font=font_text)

        img = cv2.cvtColor(np.asarray(img), cv2.COLOR_RGB2BGR)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        print(category_name, '%', str(confidence))

        h, w = img.shape[:2]
        img = QImage(img, w, h, QImage.Format_RGB888)
        img = QPixmap.fromImage(img)
        self.check_pixmap = img
        self.isChecking = True
        self.dish_label.setText("菜品名称：" + category_name)
        self.price_label.setText("金额：" + self.price[category_name] + "元")

    def ok(self):
        self.isChecking = False
        self.dish_label.setText("菜品名称：")
        self.price_label.setText("金额：")


def cv_loop():
    category = []
    with open('classes.txt', 'r') as f:
        lines = f.readlines()
        for line in lines:
            class_ = line[:-1]
            print(class_)
            category.append(class_)

    model = load_model('model.h5')

    cap = cv2.VideoCapture(0)
    cap.set(3, cam_width)
    cap.set(4, cam_height)
    while True:
        ret, fram = cap.read()
        if not ret:
            continue
        print(fram.shape)
        class_, confidence = get_class_and_confidence(fram, model)
        if not class_ == -1 and confidence == -1:
            break
        category_name = category[int(class_)]

        img = Image.fromarray(cv2.cvtColor(fram, cv2.COLOR_BGR2RGB))
        draw = ImageDraw.Draw(img)
        font_text = ImageFont.truetype("yy.ttf", 60, encoding="utf-8")
        draw.text((5, 5), category_name + ' %' +
                  str(confidence), (0, 255, 0), font=font_text)
        fram = cv2.cvtColor(np.asarray(img), cv2.COLOR_RGB2BGR)
        # cv2.putText(fram, category_name+' %'+str(confidence), (0,70), cv2.FONT_HERSHEY_COMPLEX, 3, (0, 255, 0), 6)

        print(category_name, '%', str(confidence))
        cv2.namedWindow('fram', 0)
        cv2.resizeWindow('fram', window_width, window_height)

        cv2.imshow('fram', fram)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    cap.release()
    cv2.destroyAllWindows()


def main():
    app = QApplication(sys.argv)
    my = MyWindow()
    sys.exit(app.exec_())


if __name__ == '__main__':
    main()
