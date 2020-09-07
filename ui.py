# @Author: Ivan
# @LastEdit: 2020/9/7
import sys
import cv2  # install
from keras.models import load_model
from keras import backend as K
import numpy as np  # install
from PIL import Image, ImageDraw, ImageFont  # install
from PyQt5.QtWidgets import QWidget, QPushButton, QApplication, QLabel, QMainWindow  # install
from PyQt5.QtGui import QPixmap, QImage, QFont
from PyQt5.QtCore import Qt, QTimer

np.random.seed(1337)

# input shape
width, height, depth = 100, 100, 3

# camera shape
cam_width, cam_height = 800, 600

# window shape
window_width, window_height = 1600, 1200


def predict_img(img, model):
    """get model prediction of one image

    Arguments:
        img: image ndarray
        model: keras trained model
    Returns:
        predictions: keras model prediction
    """
    if depth == 1:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
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

    predictions = model.predict(test_data)
    return predictions


def predict_class_and_confidence(img, model):
    """predict image class and confidence according to trained model

    Arguments:
        img: prediction image
        model: keras model
    Returns:
        class_: class index
        confidence: class confidence
    """
    # call base prediction function
    preds = predict_img(img, model)

    class_ = np.argmax(preds[0])
    confidence = float(preds[0][class_])

    # confidence percentage,save three decimal places
    confidence = '%.3f' % (confidence * 100)

    return class_, confidence


def predict_and_show_one_img(img, model, classes):
    """get model output of one image

    Arguments:
        img: image ndarray
        model: keras trained model
    Returns:
        class_name: class name
        confidence: class confidence
    """
    class_, confidence = predict_class_and_confidence(img, model)
    class_name = classes[int(class_)]

    img = Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    draw = ImageDraw.Draw(img)
    font_text = ImageFont.truetype("yy.ttf", 60, encoding="utf-8")
    draw.text((5, 5), class_name + ' %' +
              str(confidence), (0, 255, 0), font=font_text)
    print('class name:', class_name, '%', str(confidence))

    img = cv2.cvtColor(np.asarray(img), cv2.COLOR_RGB2BGR)

    # cv2.namedWindow('img', 0)
    # cv2.resizeWindow('img',window_width,window_height)

    # show image
    cv2.imshow('img', img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

    return class_name, confidence


class MyWindow(QMainWindow):
    """
    customized Qt window
    """

    def __init__(self):
        super().__init__()
        self.setGeometry(500, 300, cam_width, cam_height + 150)
        self.setFixedSize(cam_width, cam_height + 150)
        self.setWindowTitle('Food Recogintion System')

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
        self.check_pixmap = None
        self.fram = None

        check_button = QPushButton("结算", self)
        check_button.move(450, cam_height + 50)
        check_button.resize(130, 40)
        check_button.clicked.connect(self.predict_one_img)
        # check_button.clicked.connect(self.check)  # check button bind check function

        confirm_button = QPushButton("确定", self)
        confirm_button.move(600, cam_height + 50)
        confirm_button.resize(130, 40)
        confirm_button.clicked.connect(self.confirm)  # Ok Button

        self.classes = []
        self.prices = {}

        # load classes and price
        with open('classes.txt', 'r', encoding='utf-8') as f:
            lines = f.readlines()
            for line in lines:
                class_ = line[:-1]
                self.classes.append(class_)

        with open('price.txt', 'r', encoding='utf-8') as f:
            lines = f.readlines()
            for line in lines:
                line = line[:-1].split(' ')
                class_, price = line[0], line[1]
                self.prices[class_] = price

        print('classes:', self.classes)
        print('prices:', self.prices)

        # load model
        self.model = load_model('model.h5')
        print('Model loading complete!')

        # camera init
        self.cap = cv2.VideoCapture(0)
        self.cap.set(3, cam_width)
        self.cap.set(4, cam_height)

        self._timer = QTimer(self)
        self._timer.timeout.connect(self.update)
        # self._timer.start(33)  # 30fps

        # self.setCentralWidget(self.img_label)
        self.show()

    def predict_one_img(self):
        img = cv2.imread('test.jpg')
        img = cv2.resize(img, (cam_width, cam_height))

        class_, confidence = predict_class_and_confidence(img, self.model)
        class_name = self.classes[int(class_)]

        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = Image.fromarray(img)
        draw = ImageDraw.Draw(img)
        font_text = ImageFont.truetype("yy.ttf", 30, encoding="utf-8")
        draw.text((5, 5), class_name + str(confidence) +
                  '%', (0, 255, 0), font=font_text)
        img = cv2.cvtColor(np.asarray(img), cv2.COLOR_RGB2BGR)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        print('class name:', class_name, '%', str(confidence))

        h, w = img.shape[:2]
        img = QImage(img, w, h, QImage.Format_RGB888)
        img = QPixmap.fromImage(img)
        self.check_pixmap = img
        self.isChecking = True
        self.dish_label.setText("菜品名称：" + class_name)
        self.price_label.setText("金额：" + self.prices[class_name] + "元")
        self.img_label.setPixmap(img)
        self.img_label.setScaledContents(True)  # self adaption

    def update(self):
        # get camera frame and convert to pixmap to show on label
        # checking status, stop updating pixmap
        if self.isChecking:
            self.img_label.setPixmap(self.check_pixmap)
            # self.img_label.setScaledContents(True)  # self adaption
            return

        ret, self.fram = self.cap.read()  # read camera frame
        print('fram shape:', self.fram.shape)
        fram = cv2.cvtColor(self.fram, cv2.COLOR_BGR2RGB)
        h, w = fram.shape[:2]
        img = QImage(fram, w, h, QImage.Format_RGB888)
        img = QPixmap.fromImage(img)
        self.img_label.setPixmap(img)  # show on img label
        self.img_label.setScaledContents(True)  # self adaption

    def check(self):
        # check function
        if self.isChecking:
            return
        fram = self.fram
        class_, confidence = predict_class_and_confidence(fram, self.model)
        class_name = self.classes[int(class_)]

        img = cv2.cvtColor(fram, cv2.COLOR_BGR2RGB)
        img = Image.fromarray(img)

        draw = ImageDraw.Draw(img)
        font_text = ImageFont.truetype("yy.ttf", 30, encoding="utf-8")
        draw.text((5, 5), class_name + ' %' +
                  str(confidence), (0, 255, 0), font=font_text)

        img = cv2.cvtColor(np.asarray(img), cv2.COLOR_RGB2BGR)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        print('class name:', class_name, '%', str(confidence))

        h, w = img.shape[:2]
        img = QImage(img, w, h, QImage.Format_RGB888)
        img = QPixmap.fromImage(img)
        self.check_pixmap = img
        self.isChecking = True
        self.dish_label.setText("菜品名称：" + class_name)
        self.price_label.setText("金额：" + self.prices[class_name] + "元")

    def confirm(self):
        self.isChecking = False
        self.dish_label.setText("菜品名称：")
        self.price_label.setText("金额：")


def cv_loop():
    # loop get camera frame and show on window
    
    # load classes
    classes = []
    with open('classes.txt', 'r', encoding='utf-8') as f:
        lines = f.readlines()
        for line in lines:
            class_ = line[:-1]
            print(class_)
            classes.append(class_)

    # load model
    model = load_model('model.h5')

    cap = cv2.VideoCapture(0)
    cap.set(3, cam_width)
    cap.set(4, cam_height)

    # main loop
    while True:
        ret, fram = cap.read()
        if not ret:
            continue
        print('frame shape:', fram.shape)
        class_, confidence = predict_class_and_confidence(fram, model)
        if not class_ == -1 and confidence == -1:
            break
        class_name = classes[int(class_)]

        img = Image.fromarray(cv2.cvtColor(fram, cv2.COLOR_BGR2RGB))
        draw = ImageDraw.Draw(img)
        font_text = ImageFont.truetype("yy.ttf", 60, encoding="utf-8")
        draw.text((5, 5), class_name + ' %' +
                  str(confidence), (0, 255, 0), font=font_text)
        fram = cv2.cvtColor(np.asarray(img), cv2.COLOR_RGB2BGR)
        # cv2.putText(fram, category_name+' %'+str(confidence), (0,70), cv2.FONT_HERSHEY_COMPLEX, 3, (0, 255, 0), 6)

        print(class_name, '%', str(confidence))
        cv2.namedWindow('fram', 0)
        cv2.resizeWindow('fram', window_width, window_height)

        cv2.imshow('fram', fram)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    cap.release()
    cv2.destroyAllWindows()


def main():
    app = QApplication(sys.argv)
    window = MyWindow()
    sys.exit(app.exec_())


if __name__ == '__main__':
    main()
