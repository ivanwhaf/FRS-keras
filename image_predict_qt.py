# @Author: Ivan
# @LastEdit: 2020/8/13
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


def get_class_and_confidence(img, model):
    """get image's class and confidence according to trained model

    Args:
        img: prediction image
        model: keras model
    Returns:
        class_: class index
        confidence: class's confidence
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

    preds = model.predict(test_data)
    class_ = np.argmax(preds[0])
    confidence = float(preds[0][class_])
    confidence = '%.3f' % (confidence * 100)  # confidence percentage,save three decimal places

    return class_, confidence


def predict_one_img(img_path):
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
        check_button.clicked.connect(self.predict_one)
        # check_button.clicked.connect(self.check)  # check button bind check function

        ok_button = QPushButton("确定", self)
        ok_button.move(600, cam_height + 50)
        ok_button.resize(130, 40)
        ok_button.clicked.connect(self.ok)  # Ok Button

        self.classes = []
        self.prices = {}

        with open('classes.txt', 'r') as f:
            lines = f.readlines()
            for line in lines:
                class_ = line[:-1]
                self.classes.append(class_)

        with open('price.txt', 'r') as f:
            lines = f.readlines()
            for line in lines:
                line = line[:-1].split(' ')
                class_, price = line[0], line[1]
                self.prices[class_] = price

        print(self.classes)
        print(self.prices)

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
        img = cv2.imread('test.jpg')
        img = cv2.resize(img, (cam_width, cam_height))
        # img=cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        class_, confidence = get_class_and_confidence(img, self.model)
        class_name = self.classes[int(class_)]

        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = Image.fromarray(img)
        draw = ImageDraw.Draw(img)
        font_text = ImageFont.truetype("yy.ttf", 30, encoding="utf-8")
        draw.text((5, 5), class_name + str(confidence) +
                  '%', (0, 255, 0), font=font_text)
        img = cv2.cvtColor(np.asarray(img), cv2.COLOR_RGB2BGR)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        print(class_name, '%', str(confidence))

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
        if self.isChecking:
            self.img_label.setPixmap(self.check_pixmap)
            # self.img_label.setScaledContents(True) # self adaption
            return
        ret, self.fram = self.cap.read()  # read camera frame
        print('fram shape', self.fram.shape)
        fram = cv2.cvtColor(self.fram, cv2.COLOR_BGR2RGB)
        h, w = fram.shape[:2]
        img = QImage(fram, w, h, QImage.Format_RGB888)
        img = QPixmap.fromImage(img)
        self.img_label.setPixmap(img)  # show on img label
        self.img_label.setScaledContents(True)  # self adaption

    def check(self):
        # checks function
        if self.isChecking:
            return
        fram = self.fram
        class_, confidence = get_class_and_confidence(fram, self.model)
        class_name = self.classes[int(class_)]

        img = cv2.cvtColor(fram, cv2.COLOR_BGR2RGB)
        img = Image.fromarray(img)

        draw = ImageDraw.Draw(img)
        font_text = ImageFont.truetype("yy.ttf", 30, encoding="utf-8")
        draw.text((5, 5), class_name + ' %' +
                  str(confidence), (0, 255, 0), font=font_text)

        img = cv2.cvtColor(np.asarray(img), cv2.COLOR_RGB2BGR)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        print(class_name, '%', str(confidence))

        h, w = img.shape[:2]
        img = QImage(img, w, h, QImage.Format_RGB888)
        img = QPixmap.fromImage(img)
        self.check_pixmap = img
        self.isChecking = True
        self.dish_label.setText("菜品名称：" + class_name)
        self.price_label.setText("金额：" + self.prices[class_name] + "元")

    def ok(self):
        self.isChecking = False
        self.dish_label.setText("菜品名称：")
        self.price_label.setText("金额：")


def cv_loop():
    classes = []
    with open('classes.txt', 'r') as f:
        lines = f.readlines()
        for line in lines:
            class_ = line[:-1]
            print(class_)
            classes.append(class_)

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
