# @Author: Ivan
# @LastEdit: 2020/9/23
import threading
import cv2  # install
from keras.models import load_model
from keras import backend as K
import numpy as np  # install
from PIL import Image, ImageDraw, ImageFont  # install
import wx  # install
from predict import predict_class_name_and_confidence
from predict import load_prices

width, height, depth = 200, 200, 3

cam_width, cam_height = 800, 600

window_width, window_height = 1600, 1200


def get_class_and_confidence(img, model):
    if depth == 1:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    # test_data = np.empty((1, width*height*3))
    try:
        img = cv2.resize(img, (width, height))
    except:
        print('resize error!')
        return -1, -1
    # img = cv2.cvtColor(img,cv2.COLOR_BGR2RGB)
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
    # confidence percentage,save three decimal places
    confidence = '%.2f' % (confidence * 100)
    return class_, confidence


def predict_one_img(img_path):
    model = load_model('model.h5')
    img = cv2.imread(img_path)
    class_, confidence = get_class_name_and_confidence(img, model)

    img = Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    draw = ImageDraw.Draw(img)
    font_text = ImageFont.truetype("simsun.ttc", 60, encoding="utf-8")
    draw.text((5, 5), class_name + ' %' +
              str(confidence), (0, 255, 0), font=font_text)
    img = cv2.cvtColor(np.asarray(img), cv2.COLOR_RGB2BGR)

    print('Class name:', class_name, 'confidence:', str(confidence)+'%')
    # cv2.namedWindow('img', 0)
    # cv2.resizeWindow('img',window_width,window_height)

    cv2.imshow('img', img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


class MainFrame(wx.Frame):
    def __init__(self, title):
        size = window_width, window_height
        wx.Frame.__init__(self, None, title=title,
                          pos=wx.DefaultPosition, size=(600, 600))
        self.panel = wx.Panel(self)
        self.Center()
        self.image_cover = wx.Image(
            'test.jpg', wx.BITMAP_TYPE_ANY).Scale(350, 300)

        self.staticbitmap = wx.StaticBitmap(
            self.panel, -1, wx.Bitmap(self.image_cover))
        start_button = wx.Button(self.panel, label='Start')
        close_button = wx.Button(self.panel, label='Close')

        self.model = load_model('model.h5')
        self.cap = cv2.VideoCapture(0)
        self.cap.set(3, cam_width)
        self.cap.set(4, cam_height)
        update_thread = threading.Thread(target=self.update, args=())
        update_thread.start()
        self.Show(True)

    def update(self):
        while True:
            ret, frame = self.cap.read()
            if not ret:
                continue
            print(frame.shape)
            # frame = cv2.cvtColor(fram, cv2.COLOR_BGR2RGB)
            class_name, confidence = get_class_name_and_confidence(
                frame, self.model)

            img = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
            draw = ImageDraw.Draw(img)
            font_text = ImageFont.truetype("simsun.ttc", 60, encoding="utf-8")
            draw.text((5, 5), class_name + str(confidence) +
                      '%', (0, 255, 0), font=font_text)
            frame = cv2.cvtColor(np.asarray(img), cv2.COLOR_RGB2BGR)
            # cv2.putText(frame, category_name+' %'+str(confidence),
            #             (0, 70), cv2.FONT_HERSHEY_COMPLEX, 3, (0, 255, 0), 6)

            cv2.namedWindow('frame', 0)
            cv2.resizeWindow('frame', window_width, window_height)

            cv2.imshow('frame', frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

            h, w = frame.shape[:2]
            frame = wx.Bitmap.FromBuffer(w, h, frame)
            self.staticbitmap.SetBitmap(frame)
            # self.staticbitmap.Refresh()
        self.cap.release()


def main():
    app = wx.App(False)
    frame = MainFrame('Food Recognition System')
    app.MainLoop()
