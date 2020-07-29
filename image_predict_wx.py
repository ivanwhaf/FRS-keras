import os
import threading
import cv2  # install
import keras  # install
from keras.models import load_model
from keras import backend as K
import h5py  # install
import numpy as np  # install
from PIL import Image, ImageDraw, ImageFont  # install
import wx  # install

np.random.seed(1337)

# 图像高宽
width, height, depth = 200, 200, 3

# 摄像头高宽
cam_width, cam_height = 800, 600

# 窗口大小
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
    img_ndarray = np.asarray(img, dtype='float64')/255
    test_data = np.ndarray.flatten(img_ndarray)
    test_data = test_data.astype('float32')

    if K.image_data_format() == 'channels_first':
        test_data = test_data.reshape(1, depth, height, width)
    else:
        test_data = test_data.reshape(1, height, width, depth)

    preds = model.predict(tesst_data)
    class_ = np.argmax(preds[0], axis=1)
    confidence = float(preds[0][class_])
    confidence = '%.3f' % (confidence*100)  # 置信度转化为百分比，保留3位小数
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
    fontText = ImageFont.truetype("yy.ttf", 60, encoding="utf-8")
    draw.text((5, 5), category_name+' %' +
              str(confidence), (0, 255, 0), font=fontText)
    img = cv2.cvtColor(np.asarray(img), cv2.COLOR_RGB2BGR)

    print(category_name, '%', str(confidence))
    #cv2.namedWindow('img', 0)
    # cv2.resizeWindow('img',window_width,window_height)

    cv2.imshow('img', img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


class mainFrame(wx.Frame):
    def __init__(self, title):
        size = window_width, window_height
        wx.Frame.__init__(self, None, title=title,
                          pos=wx.DefaultPosition, size=(600, 600))
        self.panel = wx.Panel(self)
        self.Center()
        self.image_cover = wx.Image(
            'fqcd.jpg', wx.BITMAP_TYPE_ANY).Scale(350, 300)

        self.staticbitmap = wx.StaticBitmap(
            self.panel, -1, wx.Bitmap(self.image_cover))
        start_button = wx.Button(self.panel, label='Start')
        close_button = wx.Button(self.panel, label='Close')

        self.category = []
        with open('classes.txt', 'r') as f:
            lines = f.readlines()
            for line in lines:
                class_ = line[:-1]
                print(class_)
                self.category.append(class_)

        self.model = load_model('model.h5')
        self.cap = cv2.VideoCapture(0)
        self.cap.set(3, cam_width)
        self.cap.set(4, cam_height)
        update_thread = threading.Thread(target=self.update, args=())
        update_thread.start()
        self.Show(True)

    def update(self):
        while(True):
            ret, fram = self.cap.read()
            if(ret != True):
                continue
            print(fram.shape)
            fram = cv2.cvtColor(fram, cv2.COLOR_BGR2RGB)
            '''
            clas,confidence=get_class_and_confidence(fram,self.model)
            if not clas==-1 and confidence==-1:
                break
            category_name=self.category[int(clas)]

            img = Image.fromarray(cv2.cvtColor(fram, cv2.COLOR_BGR2RGB))
            draw=ImageDraw.Draw(img)
            fontText = ImageFont.truetype("yy.ttf", 60, encoding="utf-8")
            draw.text((5,5),category_name+' %'+str(confidence),(0,255,0),font=fontText)
            fram=cv2.cvtColor(np.asarray(img), cv2.COLOR_RGB2BGR)
            #cv2.putText(fram, category_name+' %'+str(confidence), (0,70), cv2.FONT_HERSHEY_COMPLEX, 3, (0, 255, 0), 6)
        
            print(category_name+'%'+str(confidence))
            #cv2.namedWindow('fram', 0)
            #cv2.resizeWindow('fram',window_width,window_height)

            #cv2.imshow('fram', fram)
            #if cv2.waitKey(1) & 0xFF==ord('q'):
                #break
            '''
            h, w = fram.shape[:2]
            fram = wx.Bitmap.FromBuffer(w, h, fram)
            self.staticbitmap.SetBitmap(fram)
            # self.staticbitmap.Refresh()
        self.cap.release()


def main():
    app = wx.App(False)
    fram = mainFrame('菜品识别系统')
    app.MainLoop()
