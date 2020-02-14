import cv2
import sys
import keras
from keras.models import load_model
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation, Flatten
from keras.layers import Conv2D, MaxPooling2D
from keras.optimizers import SGD  
from keras.utils import np_utils
from keras import backend as K
import h5py
import os
import numpy as np
np.random.seed(1337)
from PIL import Image,ImageDraw,ImageFont
import wx
import threading
from PyQt5.QtWidgets import QWidget,QPushButton,QApplication,QLabel,QMainWindow
from PyQt5.QtGui import QPixmap,QImage,QFont
from PyQt5.QtCore import Qt,QTimer
#输入图像维度
width=200
height=200  
img_size=width*height

cam_width=800
cam_height=600

window_width=1600
window_height=1200

def drawGrid(img):
    img_width=img.shape[1]
    img_height=img.shape[0]
    step=int(img_width/12)
    for i in range(step,img_height,step):
        cv2.line(img,(0,i),(img_width,i),(0,255,0),3)
    for j in range(step,img_width,step):
        cv2.line(img,(j,0),(j,img_height),(0,255,0),3)
    return img



def sliding_window(img,length,model):
    img_width=img.shape[1]
    img_height=img.shape[0]
    step=int(img_width/12)
    slid_step=step*length
    for x in range(0,img_width-slid_step,step):
        for y in range(0,img_height-slid_step,step):
            f=img[y:y+slid_step,x:x+slid_step]
            clas,confidence=get_class_and_confidence(f,model)
            print(str(int(x/step))+' '+str(int(y/step))+str(clas)+str(confidence))



def get_class_and_confidence(img,model):
    test_data=np.empty((1,img_size*3))
    try:
        img=cv2.resize(img, (width, height))
    except:
        print('resize error!')
        return -1,-1
    #img = cv2.cvtColor(img,cv2.COLOR_BGR2RGB)
    img_ndarray=np.asarray(img,dtype='float64')/255
    test_data = np.ndarray.flatten(img_ndarray)
    test_data = test_data.astype('float32')

    if K.image_data_format() == 'channels_first':
        test_data = test_data.reshape(1, 3, height, width)
    else:
        test_data = test_data.reshape(1, height, width, 3)
    predict=model.predict(test_data)
    clas=np.argmax(predict, axis=1) 
    confidence=float(predict[0][clas])
    confidence='%.3f'%(confidence*100) #置信度转化为百分比，保留3位小数 
    return clas,confidence


class mainFrame(wx.Frame):
    def __init__(self,title):
        size=window_width,window_height
        wx.Frame.__init__(self,None,title=title,pos=wx.DefaultPosition,size=(600,600))
        self.panel = wx.Panel(self)
        self.Center()
        self.image_cover=wx.Image('fqcd.jpg',wx.BITMAP_TYPE_ANY).Scale(350,300)

        self.staticbitmap=wx.StaticBitmap(self.panel,-1,wx.Bitmap(self.image_cover))
        start_button = wx.Button(self.panel,label='Start')
        close_button = wx.Button(self.panel,label='Close')

        self.category=[]
        self.path='data'
        with open('classes.txt','r') as f:
            classes=f.readlines()
            for c in classes:
                c=c[:-1]
                print(c)

        self.model=load_model('model.h5')
        self.cap=cv2.VideoCapture(0)
        self.cap.set(3,cam_width) 
        self.cap.set(4,cam_height)
        update_thread=threading.Thread(target=self.update,args=())
        update_thread.start()
        self.Show(True)


    def update(self):
        while(True):
            ret,fram=self.cap.read()
            if(ret!=True):
                continue
            print(fram.shape)
            fram=cv2.cvtColor(fram, cv2.COLOR_BGR2RGB)
            '''clas,confidence=get_class_and_confidence(fram,self.model)
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
                #break'''
            h,w=fram.shape[:2]
            fram=wx.Bitmap.FromBuffer(w,h,fram)
            self.staticbitmap.SetBitmap(fram)
            #self.staticbitmap.Refresh()
        self.cap.release()


def forecast_one_img(img_path):
    category=[]
    path='data'
    with open('classes.txt','r') as f:
        classes=f.readlines()
        for c in classes:
            c=c[:-1]
            print(c)
            category.append(c)

    model = load_model('model.h5')
    img=cv2.imread(img_path)
    clas,confidence=get_class_and_confidence(img,model)
    category_name=category[int(clas)]

    img = Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    draw=ImageDraw.Draw(img)
    fontText = ImageFont.truetype("yy.ttf", 60, encoding="utf-8")
    draw.text((5,5),category_name+' %'+str(confidence),(0,255,0),font=fontText)
    img=cv2.cvtColor(np.asarray(img), cv2.COLOR_RGB2BGR)

    print(category_name+'%'+str(confidence))
    #cv2.namedWindow('img', 0)
    #cv2.resizeWindow('img',window_width,window_height)

    cv2.imshow('img', img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


class my(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setGeometry(500,300,cam_width,cam_height+150)
        self.setFixedSize(cam_width,cam_height+150)
        self.setWindowTitle('菜品识别系统')

        self.img_label=QLabel(self)
        self.img_label.setGeometry(0,0,cam_width,cam_height)

        self.dish_label=QLabel(self)
        self.dish_label.move(70,cam_height+25)
        self.dish_label.resize(250,25)
        self.dish_label.setText("菜品名称：")
        self.dish_label.setFont(QFont("Roman times",18,QFont.Bold))

        self.price_label=QLabel(self)
        self.price_label.move(70,cam_height+70)
        self.price_label.resize(250,25)
        self.price_label.setText("金额：")
        self.price_label.setFont(QFont("Roman times",18,QFont.Bold))

        self.isChecking=False

        check_button=QPushButton("结算",self)
        check_button.move(450,cam_height+50)
        check_button.resize(130,40)
        check_button.clicked.connect(self.detect_one)

        ok_button=QPushButton("确定",self)
        ok_button.move(600,cam_height+50)
        ok_button.resize(130,40)
        ok_button.clicked.connect(self.ok)

        self.category=[]
        self.path='data'
        with open('classes.txt','r') as f:
            classes=f.readlines()
            for c in classes:
                c=c[:-1]
                print(c)
                self.category.append(c)

        self.price={}
        with open('price.txt','r') as f:
            prices=f.readlines()
            for price in prices:
                price=price[:-1]
                price=price.split(' ')
                c=price[0]
                p=price[1]
                self.price[c]=p
               
        self.model=load_model('model.h5')
        self.cap=cv2.VideoCapture(0)
        self.cap.set(3,cam_width) 
        self.cap.set(4,cam_height)

        self._timer=QTimer(self)
        self._timer.timeout.connect(self.update)
        #self._timer.start(33)

        #self.setCentralWidget(self.img_label)
        self.show()

    def detect_one(self):
        img=cv2.imread('fqcd.jpg')
        img=cv2.resize(img,(cam_width,cam_height))
        #img=cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        clas,confidence=get_class_and_confidence(img,self.model)
        category_name=self.category[int(clas)]

        img=cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = Image.fromarray(img)
        draw=ImageDraw.Draw(img)
        fontText = ImageFont.truetype("yy.ttf", 30, encoding="utf-8")
        draw.text((5,5),category_name+str(confidence)+'%',(0,255,0),font=fontText)
        img=cv2.cvtColor(np.asarray(img), cv2.COLOR_RGB2BGR)
        img=cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        print(category_name+'%'+str(confidence))

        h,w=img.shape[:2]
        img=QImage(img,w,h,QImage.Format_RGB888)
        img=QPixmap.fromImage(img)
        self.check_pixmap=img
        self.isChecking=True
        self.dish_label.setText("菜品名称："+category_name)
        self.price_label.setText("金额："+self.price[category_name]+"元")
        self.img_label.setPixmap(img)
        self.img_label.setScaledContents(True) #自适应大小


    def update(self):
        if self.isChecking:
            self.img_label.setPixmap(self.check_pixmap)
            #self.img_label.setScaledContents(True) #自适应大小
            return
        ret,self.fram=self.cap.read()
        print(self.fram.shape)
        fram=cv2.cvtColor(self.fram,cv2.COLOR_BGR2RGB)
        h,w=fram.shape[:2]
        img=QImage(fram,w,h,QImage.Format_RGB888)
        img=QPixmap.fromImage(img)
        self.img_label.setPixmap(img)
        self.img_label.setScaledContents(True) #自适应大小

    def check(self):
        if self.isChecking:
            return
        fram=self.fram
        clas,confidence=get_class_and_confidence(fram,self.model)
        category_name=self.category[int(clas)]

        img = cv2.cvtColor(fram,cv2.COLOR_BGR2RGB)
        img = Image.fromarray(img)
        draw=ImageDraw.Draw(img)
        fontText = ImageFont.truetype("yy.ttf", 30, encoding="utf-8")
        draw.text((5,5),category_name+' %'+str(confidence),(0,255,0),font=fontText)
        img=cv2.cvtColor(np.asarray(img), cv2.COLOR_RGB2BGR)
        img=cv2.cvtColor(img,cv2.COLOR_BGR2RGB)
    
        print(category_name+'%'+str(confidence))
        h,w=img.shape[:2]
        img=QImage(img,w,h,QImage.Format_RGB888)
        img=QPixmap.fromImage(img)
        self.check_pixmap=img
        self.isChecking=True
        self.dish_label.setText("菜品名称："+category_name)
        self.price_label.setText("金额："+self.price[category_name]+"元")


    def ok(self):
        self.isChecking=False
        self.dish_label.setText("菜品名称：")
        self.price_label.setText("金额：")



def main():
    category=[]
    path='data'
    with open('classes.txt','r') as f:
        classes=f.readlines()
        for c in classes:
            c=c[:-1]
            print(c)
            category.append(c)

    model = load_model('model.h5')

    cap=cv2.VideoCapture(0)
    cap.set(3,cam_width) 
    cap.set(4,cam_height)
    while(True):
        ret,fram=cap.read()
        if(ret!=True):
            continue
        print(fram.shape)
        clas,confidence=get_class_and_confidence(fram,model)
        if not clas==-1 and confidence==-1:
            break
        category_name=category[int(clas)]

        img = Image.fromarray(cv2.cvtColor(fram, cv2.COLOR_BGR2RGB))
        draw=ImageDraw.Draw(img)
        fontText = ImageFont.truetype("yy.ttf", 60, encoding="utf-8")
        draw.text((5,5),category_name+' %'+str(confidence),(0,255,0),font=fontText)
        fram=cv2.cvtColor(np.asarray(img), cv2.COLOR_RGB2BGR)
        #cv2.putText(fram, category_name+' %'+str(confidence), (0,70), cv2.FONT_HERSHEY_COMPLEX, 3, (0, 255, 0), 6)
        
        print(category_name+'%'+str(confidence))
        cv2.namedWindow('fram', 0)
        cv2.resizeWindow('fram',window_width,window_height)
        #fram=drawGrid(fram)
        cv2.imshow('fram', fram)
        if cv2.waitKey(1) & 0xFF==ord('q'):
            break
    cap.release()
    cv2.destroyAllWindows()


if __name__ == '__main__':
    '''
    app=wx.App(False)
    fram=mainFrame('菜品识别系统')
    app.MainLoop()
    '''
    #main()
    app=QApplication(sys.argv)
    my=my()
    sys.exit(app.exec_())

    #forecast_one_img('8.jpg')
