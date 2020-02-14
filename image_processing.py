# Author:Ivan
import os
import cv2
import random
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt

def load_img(path,category_number,number_per_category,width,height,depth,train_per_category,valid_per_category,test_per_category):
    '''
    加载图片函数
    1.图片文件夹path目录下必须包含以每一类图片为一个文件夹的子文件夹
    如dataset文件夹下包含c1,c2,c3三个类别的子文件夹
    每个子文件夹包含相应图片,如c1文件夹下包含1.jpg,2.jpg
    2.文件夹路径名及所有文件名必须是英文
    Args:
        path:图片文件夹路径
        category_number:图片类别数量
        number_per_category:每一类别的图片数量
        width:图片宽
        height:图片高
        train_data_number:训练图片数量
        valid_data_number:验证图片数量
        test_data_number:测试图片数量
        need_normalize:图片是否需要归一化处理,默认不需要
    Returns:
        rval包含3个元组和一个类别列表
           (train_data, train_label)训练数据和标签
           (valid_data, valid_label)验证数据和标签
           (test_data, test_label)  测试数据和标签
           category 类别列表
    '''
    number=category_number*number_per_category #图片总数
    n=0 #images[]数组下标
    c=0 #已读取的类别数量
    category=[]
    img_size=width*height #图片大小 宽*高
    images=np.empty((number,img_size*depth)) #图片集
    img_categories=os.listdir(path)
    print('image categories:')
    for img_category in img_categories:
        c=c+1
        if c>category_number:
            break
        img_category_path=os.path.join(path,img_category)
        print('<'+str(img_category)+'>')
        category.append(img_category)
        if os.path.isdir(img_category_path):
            imgs=os.listdir(img_category_path)
            m=0 #每类已经读取的图片数量
            im=[] #暂存每类的图片数量最后打乱后加入images数组
            for img in imgs:
                if m>=number_per_category:#大于设定的每类图片数量
                    break
                img_path=os.path.join(img_category_path,img)

                if depth==3:
                    #image=cv2.imread(img_path,cv2.IMREAD_COLOR)
                    image= cv2.imdecode(np.fromfile(img_path,dtype=np.uint8),cv2.IMREAD_COLOR)
                    #image=cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
                    #image=Image.open(img_path)
                elif depth==1:
                    image=cv2.imread(img_path)
                    image=cv2.cvtColor(image,cv2.COLOR_BGR2GRAY) #转换为灰度图
                    #image=cv2.imread(img_path,cv2.IMREAD_GRAYSCALE) #读取灰度图
                try:
                    #image=cv2.resize(image,(width,height),interpolation=cv2.INTER_CUBIC)#调整图片大小
                    image=cv2.resize(image,(width,height))
                except:
                    print(img_path+' resize error!')
                    #os.remove(img_path)
                    #print(img_path+' has been deleted.')
                    continue
      
                image_ndarray=np.asarray(image,dtype='float64')/255
                im.append(np.ndarray.flatten(image_ndarray))
                #images[n]=np.ndarray.flatten(image_ndarray)
                m=m+1
                n=n+1
            if(m<number_per_category):
                print('image number unequal!'+str(m))
                exit()

            random.shuffle(im)
            for i in range(n-number_per_category,n):
                images[i]=im[i%number_per_category]


    label=np.empty(number)
    for i in range(category_number):
        label[i*number_per_category:i*number_per_category+number_per_category]=i
    label=label.astype(np.int)

    train_data_number=train_per_category*category_number
    valid_data_number=valid_per_category*category_number
    test_data_number=test_per_category*category_number

    train_data = np.empty((train_data_number, img_size*depth))  
    train_label = np.empty(train_data_number)  
    valid_data = np.empty((valid_data_number, img_size*depth))  
    valid_label = np.empty(valid_data_number)  
    test_data = np.empty((test_data_number, img_size*depth))  
    test_label = np.empty(test_data_number)   

    for i in range(category_number):
        train_data [i*train_per_category : i*train_per_category+train_per_category] = images[i*number_per_category : i*number_per_category+train_per_category] # 训练集数据
        train_label[i*train_per_category : i*train_per_category+train_per_category] = label [i*number_per_category : i*number_per_category+train_per_category]  # 训练集标签
        valid_data [i*valid_per_category : i*valid_per_category+valid_per_category] = images[i*number_per_category+train_per_category : i*number_per_category+train_per_category+valid_per_category] # 验证集数据
        valid_label[i*valid_per_category : i*valid_per_category+valid_per_category] = label [i*number_per_category+train_per_category : i*number_per_category+train_per_category+valid_per_category] # 验证集标签
        test_data  [i*test_per_category : i*test_per_category+test_per_category] = images[i*number_per_category+train_per_category+valid_per_category : i*number_per_category+train_per_category+valid_per_category+test_per_category] # 测试集数据
        test_label [i*test_per_category : i*test_per_category+test_per_category] = label [i*number_per_category+train_per_category+valid_per_category : i*number_per_category+train_per_category+valid_per_category+test_per_category]   # 测试集标签

    #print('train_label:')
    #print(train_label)
    #print('valid_label:')
    #print(valid_label)
    #print('test_label:')
    #print(test_label)

    train_data=train_data.astype('float32')
    valid_data = valid_data.astype('float32')
    test_data = test_data.astype('float32')
    rval = [(train_data, train_label), (valid_data, valid_label), (test_data, test_label),category]  
    return rval



def img_normalize(path,width,height):
    '''
    图片归一化处理函数
    将图片裁剪成固定大小并转换为灰度图
    1.图片文件夹path目录下必须包含以每一类图片为一个文件夹的子文件夹
    如dataset文件夹下包含c1,c2,c3三个类别的子文件夹
    每个子文件夹包含相应图片,如c1文件夹下包含1.jpg,2.jpg
    2.文件夹路径名及所有文件名必须是英文
    3.文件所在根目录下必须包含haarcascade_frontalface_default.xml文件
    Args:
        path:图片文件夹路径
        width:要调整的图片的宽
        height:要调整的图片的长
    Returns:
        无
    '''
    w=width
    h=height
    img_categories=os.listdir(path)
    for img_category in img_categories:
        img_category_path=os.path.join(path,img_category)
        if os.path.isdir(img_category_path):
            imgs=os.listdir(img_category_path)
            for img in imgs:
                img_path=os.path.join(img_category_path,img)
                image=cv2.imread(img_path)
                try:
                    image=cv2.resize(image,(width,height),interpolation=cv2.INTER_CUBIC)#调整图片大小
                except:
                    print(img_path+' resize error!')
                    os.remove(img_path)
                    print(img_path+' has been deleted.')
                    continue
                image=cv2.cvtColor(image,cv2.COLOR_BGR2GRAY)#转换为灰度图
                cv2.imwrite(img_path,image)
                print(img_path+' normalized successfully!')
    print('all images normalized successfully!')



def img_rename(path):
    '''
    图片排序重命名函数
    遍历文件夹将所有图片从1开始重命名,如从p1.jpg~p100.jpg
    Args:
        path:图片文件夹路径
    Returns:
        无
    '''
    categories=os.listdir(path)
    for category in categories:
        number=1
        category_path=os.path.join(path,category)
        imgs=os.listdir(category_path)
        for img in imgs:
            img_path=os.path.join(category_path,img)
            new_img_path=os.path.join(category_path,'p'+str(number)+'.jpg')
            os.rename(img_path,new_img_path)
            print(img_path+'----->'+new_img_path)
            number=number+1



def face_detect():
    '''
    人脸检测函数
    调用摄像头实时检测出人脸并用矩形框框出
    '''
    face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
    cap=cv2.VideoCapture(0)
    while(True):
        ret,fram=cap.read()  
        faces=face_cascade.detectMultiScale(fram,1.1,7)
        for x,y,w,h in faces:
            cv2.rectangle(fram,(x-5,y-25),(x+w,y+h),(0,255,0),2)
            cv2.imshow('fram',fram)



def cut_face(path):
    '''
    裁剪人脸函数
    检测出人脸后用新的人脸图片覆盖原图
    1.图片文件夹path目录下必须包含以每一类图片为一个文件夹的子文件夹
    如dataset文件夹下包含c1,c2,c3三个类别的子文件夹
    每个子文件夹包含相应图片,如c1文件夹下包含1.jpg,2.jpg
    2.文件夹路径名及所有文件名必须是英文
    3.文件所在根目录下必须包含haarcascade_frontalface_default.xml文件
    Args:
        path:图片文件夹路径
    Returns:
        无

    '''
    face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
    img_categories=os.listdir(path)
    for img_category in img_categories:
        img_category_path=os.path.join(path,img_category)
        if os.path.isdir(img_category_path):
            imgs=os.listdir(img_category_path)
            for img in imgs:
                img_path=os.path.join(img_category_path,img)
                face=cv2.imread(img_path)#读取图片
                faces=face_cascade.detectMultiScale(face,1.1,7)#检测人脸
                for x,y,w,h in faces[0]:
                    #cv2.rectangle(face,(x-5,y-25),(x+w,y+h),(0,255,0),2)
                    face=face[y-60:y+h+15,x:x+w]
                cv2.imwrite(img_path,face)#用裁剪后的人脸覆盖原图片
                print(img_path+'--cuting successfully')
    print('all faces cut successfully!')



def dHash(img):#差值hash算法
    img=cv2.resize(img,(9,8),interpolation=cv2.INTER_AREA)
    img=cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
    hash=''
    for i in range(8):
        for j in range(8):
            if img[i,j]>img[i,j+1]:
                hash=hash+'1'
            else:
                hash=hash+'0'
    print("dHash:"+str(hash))
    return hash



def aHash(img):#均值hash算法
    img=cv2.resize(img,(8,8))
    img=cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
    hash=''
    average=0
    for i in range(8):
        for j in range(8):
            average=average+img[i,j]
    average=average/64

    for i in range(8):
        for j in range(8):
            if(img[i,j]>average):
                hash=hash+'1'
            else:
                hash=hash+'0'
    print("aHash:"+str(hash))
    return hash



def pHash(img):#感知hash算法
    img=cv2.resize(img,(32,32),interpolation=cv2.INTER_AREA)
    img=cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
    hash=''
    mean=0.0
    h,w=img.shape[:2]
    vis0=np.zeros((h,w),np.float32)
    vis0[:h, :w] = img
    vis1=cv2.dct(vis0)
    for i in range(8):
        for j in range(8):
            mean+=vis1[i,j]
    mean=mean/64

    for i in range(8):
        for j in range(8):
            if(vis1[i,j]>=mean):
                hash=hash+'1'
            else:
                hash=hash+'0'
    print("pHash:"+str(hash))
    return hash



def hamming_distance(hash1,hash2):#计算两值的汉明距离
    hamming=0;
    for i in range(64):
        if(hash1[i]!=hash2[i]):
            hamming=hamming+1
    return hamming



def compare(img1,img2,func):#比较两图的汉明距离
    hamming=0
    if(func=='aHash'):
        hamming=hamming_distance(aHash(img1),aHash(img2))
    elif(func=='pHash'):
        hamming=hamming_distance(pHash(img1),pHash(img2))
    elif(func=='dHash'):
        hamming=hamming_distance(dHash(img1),dHash(img2))
    return hamming



'''
图像模糊测试函数
'''
def a():
    img=cv2.imread('1.jpg')
    img=cv2.resize(img,(800,1000),interpolation=cv2.INTER_AREA)
    gray=cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
    gradX = cv2.Sobel(gray, ddepth=cv2.CV_32F, dx=1, dy=0, ksize=-1)
    gradY = cv2.Sobel(gray, ddepth=cv2.CV_32F, dx=0, dy=1, ksize=-1)
    gradient = cv2.subtract(gradX, gradY)
    gradient = cv2.convertScaleAbs(gradient)

    gradient= cv2.blur(gradient,(3,3)) 
    ret, binary = cv2.threshold(gradient,127,255,cv2.THRESH_BINARY)
    cv2.imshow('img',binary)
    i=cv2.waitKey(0)



'''
cornerHarris角点检测函数
输入图像，并在原图像上画出角点
'''
def harris(img):
    #img=cv2.imread('6.jpg')
    #img=cv2.resize(img,(800,800),interpolation=cv2.INTER_AREA)
    gray=cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
    gray=cv2.GaussianBlur(gray,(5,5),0)
    gray=np.float32(gray)
    dst=cv2.cornerHarris(gray,2,3,0.04)
    dst=cv2.dilate(dst, None)
    img[dst>0.01*dst.max()]=[0,0,255]
    cv2.imshow('harris',img)
    #i=cv2.waitKey(0)



def drawFaces(img):
    #img=cv2.resize(img,(800,800),interpolation=cv2.INTER_AREA)
    gray=cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
    face_cascade=cv2.CascadeClassifier('D:/Python36/Lib/site-packages/cv2/data/haarcascade_frontalface_default.xml')
    faces=face_cascade.detectMultiScale(gray,1.3,5)
    for (x,y,w,h) in faces:
        img=cv2.rectangle(img,(x,y),(x+w,y+h),(0,0,255),2)
    cv2.imshow('img',img)




def sift_test(img1,img2):
    sift=cv2.xfeatures2d.SIFT_create()

    img1=cv2.resize(img1,(800,800),interpolation=cv2.INTER_AREA)
    gray1=cv2.cvtColor(img1,cv2.COLOR_BGR2GRAY)
    #gray1=cv2.GaussianBlur(gray1,(5,5),0)

    img2=cv2.resize(img2,(800,800),interpolation=cv2.INTER_AREA)
    gray2=cv2.cvtColor(img2,cv2.COLOR_BGR2GRAY)
    #gray2=cv2.GaussianBlur(gray2,(5,5),0)

    kp1,des1=sift.detectAndCompute(gray1,None)
    kp2,des2=sift.detectAndCompute(gray2,None)

    #img=cv2.drawKeypoints(img,kp,img,color=(255,0,255))
    #bf=cv2.BFMatcher(cv2.NORM_HAMMING,crossCheck=True)

    FLANN_INDEX_KDTREE = 0
    index_params = dict(algorithm = FLANN_INDEX_KDTREE, trees = 5)
    search_params = dict(checks=50)
    flann = cv2.FlannBasedMatcher(index_params,search_params)

    #bf=cv2.BFMatcher()
    #matches=bf.knnMatch(des1,des2,k=2)
    matches=flann.knnMatch(des1,des2,k=2)
    print('matches:'+str(len(matches)),end='')
    good = []
    for m,n in matches: 
        if m.distance < 0.7*n.distance:
             good.append([m])
    print('  good:'+str(len(good)))
    #img3 = cv2.drawMatchesKnn(img1,kp1,img2,kp2,matches,None, flags=2)
    #img3 = cv2.drawMatchesKnn(img1,kp1,img2,kp2,good,None, flags=2)
    #cv2.imshow('img',img3)
    #i=cv2.waitKey(0)
    cv2.imshow('img',img2)



def surf_test_fast(kp1,des1,img1,img2):
    surf = cv2.xfeatures2d.SURF_create(400)

    img2=cv2.resize(img2,(800,800),interpolation=cv2.INTER_AREA)
    gray2=cv2.cvtColor(img2,cv2.COLOR_BGR2GRAY)
    #gray2=cv2.GaussianBlur(gray2,(5,5),0)
    #surf.hessianThreshold = 500
    kp2,des2=surf.detectAndCompute(gray2,None)
    #print(len(des1))
    #img=cv2.drawKeypoints(img,kp,img,color=(255,0,255))
    #bf=cv2.BFMatcher(cv2.NORM_HAMMING,crossCheck=True)

    FLANN_INDEX_KDTREE=0
    index_params = dict(algorithm = FLANN_INDEX_KDTREE, trees = 5)
    search_params = dict(checks=50)
    flann = cv2.FlannBasedMatcher(index_params,search_params)

    #bf=cv2.BFMatcher()
    #matches=bf.knnMatch(des1,des2,k=2)
    try:
        matches=flann.knnMatch(np.asarray(des1,np.float32),np.asarray(des2,np.float32),k=2)
    except:
        return
    #matches=flann.knnMatch(des1,des2,k=2)
    print('matches:'+str(len(matches)),end='')
    good = []
    for m,n in matches: 
        if m.distance < 0.65*n.distance:
             good.append([m])
    print('  good:'+str(len(good)))
    #img3 = cv2.drawMatchesKnn(img1,kp1,img2,kp2,matches,None, flags=2)
    img3 = cv2.drawMatchesKnn(img1,kp1,gray2,kp2,good,None, flags=2)
    cv2.imshow('img3',img3)




def surf_test(img1,img2):
    surf = cv2.xfeatures2d.SURF_create()
    img1=cv2.resize(img1,(800,800),interpolation=cv2.INTER_AREA)
    gray1=cv2.cvtColor(img1,cv2.COLOR_BGR2GRAY)
    #gray1=cv2.GaussianBlur(gray1,(5,5),0)

    img2=cv2.resize(img2,(800,800),interpolation=cv2.INTER_AREA)
    gray2=cv2.cvtColor(img2,cv2.COLOR_BGR2GRAY)
    #gray2=cv2.GaussianBlur(gray2,(5,5),0)
    #surf.hessianThreshold = 500
    kp1,des1=surf.detectAndCompute(gray1,None)
    kp2,des2=surf.detectAndCompute(gray2,None)
    #print(len(des1))
    #img=cv2.drawKeypoints(img,kp,img,color=(255,0,255))
    #bf=cv2.BFMatcher(cv2.NORM_HAMMING,crossCheck=True)

    FLANN_INDEX_KDTREE=0
    index_params = dict(algorithm = FLANN_INDEX_KDTREE, trees = 5)
    search_params = dict(checks=50)
    flann = cv2.FlannBasedMatcher(index_params,search_params)

    #bf=cv2.BFMatcher()
    #matches=bf.knnMatch(des1,des2,k=2)
    try:
        matches=flann.knnMatch(np.asarray(des1,np.float32),np.asarray(des2,np.float32),k=2)
    except:
        return
    #matches=flann.knnMatch(des1,des2,k=2)
    print('matches:'+str(len(matches)),end='')
    good = []
    for m,n in matches: 
        if m.distance < 0.7*n.distance:
             good.append([m])
    print('  good:'+str(len(good)))
    #img3 = cv2.drawMatchesKnn(img1,kp1,img2,kp2,matches,None, flags=2)
    img3 = cv2.drawMatchesKnn(img1,kp1,img2,kp2,good,None, flags=2)
    cv2.imshow('img3',img3)
    #i=cv2.waitKey(0)



def orb_test(img1,img2):
    orb = cv2.ORB_create()

    img1=cv2.resize(img1,(800,800),interpolation=cv2.INTER_AREA)
    gray1=cv2.cvtColor(img1,cv2.COLOR_BGR2GRAY)
    gray1=cv2.GaussianBlur(gray1,(5,5),0)

    img2=cv2.resize(img2,(800,800),interpolation=cv2.INTER_AREA)
    gray2=cv2.cvtColor(img2,cv2.COLOR_BGR2GRAY)
    gray2=cv2.GaussianBlur(gray2,(5,5),0)

    kp1,des1=orb.detectAndCompute(gray1,None)
    kp2,des2=orb.detectAndCompute(gray2,None)

    #img=cv2.drawKeypoints(img,kp,img,color=(255,0,255))
    #bf=cv2.BFMatcher(cv2.NORM_HAMMING,crossCheck=True)

    FLANN_INDEX_KDTREE = 0
    #index_params = dict(algorithm = FLANN_INDEX_KDTREE, trees = 5)
    search_params = dict(checks=50)


    index_params= dict(algorithm = FLANN_INDEX_LSH,
                   table_number = 6, # 12
                   key_size = 12,     # 20
                   multi_probe_level = 1) #2


    flann = cv2.FlannBasedMatcher(index_params,search_params)

    #bf=cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
    #f=cv2.BFMatcher()
    #try:

    des1=np.asarray(des1,np.float32)
    des2=np.asarray(des2,np.float32)
    #matches=flann.knnMatch(np.asarray(des1,np.float32),np.asarray(des2,np.float32),k=2)
    try:
        matches=flann.knnMatch(des1,des2,k=2)
    except:
        return
    #matches=bf.match(des1,des2)
    #except:
            #return
    #matches=flann.knnMatch(des1,des2,k=2)
    print('matches:'+str(len(matches)),end='')
    good = []
    try:
        for m,n in matches:
            if(m.distance < 0.75*n.distance):
                good.append([m])
    except:
        return
    print('  good:'+str(len(good)))
    #img3 = cv2.drawMatchesKnn(img1,kp1,img2,kp2,matches,None, flags=2)
    img3 = cv2.drawMatchesKnn(img1,kp1,img2,kp2,good,None, flags=2)
    #img3 = cv2.drawMatches(img1,kp1,img2,kp2,matches,None, flags=2)
    cv2.imshow('img',img3)
    #i=cv2.waitKey(0)   




def getMaxContour(contours):#获取最大的轮廓
    max_area=0
    max_cnt=0
    for i in range(len(contours)):
        cnt=contours[i]
        area=cv2.contourArea(cnt)
        if(area>max_area):
            max_area=area
            max_cnt=cnt
    return max_cnt



def getRangeContours(contours,low,high):#获取指定面积范围内的轮廓，返回轮廓列表list
    contours_list=[]
    for i in range(len(contours)):
        cnt=contours[i]
        area=cv2.contourArea(cnt)
        if(area>low and area<high):
            contours_list.append(cnt)
    return contours_list




'''
画轮廓函数
'''
def draw(img):
    #img=cv2.resize(img,(720,1280),interpolation=cv2.INTER_AREA)
    #img= cv2.blur(img,(3,3))    #进行滤波去掉噪声
    #img= cv2.medianBlur(img,5)    #进行滤波去掉噪声
    gray=cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
    #gray=img
    #kernel = cv2.getStructuringElement(cv2.MORPH_RECT,(50, 50))
    #开闭运算，先开运算去除背景噪声，再继续闭运算填充目标内的孔洞
    #opened = cv2.morphologyEx(gray, cv2.MORPH_OPEN, kernel) 
    
    #closed = cv2.morphologyEx(opened, cv2.MORPH_CLOSE, kernel) 
    #cv2.imshow('sa',closed)

    #cv2.imshow('gray',gray)
    ret, binary = cv2.threshold(gray,160,255,cv2.THRESH_BINARY)
    #cv2.imshow('binary',binary)
    #contours,hierarchy=cv2.findContours(binary,cv2.RETR_TREE,cv2.CHAIN_APPROX_NONE)
    contours,hierarchy=cv2.findContours(binary,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)
    #contours,hierarchy=cv2.findContours(binary,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)
    #cv2.drawContours(img,contours,-1,(0,0,0),cv2.FILLED)
    print(len(contours))
    if contours:
        c_max=[]
        max_contour=getMaxContour(contours)
        c_max.append(max_contour)
        try:
            cv2.drawContours(img,c_max,-1,(0,0,255),3)
        except:
            return

        #r1=np.zeros(img.shape[:2],dtype="uint8")#创建黑色图像
        x, y, w, h = cv2.boundingRect(max_contour)   # 将轮廓信息转换成(x, y)坐标，并加上矩形的高度和宽度
        cv2.rectangle(img, (x,y), (x+w, y+h), (255, 255, 255), 2)  # 画出矩形
        
        #mask=r1
        #masked=cv2.bitwise_and(img,img,mask=mask)

        #rect = cv2.minAreaRect(max_contour)
        #box = cv2.boxPoints(rect)
        #box =np.int0(box)
        #cv2.drawContours(img, [box], 0, (0, 0, 255), 3)  # 画出该矩形
        # 注：OpenCV没有函数能直接从轮廓信息中计算出最小矩形顶点的坐标。所以需要计算出最小矩形区域，
        # 然后计算这个矩形的顶点。由于计算出来的顶点坐标是浮点型，但是所得像素的坐标值是整数（不能获取像素的一部分），
        # 所以需要做一个转换


        #(x,y),radius = cv2.minEnclosingCircle(max_contour)
        #center = (int(x),int(y))
        #radius = int(radius)
        #img = cv2.circle(img,center,radius,(0,255,0),2)

        #cv2.imshow('final',masked)
        cv2.imshow('final',img)
    #i=cv2.waitKey(0)




'''
获取视频帧测试函数
'''
def getVideo():
    cap=cv2.VideoCapture(0)
    #img1=cv2.imread('1.jpg')
    #img1=cv2.imread('5.jpg')
    #surf = cv2.xfeatures2d.SURF_create(400)
    #img1=cv2.resize(img1,(800,800),interpolation=cv2.INTER_AREA)
    #gray1=cv2.cvtColor(img1,cv2.COLOR_BGR2GRAY)
    #kp1,des1=surf.detectAndCompute(gray1,None)

    #mog2=cv2.createBackgroundSubtractorMOG2()
    #mog2=cv2.bgsegm.createBackgroundSubtractorGMG()
    while(True):
        #time.sleep(0.034)
        ret,frame=cap.read()
        if(ret != True):
            continue
        #hamming=ImgHash.compare(img1,frame,'dHash')
        #print(hamming)
        #frame=cv2.blur(frame,(3,3))
        #frame=cv2.GaussianBlur(frame,(5,5),0)
        #fgmask=mog2.apply(frame)
        #draw(fgmask)
        #cv2.imshow('1',fgmask)
        draw(frame)
        #surf_test_fast(kp1,des1,img1,frame)
        #drawFaces(frame)
        if cv2.waitKey(1) & 0xFF==ord('q'):
            break
    cap.release()
    cv2.destroyAllWindows()



def main():
    getVideo()
    #img1=cv2.imread('1.jpg')
    #img2=cv2.imread('3.jpg')
    #orb_test(img1,img2)
    #img=cv2.imread('8.jpg')
    #img=cv2.resize(img,(800,800),interpolation=cv2.INTER_AREA)
    #drawFaces(img)
    #i=cv2.waitKey(0)

if __name__ == '__main__':
    main()

#face_detect()