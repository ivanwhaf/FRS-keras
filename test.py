import cv2
import numpy as np
from keras.models import load_model
from image_processing import *
from keras import backend as K
from keras.models import Model


# 输入图像维度
width, height = 200,180 
img_size=width*height


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



def show_intermediate_output(model,image):
    image=cv2.resize(image,(width,height))
    image=cv2.cvtColor(image,cv2.COLOR_BGR2GRAY)

    img_ndarray=np.asarray(image,dtype='float64')/255
    image=np.ndarray.flatten(img_ndarray)
    test_data=image
    test_data = test_data.astype('float32')

    if K.image_data_format() == 'channels_first':
        test_data = test_data.reshape(1, 1, height, width)
    else:
        test_data = test_data.reshape(1, height, width, 1)
    o=conv_output(model,'conv1',test_data)
    print(o.shape)



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
    n=0
    for x in range(0,img_width-slid_step,step):
        for y in range(0,img_height-slid_step,step):
            f=img[y:y+slid_step,x:x+slid_step]
            clas,confidence=get_class_and_confidence(f,model)
            n=n+1
            print(str(int(x/step))+' '+str(int(y/step))+str(clas)+str(confidence))



def get_class_and_confidence(img,model):
    test_data=np.empty((1,img_size))
    try:
        img=cv2.resize(img, (width, height), interpolation=cv2.INTER_CUBIC)
    except:
        print('resize error!')
        return 0
    img = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
    img_ndarray=np.asarray(img,dtype='float64')/255
    test_data==np.ndarray.flatten(img_ndarray)
    test_data = test_data.astype('float32')

    if K.image_data_format() == 'channels_first':
        test_data = test_data.reshape(1, 1, height, width)
    else:
        test_data = test_data.reshape(1, height, width, 1)

    predict=model.predict(test_data,batch_size = 1)
    clas=np.argmax(predict, axis=1) 
    confidence=float(predict[0][clas])
    confidence='%.3f'%(confidence*100) #置信度转化为百分比，保留3位小数 
    return clas,confidence


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

def main():
	model=load_model('model.h5')
	#show_intermediate_output(model,image)
	image=cv2.imread('test2.jpg')
	#image=drawGrid(image)
	sliding_window(image,9,model)
	#cv2.imshow('fram', image)
	if cv2.waitKey(0) & 0xFF==ord('q'):
		cv2.destroyAllWindows()

if __name__ == '__main__':
	#main()
    img=cv2.imread('fqcd.jpg')
    img=cv2.resize(img,(800,600))
    gray=cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
    
    ret, binary = cv2.threshold(gray,160,255,cv2.THRESH_BINARY)
    contours,hierarchy=cv2.findContours(binary,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)
    print(len(contours))
    if contours:
        c_max=[]
        max_contour=getMaxContour(contours)
        c_max.append(max_contour)
        try:
            cv2.drawContours(img,c_max,-1,(0,0,255),3)
        except:
            pass
            
    '''
    circles=cv2.HoughCircles(gray,cv2.HOUGH_GRADIENT,1,120,param1=120,param2=100,minRadius=200,maxRadius=1000)
    if not circles is None:
        circles = np.uint16(np.around(circles))
        for i in circles[0,:]:
            cv2.circle(img,(i[0],i[1]),i[2],(0,0,255),2)'''

    cv2.imshow('final',img)
    i=cv2.waitKey(0)