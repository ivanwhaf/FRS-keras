# Author:Ivan
import os
import random
import cv2  # install
import numpy as np  # install
from PIL import Image  # install
# import matplotlib.pyplot as plt  # install


def load_img(path, category_number, number_per_category, width, height, depth, train_per_category, valid_per_category, test_per_category):
    """加载图片函数
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
        depth:读取的图片深度
        train_per_category:每类训练图片数量
        valid_per_category:每类验证图片数量
        test_per_category:每类测试图片数量
        normalize:图片是否需要归一化处理,默认不需要
    Returns:
        rval包含3个元组和一个类别列表
           (train_data, train_label):训练数据和标签
           (valid_data, valid_label):验证数据和标签
           (test_data, test_label):测试数据和标签
           category:类别列表
    """
    number = category_number*number_per_category  # 图片总数
    n = 0  # images[]数组下标
    category = []  # 图片类别列表
    img_size = width*height  # 图片大小 宽*高
    images = np.empty((number, img_size*depth))  # 图片集

    img_categories = os.listdir(path)
    print('Image categories:')
    for c, img_category in enumerate(img_categories):
        # 若读取的图片类别足够，停止加载数据集
        if c >= category_number:
            break
        # 若不为文件夹，跳过
        if not os.path.isdir(img_category_path):
            continue

        print('<', img_category, '>')
        category.append(img_category)
        img_category_path = os.path.join(path, img_category)  # 每一类图片的路径

        imgs = os.listdir(img_category_path)
        im = []  # 暂存每类的图片，最后打乱后加入images数组
        m = 0  # 每类已经成功读取的图片数量
        for img in imgs:
            # 每类已经读取的图片数量若大于设定的每类图片数量，停止加载数据集
            if m >= number_per_category:
                break
            img_path = os.path.join(img_category_path, img)
            if depth == 3:
                # image=cv2.imread(img_path,cv2.IMREAD_COLOR)
                # image=Image.open(img_path)
                image = cv2.imdecode(np.fromfile(
                    img_path, dtype=np.uint8), cv2.IMREAD_COLOR)  # 中文路径
            elif depth == 1:
                image = cv2.imread(img_path)
                image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)  # 转换为灰度图!
                # image=cv2.imread(img_path,cv2.IMREAD_GRAYSCALE)  # 读取灰度图
            try:
                # image=cv2.resize(image,(width,height),interpolation=cv2.INTER_CUBIC)  # 调整图片大小
                image = cv2.resize(image, (width, height))  # 调整图片大小
            except:
                print(img_path, 'resize error!')
                # os.remove(img_path)
                # print(img_path+' has been deleted.')
                continue

            image_ndarray = np.asarray(image, dtype='float64')/255
            im.append(np.ndarray.flatten(image_ndarray))
            # images[n]=np.ndarray.flatten(image_ndarray)
            n = n+1
            m = m+1

        # 每类的图片数量不足
        if((m+1) < number_per_category):
            print('image number unequal!'+str(m))
            exit()

        random.shuffle(im)  # 随机打乱每类图像
        # 添加到总的images数组里
        for i in range(n-number_per_category, n):
            images[i] = im[i % number_per_category]

    label = np.empty(number)  # 标签数组
    for i in range(category_number):
        label[i*number_per_category:(i+1)*number_per_category] = i
    label = label.astype(np.int)

    train_data_number = train_per_category*category_number
    valid_data_number = valid_per_category*category_number
    test_data_number = test_per_category*category_number

    train_data = np.empty((train_data_number, img_size*depth))
    train_label = np.empty(train_data_number)
    valid_data = np.empty((valid_data_number, img_size*depth))
    valid_label = np.empty(valid_data_number)
    test_data = np.empty((test_data_number, img_size*depth))
    test_label = np.empty(test_data_number)

    # 遍历每一个类别，构造数据集和标签数组
    for i in range(category_number):
        train_data[i*train_per_category: (i+1)*train_per_category] = images[i *
                                                                            number_per_category: i*number_per_category+train_per_category]  # 训练集数据
        train_label[i*train_per_category: (i+1)*train_per_category] = label[i *
                                                                            number_per_category: i*number_per_category+train_per_category]  # 训练集标签
        valid_data[i*valid_per_category: (i+1)*valid_per_category] = images[i*number_per_category +
                                                                            train_per_category: i*number_per_category+train_per_category+valid_per_category]  # 验证集数据
        valid_label[i*valid_per_category: (i+1)*valid_per_category] = label[i*number_per_category +
                                                                            train_per_category: i*number_per_category+train_per_category+valid_per_category]  # 验证集标签
        test_data[i*test_per_category: (i+1)*test_per_category] = images[i*number_per_category+train_per_category +
                                                                         valid_per_category: i*number_per_category+train_per_category+valid_per_category+test_per_category]  # 测试集数据
        test_label[i*test_per_category: (i+1)*test_per_category] = label[i*number_per_category+train_per_category +
                                                                         valid_per_category: i*number_per_category+train_per_category+valid_per_category+test_per_category]   # 测试集标签

    # print('train_label:',train_label)
    # print('valid_label:',valid_label)
    # print('test_label:',test_label)

    train_data = train_data.astype('float32')
    valid_data = valid_data.astype('float32')
    test_data = test_data.astype('float32')

    rval = [(train_data, train_label), (valid_data, valid_label),
            (test_data, test_label), category]
    return rval


def img_normalize(path, width, height):
    """
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
    """
    img_categories = os.listdir(path)
    for img_category in img_categories:
        img_category_path = os.path.join(path, img_category)
        if os.path.isdir(img_category_path):
            imgs = os.listdir(img_category_path)
            for img in imgs:
                img_path = os.path.join(img_category_path, img)
                image = cv2.imread(img_path)
                # resize
                try:
                    image = cv2.resize(image, (width, height),
                                       interpolation=cv2.INTER_CUBIC)  # 调整图片大小
                except:
                    print(img_path+' resize error!')
                    os.remove(img_path)
                    print(img_path+' has been deleted.')
                    continue
                # convert to gray
                image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)  # 转换为灰度图
                cv2.imwrite(img_path, image)
                print(img_path+' normalized successfully!')
    print('all images normalized successfully!')


def img_rename(path):
    """
    图片排序重命名函数
    遍历文件夹将所有图片从1开始重命名,如从p1.jpg~p100.jpg
    Args:
        path:图片文件夹路径
    Returns:
        无
    """
    categories = os.listdir(path)
    for category in categories:
        number = 1
        category_path = os.path.join(path, category)
        imgs = os.listdir(category_path)
        for img in imgs:
            img_path = os.path.join(category_path, img)
            new_img_path = os.path.join(category_path, 'p'+str(number)+'.jpg')
            os.rename(img_path, new_img_path)
            print(img_path+'----->'+new_img_path)
            number = number+1
    print('all images renamed successfully!')


def main():
    pass


if __name__ == '__main__':
    main()
