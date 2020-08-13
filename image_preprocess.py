# @Author: Ivan
# @LastEdit: 2020/8/13
import os
import random
import cv2  # install
import numpy as np  # install


def load_img(path, nb_classes, nb_per_class, width, height, depth, train_proportion, valid_proportion,
             test_proportion, normalize=True):
    """function of loading images dataset.
    
    * image dataset folder must contain each class's children class folder.
    example:
        folder 'vehicle' contains 'car','train' and 'bicycle' folder.
        each children class folder has corresponding images,'car' folder has 'car1.jpg','car2.jpg'.
    * file path name and image name better be named by english.

    Args:
        path: image dataset's path
        nb_classes: number of image classes
        nb_per_class: number of each class's image
        width: width of output image
        height: height of output image
        depth: depth of image,1 for gray,3 for clolr
        train_proportion: the proportion of train dataset
        valid_proportion: the proportion of valid dataset
        test_proportion: the proportion of test dataset
        normalize: images whether need normalized
    Returns:
        rval:
           (train_data, train_label): train data and label
           (valid_data, valid_label): valid data and label
           (test_data, test_label): test data and label
           classes: class list
    """
    train_per_class = int(train_proportion * nb_per_class)
    valid_per_class = int(valid_proportion * nb_per_class)
    test_per_class = int(test_proportion * nb_per_class)
    number = nb_classes * nb_per_class  # number of all images
    n = 0  # images array's index
    classes = []  # images classes list
    images = np.empty((number, width * height * depth))  # images set

    print('Image classes:')
    img_classes = os.listdir(path)
    for c, img_class in enumerate(img_classes):
        # stop loading dataset when class number is enough
        if c == nb_classes:
            break
        img_class_path = os.path.join(path, img_class)
        # skip if not folder
        if not os.path.isdir(img_class_path):
            continue

        print('<', img_class, '>')
        classes.append(img_class)

        temp = []  # store each class's images temporarily,then shuffle them and add into 'images' array
        m = 0  # number of each class's images that loaded successfully

        imgs = os.listdir(img_class_path)
        for img in imgs:
            # stop loading dataset when image number of each class is enough
            if m == nb_per_class:
                break
            img_path = os.path.join(img_class_path, img)
            # read image
            if depth == 3:
                image = cv2.imdecode(np.fromfile(
                    img_path, dtype=np.uint8), cv2.IMREAD_COLOR)  # chinese path color
            elif depth == 1:
                image = cv2.imdecode(np.fromfile(
                    img_path, dtype=np.uint8), cv2.cv2.IMREAD_GRAYSCALE)  # chinese path gray
            # scaling and filtering
            try:
                # default interpolation=cv2.INTER_LINEAR - Bilinear Interpolation
                image = cv2.resize(image, (width, height))
                if normalize:
                    image = cv2.medianBlur(image, 3)  # filtering
            except:
                print(img_path, 'resize error!')
                # os.remove(img_path)
                # print(img_path+' has been deleted!')
                continue

            image_ndarray = np.asarray(image, dtype='float64') / 255
            temp.append(np.ndarray.flatten(image_ndarray))
            # images[n]=np.ndarray.flatten(image_ndarray)
            n = n + 1
            m = m + 1

        # image number is not enough
        if m < nb_per_class:
            print('Image number insufficient!', m)
            raise Exception('Image number insufficient!')

        random.shuffle(temp)  # shuffle each class's images

        # add each class's images into all images array
        images[n - nb_per_class:n] = temp[:]

    # construct label array
    label = np.empty(number, dtype='uint8')
    for i in range(nb_classes):
        label[i * nb_per_class:(i + 1) * nb_per_class] = i

    train_number = train_per_class * nb_classes  # number of train set
    valid_number = valid_per_class * nb_classes  # number of valid set
    test_number = test_per_class * nb_classes  # number of test set

    # train dataset,valid dataset,test dataset,convert float64 to float32 for saving memory
    train_data = np.empty((train_number, width * height * depth), dtype='float32')
    train_label = np.empty(train_number)
    valid_data = np.empty((valid_number, width * height * depth), dtype='float32')
    valid_label = np.empty(valid_number)
    test_data = np.empty((test_number, width * height * depth), dtype='float32')
    test_label = np.empty(test_number)

    # traversal each class,construct dataset and label set
    for i in range(nb_classes):
        train_data[i * train_per_class: (i + 1) * train_per_class] = images[i *
                                                                            nb_per_class: i * nb_per_class + train_per_class]  # train dataset
        train_label[i * train_per_class: (i + 1) * train_per_class] = label[i *
                                                                            nb_per_class: i * nb_per_class + train_per_class]  # train label
        valid_data[i * valid_per_class: (i + 1) * valid_per_class] = images[i * nb_per_class +
                                                                            train_per_class: i * nb_per_class + train_per_class + valid_per_class]  # valid dataset
        valid_label[i * valid_per_class: (i + 1) * valid_per_class] = label[i * nb_per_class +
                                                                            train_per_class: i * nb_per_class + train_per_class + valid_per_class]  # test label
        test_data[i * test_per_class: (i + 1) * test_per_class] = images[i * nb_per_class + train_per_class +
                                                                         valid_per_class: i * nb_per_class + train_per_class + valid_per_class + test_per_class]  # test dataset
        test_label[i * test_per_class: (i + 1) * test_per_class] = label[i * nb_per_class + train_per_class +
                                                                         valid_per_class: i * nb_per_class + train_per_class + valid_per_class + test_per_class]  # test label

    rval = [(train_data, train_label), (valid_data, valid_label),
            (test_data, test_label), classes]
    return rval


def img_normalize(path, width, height, gray=True):
    """function of normalizing images.

    * resize images and convert to gray.
    * image dataset folder must contain each class's children class folder.
    example:
        folder 'vehicle' contains 'car','train' and 'bicycle' folder,each children class
        folder has corresponding images,'car' folder has 'car1.jpg','car2.jpg'.
    * file path name and image name better be named by english.
    * when detecting faces,'aarcascade_frontalface_default.xml' shoud be included.

    Args:
        path: images path
        width: width of output image
        height: height of output image
        gray: whether need convert to gray,default need
    Returns:
        None
    """
    img_classes = os.listdir(path)
    for img_class in img_classes:
        img_class_path = os.path.join(path, img_class)
        if os.path.isdir(img_class_path):
            imgs = os.listdir(img_class_path)
            for img in imgs:
                img_path = os.path.join(img_class_path, img)
                image = cv2.imread(img_path)
                # resize
                try:
                    image = cv2.resize(image, (width, height),
                                       interpolation=cv2.INTER_CUBIC)
                except:
                    print(img_path + ' resize error!')
                    os.remove(img_path)
                    print(img_path + ' has been deleted!')
                    continue
                # convert to gray
                if gray:
                    image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
                cv2.imwrite(img_path, image)
                print(img_path, 'normalized successfully!')
    print('All images normalized successfully!')


def img_rename(path):
    """function of renaming and sorting images.

    * traversal folder,rename all images.
    example:
        'xxx.jpg','xxx.jpg' -> '1.jpg','2.jpg'

    Args:
        path: images path
    Returns:
        None
    """
    classes = os.listdir(path)
    for class_ in classes:
        number = 1
        class_path = os.path.join(path, class_)
        imgs = os.listdir(class_path)
        for img in imgs:
            img_path = os.path.join(class_path, img)
            new_img_path = os.path.join(class_path, str(number) + '.jpg')
            os.rename(img_path, new_img_path)
            print(img_path + '--->' + new_img_path)
            number = number + 1
    print('All images renamed successfully!')


def main():
    pass


if __name__ == '__main__':
    main()
