# @Author: Ivan
# @LastEdit: 2020/9/4
import os
import random
import cv2  # install
import numpy as np  # install


def load_img(root, nb_classes, nb_per_class, width, height, depth, train_proportion, valid_proportion,
             test_proportion, normalize=True):
    """function of loading image dataset.

    * image dataset's root folder must contain each class's children class folder.
    example:
        folder 'dataset' contains 'car','train' and 'bicycle' folder.
        each children class folder has corresponding images,'car' folder has 'car1.jpg','car2.jpg'.
    * file path name and image name better be named by english.

    Args:
        root: image dataset's root path
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
    """
    train_per_class = int(train_proportion * nb_per_class)
    valid_per_class = int(valid_proportion * nb_per_class)
    test_per_class = int(test_proportion * nb_per_class)
    train_number = train_per_class * nb_classes  # number of train set
    valid_number = valid_per_class * nb_classes  # number of valid set
    test_number = test_per_class * nb_classes  # number of test set

    classes = []  # images classes list
    dataset = []  # dataset list,including image and label sequences

    print('Image classes:')
    img_classes = os.listdir(root)
    for c, img_class in enumerate(img_classes):
        # stop loading dataset when class number is enough
        if c == nb_classes:
            break
        img_class_path = os.path.join(root, img_class)
        # skip if not folder
        if not os.path.isdir(img_class_path):
            continue

        print('<', img_class, '>')
        classes.append(img_class)
        m = 0
        imgs = os.listdir(img_class_path)
        for _, img in enumerate(imgs):
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
                # default interpolation = cv2.INTER_LINEAR - Bilinear Interpolation
                image = cv2.resize(image, (width, height))
                if normalize:
                    image = cv2.medianBlur(image, 3)  # filtering
            except:
                print(img_path, 'resize error!')
                # os.remove(img_path)
                # print(img_path+' has been deleted!')
                continue
            m = m + 1

            # add single data(including data and label) to dataset array
            image_ndarray = np.asarray(image, dtype='float64') / 255
            dataset.append([np.ndarray.flatten(image_ndarray), c])

        # image not enough
        if m < nb_per_class:
            print('Image number insufficient!', m)
            raise Exception('Image number insufficient!')

    # shuffle the whole dataset
    random.shuffle(dataset)

    # construct data set and label set
    train_data = [data[0] for data in dataset[:train_number]]
    train_label = [data[1] for data in dataset[:train_number]]
    train_data = np.array(train_data, dtype='float32')
    train_label = np.array(train_label, dtype='uint8')

    valid_data = [data[0]
                  for data in dataset[train_number:train_number + valid_number]]
    valid_label = [data[1]
                   for data in dataset[train_number:train_number + valid_number]]
    valid_data = np.array(valid_data, dtype='float32')
    valid_label = np.array(valid_label, dtype='uint8')

    test_data = [data[0] for data in dataset[train_number + valid_number:]]
    test_label = [data[1] for data in dataset[train_number + valid_number:]]
    test_data = np.array(test_data, dtype='float32')
    test_label = np.array(test_label, dtype='uint8')

    # write all classes into 'classes.txt' file
    with open('classes.txt', 'w', encoding='utf-8') as f:
        for class_ in classes:
            f.write(class_ + '\n')

    rval = [(train_data, train_label), (valid_data, valid_label),
            (test_data, test_label)]

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
