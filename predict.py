# @Author: Ivan
# @LastEdit: 2020/9/23
import os
import argparse
import cv2  # install
from PIL import Image, ImageDraw, ImageFont  # install
from keras.models import load_model
from keras import backend as K
import numpy as np  # install

# input shape
width, height, depth = 100, 100, 3


def load_keras_model(path):
    return load_model(path)


def load_classes(path):
    classes = {}
    with open(path, 'r', encoding='utf-8') as f:
        lines = f.readlines()
        for line in lines:
            line = line.strip('\n').split(' ')
            class_, idx = line[0], line[1]
            classes[int(idx)] = class_
    return classes


def load_prices(path):
    prices = {}
    with open(path, 'r', encoding='utf-8') as f:
        lines = f.readlines()
        for line in lines:
            line = line.strip('\n').split(' ')
            class_, price = line[0], line[1]
            prices[class_] = price
    return prices


def predict_img(img, model):
    """get model prediction of one image

    Args:
        img: image ndarray
        model: keras trained model
    Returns:
        preds: keras model predictions
    """
    if depth == 1:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    try:
        img = cv2.resize(img, (width, height))
    except:
        print('Img resize error!')
        raise Exception('Resize error!')

    img_ndarray = np.asarray(img, dtype='float64') / 255
    test_data = np.ndarray.flatten(img_ndarray)
    test_data = test_data.astype('float32')

    if K.image_data_format() == 'channels_first':
        test_data = test_data.reshape(1, depth, height, width)
    else:
        test_data = test_data.reshape(1, height, width, depth)

    preds = model.predict(test_data)
    return preds


def predict_class_idx_and_confidence(img, model):
    """predict image class and confidence according to trained model

    Args:
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
    confidence = '%.2f' % (confidence * 100)

    return class_, confidence


classes = load_classes('cfg/classes.cfg')


def predict_class_name_and_confidence(img, model):
    """predict image class and confidence according to trained model

    Args:
        img: prediction image
        model: keras model
    Returns:
        class_: class index
        confidence: class confidence
    """
    class_, confidence = predict_class_idx_and_confidence(img, model)
    class_name = classes[int(class_)]

    return class_name, confidence


def predict_and_show_one_img(img, model, classes):
    """get model output of one image

    Args:
        img: image ndarray
        model: keras trained model
        classes: classes list
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

    img = cv2.cvtColor(np.asarray(img), cv2.COLOR_RGB2BGR)

    # cv2.namedWindow('img', 0)
    # cv2.resizeWindow('img',window_width,window_height)

    # show image
    cv2.imshow('img', img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

    return class_name, confidence


def arg_parse():
    """
    Parse arguements to the detect module

    """

    parser = argparse.ArgumentParser(description='Food Recognition System')

    parser.add_argument("--image", "-i", dest='image',
                        help="Path of image to perform detection upon", type=str)

    parser.add_argument("--video", "-v", dest='video',
                        help="Path of video to run detection upon", type=str)

    parser.add_argument("--model", "-m", dest='model', help="Path of network model",
                        default="model.h5", type=str)

    return parser.parse_args()


if __name__ == "__main__":
    args = arg_parse()
    image_path = args.image
    video_path = args.video
    model_path = args.model

    model = load_keras_model(model_path)
    print('Model successfully loaded...')

    if image_path:
        img = cv2.imread(image_path)
        class_name, confidence = predict_class_name_and_confidence(img, model)
        print('Class name:', class_name, 'confidence:', str(confidence)+'%')
    elif video_path:
        cap = cv2.VideoCapture(video_path)
        while cap.isOpened():
            ret, frame = cap.read()
            # frame = raw_frame
            if not ret:
                break
            class_name, confidence = predict_class_name_and_confidence(
                frame, model)

            img = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
            draw = ImageDraw.Draw(img)
            font_text = ImageFont.truetype("simsun.ttc", 60, encoding="utf-8")
            draw.text((5, 5), class_name +
                      str(confidence)+'%', (0, 255, 0), font=font_text)
            frame = cv2.cvtColor(np.asarray(img), cv2.COLOR_RGB2BGR)

            print('Class name:', class_name, 'confidence:', str(confidence)+'%')
            cv2.resizeWindow('frame', (int(cap.get(3)), int(cap.get(4))))
            cv2.imshow("frame", frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

        cap.release()
        cv2.destroyAllWindows()
