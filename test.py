import cv2
import numpy as np
from keras.models import load_model
from keras import backend as K
from keras.models import Model
import matplotlib.pyplot as plt  # install
from image_preprocess import *

# input shape
width, height, depth = 100, 100, 3


def get_class_and_confidence(img, model):
    # test_data = np.empty((1, width*height*3))
    if depth == 1:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
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

    predict = model.predict(test_data)
    class_ = np.argmax(predict, axis=1)
    confidence = float(predict[0][class_])
    confidence = '%.3f' % (confidence * 100)
    return class_, confidence


def get_intermediate_output(model, layer_name, img):
    """Get the output of intermediate layer.
    Args:
           model: keras model.
           layer_name: name of layer in the model.
           img: processed input image.
    Returns:
           intermediate_output: feature map.
    """
    try:
        # this is the placeholder for the intermediate output
        out_intermediate = model.get_layer(layer_name).output
    except:
        raise Exception('Not layer named {}!'.format(layer_name))

    # get the intermediate layer model
    intermediate_layer_model = Model(
        inputs=model.input, outputs=out_intermediate)

    # get the output of intermediate layer model
    intermediate_output = intermediate_layer_model.predict(img)
    return intermediate_output[0]


def show_intermediate_output(model, layer_name, image):
    if depth == 1:
        image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    image = cv2.resize(image, (width, height))

    img_ndarray = np.asarray(image, dtype='float64') / 255
    test_data = np.ndarray.flatten(img_ndarray)
    test_data = test_data.astype('float32')

    if K.image_data_format() == 'channels_first':
        test_data = test_data.reshape(1, depth, height, width)
    else:
        test_data = test_data.reshape(1, height, width, depth)

    output = get_intermediate_output(model, layer_name, test_data)
    n = output.shape[-1]  # 特征图中特征个数
    size = output.shape[1]
    display_grid = np.zeros((size * 1, n * size))
    for i in range(n):
        channel_image = output[:, :, i]
        display_grid[0:size, i * size:(i + 1) * size] = channel_image
    # plt.figure()
    # plt.title(layer_name)
    # plt.grid(False)
    # plt.imshow(display_grid, cmap='viridis')
    # plt.savefig('visualize/'+layer_name+'_output.jpg')  # 保存中间层输出图
    # plt.show()  # must show after imshow


def show_heatmap(model, layer_name, image):
    img = image.copy()
    if depth == 1:
        image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    image = cv2.resize(image, (width, height))

    img_ndarray = np.asarray(image, dtype='float64') / 255
    test_data = np.ndarray.flatten(img_ndarray)
    test_data = test_data.astype('float32')

    if K.image_data_format() == 'channels_first':
        test_data = test_data.reshape(1, depth, height, width)
    else:
        test_data = test_data.reshape(1, height, width, depth)

    preds = model.predict(test_data)
    index = np.argmax(preds[0])  # index of output class
    output = model.output[:, index]

    layer = model.get_layer(layer_name)  # intermediate layer

    grads = K.gradients(output, layer.output)[0]

    pooled_grads = K.mean(grads, axis=(0, 1, 2))
    iterate = K.function([model.input], [pooled_grads, layer.output[0]])

    pooled_grads_value, layer_output_value = iterate([test_data])

    for i in range(layer_output_value.shape[-1]):
        layer_output_value[:, :, i
        ] *= pooled_grads_value[i]
    heatmap = np.mean(layer_output_value, axis=-1)

    # heatmap = np.maximum(heatmap, 0)
    # heatmap /= np.max(heatmap)

    plt.matshow(heatmap)
    plt.savefig('visualize/heatmap.jpg')
    plt.show()

    # img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    heatmap = cv2.resize(heatmap, (img.shape[1], img.shape[0]))
    heatmap = np.uint8(255 * heatmap)  # convert to rgb
    heatmap = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)  # heatmap apply to raw image
    superimposed_img = heatmap * 0.4 + img  # heatmap intensity factor - 0.4
    cv2.imwrite('visualize/heatmap_apply.jpg', superimposed_img)


def main():
    model = load_model('model.h5')
    model.summary()
    # image = cv2.imread('test.jpg')
    # show_intermediate_output(model, 'conv1', image)
    # show_intermediate_output(model, 'maxpooling1', image)
    # show_intermediate_output(model, 'conv2', image)
    # show_intermediate_output(model, 'maxpooling2', image)

    image = cv2.imread('test.jpg')
    show_heatmap(model, 'conv1', image)


if __name__ == "__main__":
    main()
