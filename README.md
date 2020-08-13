# Food-Recognition-System
* Deep Learning based food recognition system
Using `Keras`(`Tensorflow` backend) deep learning framwork
* Also can be transplanted to other platforms like `Raspberry Pi`

# Usage
* 1.run spiders`spider_baidu.py`,`spider_douguo.py` to crawl raw image data from the internet
* 2.create dataset folder and move raw image data into it.
* 3.run `image_train.py` to train the model (only when dataset is downloaded)
* 4.run `image_predict_qt.py` to load the model and recongnize the food image

# Program Structure
## Image Spiders module
* folder: spiders 
* file: `image_baidu.py` , `image_douguo.py`

## Image Preprocessing module
* file:`image_preprocess.py`

## Image Utils module
* file:`image_util.py`

## Training module
* file:`image_train.py`

## UI and Predicting module
* file:`image_predict_qt.py`,`image_predict_wx.py`

# Environment
## PC Ⅰ
* Windows 10
* Python 3.6.8
* CUDA 9.0
* cuDNN 7.4
* tensorflow-gpu 1.9.0
* Keras 2.2.4
* PyQt5 5.15.0
* Nvidia GTX 1060 3G

## PC Ⅱ
* Windows 10
* Python 3.7.8
* CUDA 10.0
* cuDNN 7.4
* tensorflow-gpu 1.14.0
* Keras 2.3.1
* PyQt5 5.15.0
* Nvidia MX350 2G

## PC Ⅲ
* Raspbian(Debian based)
* ...