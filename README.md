# Food-Recognition-System
Deep Learning based food recognition system
Using `Keras`(`Tensorflow` backend) deep learning framwork
Also can be transplanted to other platforms like `Raspberry Pi`

# Usage
* 1.run spiders(`spider_baidu`,`spider_douguo.py`) to crawl raw image data from the internet
* 2.run `image_preorocessing.py` to preprocess the image and make dataset
* 3.run `image_train.py` to train the model (only when dataset is finished)
* 4.run `image_predict.py` to load the model and recongnize the food image

# Program Structure
## 1.Image Spiders module
* folder: 'spiders' 
* file: `image_baidu.py` , `image_douguo.py`

## 2.Image Preprocessing module
* file:`image_preprocessing.py`

## 3.Training module
* file:`image_train.py`

## 4.UI and Predicting module
* file:`image_predict.py`

# Environment
* Windows 10 / Raspbian(Debian based)
* Python 3.6.8
* Tensorflow
* Keras
* GTX 1060 3G
* Raspberry Pi 3