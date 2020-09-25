# Food-Recognition-System
* Deep Learning based food recognition system
Using `Keras`(`Tensorflow` backend) deep learning framwork
* Also can be transplanted to other platforms like `Raspberry Pi`

# Usage
## run directly
* 1.run spiders`spider_baidu.py` or `spider_douguo.py` to crawl raw image data from the internet
* 2.create an empty folder and move raw images into it,in this project was `dataset` folder
* 3.run `train.py` to train the model (only when dataset was downloaded)
* 4.run `cam_demo.py` to show ui,load the model and recongnize the food
## run in command line
* `python detect.py -i test.jpg`
* `python detect.py -v test.mp4`
## caution
* need plotting model structure? just install `graphviz` first
* please screen out unqualified raw images manually after crawling

# Program Structure
## Image Preprocessing module
* file:`preprocess.py`
* preprocess image dataset

## Image Utils module
* file:`image_util.py`
* some image utils and algorithms

## Training module
* file:`train.py`
* main training program

## UI and Predicting module
* file:`cam_demo.py`,`predict.py`
* user interface,just to predict image,using pyqt5 and wxpython respectively

## Image Spiders module
* folder: spiders 
* file: `spider_baidu.py` , `spider_douguo.py`
* use spiders to crawl raw images from the internet

## Image detection module
* file: `detect.py`
* run in command line to detect

# Requirements
```bash
$ pip install -r requirements.txt
```
# Dependency
* keras
* tensorflow-gpu
* numpy
* opencv-python
* pillow
* matplotlib (used to show parameter change)
* pyqt5 or wxpython (UI)
* graphviz and pydot (used to save network model)
* h5py (used to save model .h5 file)

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

## Micro PC Ⅲ
* Raspbian(Debian based)
* etc.