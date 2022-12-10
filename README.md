# Sufferon-Object-Detection
Identifying saffron flower using artificial intelligence and crabbing its period using yolov5.

### Description to use
1) you must create tow folder `flowers_train` and `valid_flower` and put data image into them. 3 type of images Flower of terror, sufferon and Half open flower. 
2) run the `preprocessing_data.py` python file for prepare data to fit the model. 
3) run the `test_pickle_together.py` python file for check the data shape is (free, 150, 150, 3) or note. 
4) run the `models.py` python file to craete the model and fit it to the data and save the model.
5) run the `load_models.py` python file to use the save model we created and saved in `models.py`.
6) if the result in  `load_models.py` is sufferon we must use the yolov5.
7) Check the site below to learn more about YOLOv5: 
https://github.com/ultralytics/yolov5

### Requirements
Running `sufferon-detection` requires:
* Python 3.8 (tested under Python 3.10.4)
* Using Google Colab GPU
* Using YOLOv5 
* tensorflow 2.9.1
* keras 2.9.0
* numpy 1.22.3
* cv2 4.6.0
* matplotlib 3.5.1

### Installation
In order to test the script please run the following commands:
```sh
# install numpy
pip install numpy
# install cv2
pip install opencv-python
# install tensoflow
pip install tensoflow
# install matplotlib
pip install matplotlib
# install keras
pip install keras
# Clone and install YOLOYv5
git clone https://github.com/ultralytics/yolov5  
cd yolov5
pip install -r requirements.txt  
```

### Description of the library
A summary of explanations about libraries:
* The `numpy` it makes it easy to do math and work with matrices.
* The `opencv-python` Python  library helps to Working with Image and Videos data.
* The `tensorflow and keras` Python  library helps to use Multiplying the weights in the matrix and artificial intelligence algorithms.
* The `matplotlib` Python  library helps to show plot and image data.
* The `YOLOYv5` Python  library helps to Detec object for sufferon images.

