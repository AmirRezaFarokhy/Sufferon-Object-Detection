import cv2 
import numpy as np 

import tensorflow as tf 
from tensorflow.keras.models import load_model, model_from_json
from YoloModel import RealTimeObjectDetection

PATH_TEST = ""
PATH_MODEL = "Model_Weight/"
PATH_YOLO_MODEL = 'Model_Weight/best.pt'
LIST_CATEGORIES = ["Flowerـofـterror",
                   "2_sufferon",
                   "Halfـopenـflower"]
SHAPES = (150, 150)


# model created and how many data we want can check.
def PreprocessingTest(test_path, size_shape):
    img_test = cv2.imeard(test_path, cv2.IMREAD_COLOR)
    X_test = cv2.resize(img_test, size_shape)
    X_test = X_test / 255.0
    return X_test 

    
# load model one for check.
# if model predict sufferon we must crop the image. we must do yolov5 model for sufferon flowers image. 
def PredictTestOne(model_path):
    x_test = PreprocessingTest(PATH_TEST)
    model = load_model(f"{model_path}/weights.hdf5")
    print("Loaded model...")
    y_test = model.predict(x_test)
    y_test_category = np.argmax(y_test)
    print(f"the model say it is {LIST_CATEGORIES[y_test_category]} Flower :)")
    if LIST_CATEGORIES[y_test_category]=='2_sufferon':
        print("Yes it is sufferon --- you must do YOLOv5 for Object Detection sufferon...")
    else:
        print("No... it is another type of flower...")


# load model tow for chekc.
# if model predict sufferon we must crop the image. we must do yolov5 model for sufferon flowers image.      
def PredictTestTow(model_path):
    json_file = open(f'{model_path}/model.json', 'r')
    loaded_model_json = json_file.read()
    json_file.close()
    loaded_model = model_from_json(loaded_model_json)
    loaded_model.load_weights(f"{model_path}/model.h5")
    print("Loaded model...")
    x_test = PreprocessingTest(PATH_TEST)
    y_test = model.predict(x_test)
    y_test_category = np.argmax(y_test)
    print(f"the model say it is {LIST_CATEGORIES[y_test_category]} Flower :)")
    if LIST_CATEGORIES[y_test_category]=='2_sufferon':
        print("Yes it is sufferon --- you must do YOLOv5 for Object Detection sufferon...")
    else:
        print("No... it is another type of flower...")
        
        
 
# load model tow for chekc.
# if model predict sufferon we must crop the image. we must do yolov5 model for sufferon flowers image.      
def PredictTestThree(model_path):
    json_file = open(f'{model_path}/model_tow.json', 'r')
    loaded_model_json = json_file.read()
    json_file.close()
    loaded_model = model_from_json(loaded_model_json)
    loaded_model.load_weights(f"{model_path}/model_tow.h5")
    print("Loaded model...")
    x_test = PreprocessingTest(PATH_TEST)
    y_test = model.predict(x_test)
    y_test_category = np.argmax(y_test)
    print(f"the model say it is {LIST_CATEGORIES[y_test_category]} Flower :)")
    if LIST_CATEGORIES[y_test_category]=='2_sufferon':
        print("Yes it is sufferon --- you must do YOLOv5 for Object Detection sufferon...")
    else:
        print("No... it is another type of flower...")
        
        
def Yolov5Model(camera_index=0, model_path):
    model_detection = RealTimeObjectDetection(capture_index=camera_index, model_name=model_path)
    model_detection()
    
    

PredictTestOne(PATH_MODEL)
print("model One Done!!!")

PredictTestTow(PATH_MODEL)
print("model Tow Done!!!")

PredictTestThree(PATH_MODEL)
print("model Three Done!!!")

Yolov5Model(camera_index=0, PATH_YOLO_MODEL)
print("yolo model detect!!!")


