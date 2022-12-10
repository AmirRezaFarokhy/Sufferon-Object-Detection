import cv2 
import os  

PATH_TRAIN_YOLO = "yolo_sofferon/real_shape_train/"
PATH_VALID_YOLO = "yolo_sofferon/real_shape_val/"
SIZE_SHAPE = (720, 720)
SHOW_EVERY = 10

# in yolov5 we must prepare our image and label them. labeled with makesense.ai. use google colab GPU for do fast with yolov5.
def Convert720Pixel(path, sizes, validation=False):
    for cnt, image_name in enumerate(os.listdir(path)):
        image_name_path = os.path.join(path, image_name)
        img = cv2.imread(image_name_path, cv2.IMREAD_COLOR)
        img = cv2.resize(img, sizes)
        if cnt % SHOW_EVERY==0:
            print(f"{cnt} images complete!!!")
            
        if not validation:
            if os.path.exists('train'):
                cv2.imwrite(f"yolo_sofferon/train/{image_name}", img)
            else:
                os.makedirs('train')
                cv2.imwrite(f"yolo_sofferon/train/{image_name}", img)
        else:
            if os.path.exists('val'):
                cv2.imwrite(f"yolo_sofferon/val/{image_name}", img)

            else:
                os.makedirs('val')
                cv2.imwrite(f"yolo_sofferon/val/{image_name}", img)            



Convert720Pixel(PATH_TRAIN_YOLO, SIZE_SHAPE, validation=False)
print(f"training image comploted... \n")

Convert720Pixel(PATH_VALID_YOLO, SIZE_SHAPE, validation=True)
print(f"validation image comploted... \n")
