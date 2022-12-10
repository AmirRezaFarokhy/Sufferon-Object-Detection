import cv2
import os
import numpy as np
import pickle
import random
import matplotlib.pyplot as plt 

PATH_FILE_TRAIN = "flowers_train/"
PATH_FILE_VALID = "valid_flower/"
SHAPES = (150, 150) 
SHOW_EVERY = 50 

# We must preprocess data to model fit it fast.
def preprocessing(path, size_shape, validation=False):
	datasets_main = []
	datasets_augment1 = []
	datasets_augment2 = []
	datasets_augment3 = []
	big_dataset = []
	for category, folder_name in enumerate(os.listdir(path)):
		img_folder = os.path.join(path, folder_name)
		for cnt, images_path in enumerate(os.listdir(img_folder)):
			images = os.path.join(img_folder, images_path)
			if images[-3:]!='dng':
				try:
					main_img = cv2.imread(images, cv2.IMREAD_COLOR)
					main_img = cv2.resize(main_img, (size_shape))
					img_augment1 = cv2.flip(main_img, -40)
					img_augment2 = cv2.rotate(main_img, cv2.ROTATE_90_COUNTERCLOCKWISE)
					img_augment3 = cv2.flip(main_img, 13)
					main_img = main_img / 255.0
					# we augment 3 type for each image to avoid overfiting.
					img_augment3 = img_augment3 / 255.0
					img_augment2 = img_augment2 / 255.0
					img_augment1 = img_augment1 / 255.0
					datasets_main.append([main_img, category])
					datasets_augment1.append([img_augment1, category])
					datasets_augment2.append([img_augment2, category])
					datasets_augment3.append([img_augment3, category])
					if cnt % SHOW_EVERY==0:
						print(f"{cnt} images complete!!!")

				except Exception as e:
					print(str(e),"\n", images)

		big_dataset.extend(datasets_main)
		big_dataset.extend(datasets_augment1)
		big_dataset.extend(datasets_augment2)
		big_dataset.extend(datasets_augment3)
		random.shuffle(big_dataset) # shuffling data to model can fit very well.
		if validation:
			if os.path.exists('validation_data'):
				with open("validation_data/valid.pickle", "wb") as file: 
					pickle.dump(big_dataset, file) # saving data into the pickle format for reading and fast loading in python
			else:
				os.makedirs("validation_data")
				with open("validation_data/valid.pickle", "wb") as file:
					pickle.dump(big_dataset, file)
		else:
			if os.path.exists('training_data'):
				with open("training_data/train.pickle", "wb") as file:
					pickle.dump(big_dataset, file)
			else:
				os.makedirs("training_data")
				with open("training_data/train.pickle", "wb") as file:
					pickle.dump(big_dataset, file)
		

preprocessing(PATH_FILE_TRAIN, SHAPES, validation=False)
print(f"training data comploted... \n")

preprocessing(PATH_FILE_VALID, SHAPES, validation=True)
print(f"validation data comploted...")


