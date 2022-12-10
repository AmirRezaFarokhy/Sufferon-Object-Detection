import pickle 
import os
import numpy as np 
import cv2
import matplotlib.pyplot as plt

import tensorflow as tf 
from tensorflow.keras.models import Sequential, model_from_json 
from tensorflow.keras.callbacks import ModelCheckpoint
from tensorflow.keras.layers import Dense, Conv2D, MaxPooling2D, Flatten, Dropout

PATH_TRAIN = "training_data/train.pickle"
PATH_VALID = "validation_data/valid.pickle"
PATH_FILE_TEST = "test/"
LIST_CATEGORIES = ["Flowerـofـterror",
				   "2_sufferon",
				   "Halfـopenـflower"]
LIST_LEARNING_RATES = [0.1, 0.01, 0.001, 0.05, 0.75]
SHAPES = (150, 150) 
EPOCH = 10
BATCH_SIZE = 16

# loading data that created in preprocessing_data.py
def load_data(path):
	with open(path, 'rb') as f:
		data = pickle.load(f)

	X = []
	y = []
	for value, label in data:
		X.append(value)
		y.append(label)
	
	return np.array(X), np.array(y)


# create model_1 in 4 conv layer and 1 dense... output layer is 3 for every 3 classes.
def CreatModelOne():
	model = Sequential()

	model.add(Conv2D(32, activation='relu', kernel_size=(3, 3), padding='Same', input_shape=(x_train.shape[1:])))
	model.add(MaxPooling2D(pool_size=(2, 2)))
	model.add(Dropout(0.2))

	model.add(Conv2D(64, activation='relu', kernel_size=(3, 3), padding='Same'))
	model.add(MaxPooling2D(pool_size=(2,2)))
	model.add(Dropout(0.2))

	model.add(Conv2D(64, activation='relu', kernel_size=(3, 3), padding='Same'))
	model.add(MaxPooling2D(pool_size=(2, 2)))
	model.add(Dropout(0.2))

	model.add(Conv2D(32, activation='relu', kernel_size=(3, 3), padding='Same'))
	model.add(MaxPooling2D(pool_size=(2, 2)))
	model.add(Dropout(0.2))

	model.add(Flatten())
	model.add(Dense(32, activation='relu'))
	model.add(Dropout(0.2))
	model.add(Dense(len(LIST_CATEGORIES), activation='softmax'))

	return model

# create model_1 in 3 conv layer and 1 dense... output layer is 3 for every 3 classes.
def CreatModelTow():
	model = Sequential()

	model.add(Conv2D(8, activation='relu', kernel_size=(3, 3), padding='Same', input_shape=(x_train.shape[1:])))
	model.add(MaxPooling2D(pool_size=(2, 2)))
	model.add(Dropout(0.2))

	model.add(Conv2D(16, activation='relu', kernel_size=(3, 3), padding='Same'))
	model.add(MaxPooling2D(pool_size=(2,2)))
	model.add(Dropout(0.2))

	model.add(Conv2D(16, activation='relu', kernel_size=(3, 3), padding='Same'))
	model.add(MaxPooling2D(pool_size=(2, 2)))
	model.add(Dropout(0.2))

	model.add(Flatten())
	model.add(Dense(16, activation='relu'))
	model.add(Dropout(0.2))

	model.add(Dense(len(LIST_CATEGORIES), activation='softmax'))

	return model


# we saving the model to use it in the future.
def SaveModels():
	if os.path.exists('Model_Weight'):
		model_json = model.to_json()
		with open("Model_Weight/model.json", "w") as json_file:
			json_file.write(model_json)
		model.save_weights("Model_Weight/model.h5")
		print("Saved model...")
	else:
		os.makedirs('Model_Weight')
		model_json = model.to_json()
		with open("Model_Weight/model.json", "w") as json_file:
			json_file.write(model_json)
		model.save_weights("Model_Weight/model.h5")
		print("Saved model...")


# we check each accuracy of classes to see the result of each class.
def AccuracyEachGroup(y_true, y_pred):
	def func(data):
		y_index = [i for i in range(len(LIST_CATEGORIES))]
		cnt = 0
		y_tru = []
		for i in range(len(LIST_CATEGORIES)):
			lst = []
			for j in data:
				if j==y_index[i]:
					lst.append(j)
			y_tru.append(lst)
		return y_tru

	actual = func(y_true)
	predict = func(y_pred)
	cnt = 0
	for ac, pre in zip(actual, predict):
		length_max = max(len(ac), len(pre))
		length_min = min(len(ac), len(pre)) 
		accuracy = length_min/length_max
		print(f"this {LIST_CATEGORIES[cnt]} --- group accuracy is {accuracy} \n")
		cnt += 1


# we do this function to find the best learning rate for the model.
def BestLearningRate(learning_rate, epoch_testing=20):
	tf.random.set_seed(1234)
	models = [None] * len(learning_rate)
	for i in range(len(learning_rate)):
		learn_rate = learning_rate[i]
		models[i] = CreatModelTow()
		models[i].compile(
						loss='sparse_categorical_crossentropy',
						optimizer=tf.keras.optimizers.Adam(lr=learn_rate),
						metrics=['accuracy'])

		models[i].fit(x_train, y_train, epochs=epoch_testing)
		print(f"Finished Learning rate -- {learn_rate}")


# plot the loss function and accuracy for model.
def VisulizeModel():
	plt.plot(hist.history["loss"], label="train_loss")
	plt.plot(hist.history["val_loss"], label="val_loss")
	plt.legend(loc="upper right")
	plt.show()

	plt.plot(hist.history["accuracy"], label="train_acc")
	plt.plot(hist.history["val_accuracy"], label="val_acc")
	plt.legend(loc="upper right")
	plt.show()


# predict some image data for check the model is good or not.
def TestImages(image_path, sizes):
    image = []
    X_test = []
    y_test_true = []
    for category, path in enumerate(os.listdir(image_path)):
        images = os.path.join(image_path, path)
        for img_pathes in os.listdir(images):
            img = cv2.imread(os.path.join(images, img_pathes), cv2.IMREAD_COLOR)
            img = cv2.resize(img, (sizes))
            img = img / 255.0
            X_test.append(img)
            y_test_true.append(category)
	
    return np.array(X_test), np.array(y_test_true)


x_train, y_train = load_data(PATH_TRAIN)
x_valid, y_valid = load_data(PATH_VALID)

print(f"shape of x train {x_train.shape}")
print(f"shape of y train {y_train.shape}")
print(f"shape of x valid {x_valid.shape}")
print(f"shape of y valid {y_valid.shape}")

#model = CreatModelOne()
model = CreatModelTow()
model.compile(optimizer=tf.keras.optimizers.Adam(lr=0.001), 
			  loss='sparse_categorical_crossentropy', 
			  metrics=['accuracy'])

model.summary()

if os.path.exists("Model_Weight"):
	checkpoint = ModelCheckpoint(filepath="/Model_Weight/weights.hdf5", 
								 verbose=1, 
								 save_best_only=True)
else:
	os.makedirs("Model_Weight")
	checkpoint = ModelCheckpoint(filepath="/Model_Weight/weights.hdf5", 
								 verbose=1, 
								 save_best_only=True)

# model learning...
hist = model.fit(x_train, y_train, 
				epochs=EPOCH,
				batch_size=BATCH_SIZE,
				validation_data=[x_valid, y_valid],
				callbacks=[checkpoint])

VisulizeModel()

# check the score for model is validation data.
scores = model.evaluate(x_valid, y_valid)
print(f"score and loss for validation data is \n {scores}")

if scores[1]*100 > 90.0:

	SaveModels()

	x_test, y_test_true = TestImages(PATH_FILE_TEST, SHAPES)
	y_pred_test = model.predict(x_test)

	for pred, actual in zip(y_pred_test, y_test_true):
		print(f"""result for Test images -- predict: {np.argmax(pred)}, real: {actual}
			and the category for this images -- predict: {LIST_CATEGORIES[np.argmax(pred)]}, real: {LIST_CATEGORIES[actual]}""")

	y_pred = model.predict(x_valid)
	y_pred_class = []
	for i in y_pred:
		y_pred_class.append(np.argmax(i))

	AccuracyEachGroup(y_valid, y_pred_class)



