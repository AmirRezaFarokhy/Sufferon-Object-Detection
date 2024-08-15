import pickle 
import numpy as np
import os

PATH_TRAIN = "training_data/train.pickle"
PATH_VALID = "validation_data/valid.pickle"

# test the array we created in preprocessing_data.py
def Test_validation(path):
    with open(path, 'rb') as f:
        test = pickle.load(f)

    print(np.array(test).shape)
    for value, _ in test:
        print(np.array(value).shape)
        break

Test_validation(PATH_TRAIN)


