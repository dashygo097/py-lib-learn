import numpy as np
import numpy.random as random
import os

import matplotlib.pyplot as plt
from datasets import Dataset,DatasetDict

data_dir = "./data"
os.makedirs(data_dir, exist_ok=True)

def create_example(num , w_true , b_true):
    noise = random.normal(0 , 0.01 , size=(num , w_true.shape[0] + 1))

    x = random.random(size=(num , w_true.shape[0])) * 1 + noise[:,:-1]
    y = x @ w_true + noise[:,-1] + b_true

    return x,y

w_true = np.array([2.0 ,-4.2])
b_true = 0.5

x_train,y_train = create_example(1000 , w_true , b_true)
training_set = Dataset.from_dict({"x_axis":x_train , "y_axis" : y_train})

x_test,y_test = create_example(100 , w_true , b_true)
test_set = Dataset.from_dict({"x_axis":x_test , "y_axis":y_test})


dict = {"train":training_set , "test": test_set}
dataset = DatasetDict(dict)

dataset.save_to_disk(data_dir)



