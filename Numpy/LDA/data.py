import matplotlib.pyplot as plt 
import numpy as np
import os
import pandas as pd
import numpy.random as random


os.makedirs(os.path.join('./' , 'data') , exist_ok = True)
data_file = os.path.join('./' , 'data' , 'melon.csv')


w_true = np.array([1. , 2.])
b_true = np.array(0.)


data_rou = random.normal(0.05,  0.25, 100) + 0.85
data_sugar = random.normal(0.15 , 0.25 , 100) + 0.45
data_para = np.block([[data_rou] , [data_sugar]])



def Is_Good(data):
    return np.dot(w_true , data) > 2.

data_isgood = np.where(Is_Good(data_para) , 1,0)
data = np.block([[data_para] , [data_isgood]])


with open(data_file , 'w') as f:
    f.write('Rou,Sugar Content,Is_Good\n')
    for i in range(len(data[0])):
        str = f'{data[0,i]},{data[1,i]},{data[2,i]}\n'
        f.write(str)




