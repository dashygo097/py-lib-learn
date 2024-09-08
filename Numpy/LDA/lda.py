import numpy as np 
import matplotlib.pyplot as plt
import numpy.random as random
import os
import pandas as pd
import numpy.linalg as linalg

data_reg = pd.read_csv('./data/melon.csv')
print(data_reg)



def Get_Sb(data_set):
    
    ave_positive = np.array(data_set[data_set['Is_Good'] ==1.].mean())
    ave_negative = np.array(data_set[data_set['Is_Good'] ==0.].mean())
    delta = ave_negative - ave_positive

    return delta[:-1:,np.newaxis] * delta[np.newaxis , :-1:]


def Get_Sw(data_set):

    data_posi = np.array(data_set[data_set['Is_Good'] ==1.])[:,:-1:]
    data_nega = np.array(data_set[data_set['Is_Good'] ==0.])[:,:-1:]


    ave_positive = data_posi.mean(axis = 0)
    ave_negative = data_nega.mean(axis = 0)

    delta = np.zeros((ave_positive.shape[0], ave_positive.shape[0]))
    
    for (x,y) in zip(data_posi,data_nega):
        delta += (x[:,np.newaxis]-ave_positive[:,np.newaxis]) * (x[np.newaxis , :]-ave_positive[np.newaxis , :])
        delta += (y[:,np.newaxis]-ave_negative[:,np.newaxis]) * (y[np.newaxis , :]-ave_negative[np.newaxis , :])

    return delta


def Get_w(data_set):

    data_posi = np.array(data_set[data_set['Is_Good'] ==1.])[:,:-1:]
    data_nega = np.array(data_set[data_set['Is_Good'] ==0.])[:,:-1:]


    ave_positive = data_posi.mean(axis = 0)
    ave_negative = data_nega.mean(axis = 0)
    delta = ave_negative - ave_positive
    
    Sw = Get_Sw(data_set)

    U,s,Vt = linalg.svd(Sw)

    s = np.diag(s)

    Sw_i = np.dot(linalg.inv(Vt), linalg.inv(s))
    Sw_i = np.dot(Sw_i , linalg.inv(U))

    w = np.dot(Sw_i , delta)

    return w


def show_img(data_set):

    
    data_posi = np.array(data_set[data_set['Is_Good'] ==1.])[:,:-1:]
    data_nega = np.array(data_set[data_set['Is_Good'] ==0.])[:,:-1:]


    ave_positive = data_posi.mean(axis = 0)
    ave_negative = data_nega.mean(axis = 0)
    delta = ave_negative - ave_positive

    w = Get_w(data_set)
    double_points = np.array([0,1])
    double_points = double_points * [w[1]]/w[0]
    plt.plot([0,1], double_points) 


    plt.scatter(ave_positive[0],ave_positive[1], color = 'red' , marker = 'o',s = 50, label = 'posi central')
    plt.scatter(ave_negative[0],ave_negative[1], color = 'green', marker = '^' ,s = 50, label = 'nega central')
    plt.scatter(data_posi[:,0] ,data_posi[:,-1], color = 'red' , marker = 'o',s = 5, label = 'posi sample')
    plt.scatter(data_nega[:,0] ,data_nega[:,-1], color = 'green', marker = '^' ,s = 5, label = 'nega sample')

    plt.xticks(np.linspace(0 , 2 , 11))
    plt.xlabel('Intensity(g/cm^3)')
    plt.ylabel('Sugar Content')
    plt.title('LDA Analysis Example')
    plt.legend()
    plt.show()



show_img(data_reg)
