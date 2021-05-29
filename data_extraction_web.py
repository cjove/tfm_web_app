# -*- coding: utf-8 -*-
"""
Created on Thu May 13 15:15:58 2021

@author: cjove
"""

import pandas as pd
import numpy as np
from itertools import groupby
from operator import itemgetter

import signal_processing as sp

def data_extraction_step(uploaded_file, names, emg_cols,imu_cols, mode):
    #print('Empezando la importación de los datos')
    muscle_names = names[emg_cols]
    acc_names = names[imu_cols]
    #gon_names = names[44:-1]
    #dataframe_test = pd.DataFrame(columns= muscle_names)
    #dir_dataset = 'C:/Users/cjove/Desktop/TFM/Dataset/'
    #results = []
    #name = []
    #circuits = np.arange(1,51,1)
    mus = []
    acc = []
    #goni = []
    index_steps = []
    length = np.arange(0,len(uploaded_file),1)

    for i in length:
        test = uploaded_file[i]
        musc =[]
        for muscle in muscle_names:
            m_filtered = sp.butter_bandpass_filter(np.asarray(test[muscle]))
            #print(len(filtered))
            musc.append(m_filtered)
            #name.append(semi_dir+str(rep)+
            #          "_raw.csv")
        acce = []
        for accel in acc_names:
            a_filtered = sp.butter_lowpass_filter(np.asarray(test[accel]))
            acce.append(a_filtered)
        
        mus.append(musc)
        acc.append(acce)
        #ind = test['Mode'][test['Mode']==1].index
        ind = test[mode][test[mode]==1].index
        ind_div = [list(map(itemgetter(1), g)) for k, g in groupby(enumerate(ind), lambda x: x[0]-x[1])]
        index_steps.append(ind_div)
                   
        
    
    emg_filt = pd.DataFrame(mus, columns = muscle_names)
    acc_filt = pd.DataFrame(acc, columns = acc_names)

    #print('Importación terminada')
    return emg_filt, acc_filt, acc_names, muscle_names,index_steps

"""
Remember index steps are the reference to make sure we are in the walking segment
"""

def data_extraction(uploaded_file, names, emg_cols,imu_cols):
    #print('Empezando la importación de los datos')
    muscle_names = names[emg_cols]
    acc_names = names[imu_cols]
    #gon_names = names[44:-1]
    #dataframe_test = pd.DataFrame(columns= muscle_names)
    #dir_dataset = 'C:/Users/cjove/Desktop/TFM/Dataset/'
    #results = []
    #name = []
    #circuits = np.arange(1,51,1)
    mus = []
    acc = []
    #goni = []
    #index_steps = []
    length = np.arange(0,len(uploaded_file),1)

    for i in length:
        test = uploaded_file[i]
        musc =[]
        for muscle in muscle_names:
            m_filtered = sp.butter_bandpass_filter(np.asarray(test[muscle]))
            #print(len(filtered))
            musc.append(m_filtered)
            #name.append(semi_dir+str(rep)+
            #          "_raw.csv")
        acce = []
        for accel in acc_names:
            a_filtered = sp.butter_lowpass_filter(np.asarray(test[accel]))
            acce.append(a_filtered)
        
        mus.append(musc)
        acc.append(acce)
        #ind = test['Mode'][test['Mode']==1].index
        #ind_div = [list(map(itemgetter(1), g)) for k, g in groupby(enumerate(ind), lambda x: x[0]-x[1])]
        #index_steps.append(ind_div)
                   
        
    
    emg_filt = pd.DataFrame(mus, columns = muscle_names)
    acc_filt = pd.DataFrame(acc, columns = acc_names)

    #print('Importación terminada')
    return emg_filt, acc_filt, acc_names, muscle_names #,index_steps


