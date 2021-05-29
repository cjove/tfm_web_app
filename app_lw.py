# -*- coding: utf-8 -*-
"""
Created on Sun May 16 19:21:43 2021

@author: cjove
"""

import streamlit as st
import pandas as pd
import base64
import data_extraction_web as de
import dataset_generation_step_web as dgs

import numpy as np
import pickle

from typing import Dict
def app():
    pickle_in = open('multioutput_regression_step.pkl', 'rb')
    regressor = pickle.load(pickle_in)
    
    header = st.beta_container()
    dataset = st.beta_container()
    form = st.form(key='my_form2')
    processed = st.beta_container()
    model_application = st.form(key='second_step')
    
    def aplicar_modelo(data, length):
        predict = []
        for x in length:
            input_data = data[x].reshape(1, -1)
            prediction = regressor.predict(input_data)
            predict.append(prediction)
        return predict
    
    def download_link(object_to_download, download_filename, download_link_text):
        """
        Generates a link to download the given object_to_download.
    
        object_to_download (str, pd.DataFrame):  The object to be downloaded.
        download_filename (str): filename and extension of file. e.g. mydata.csv, some_txt_output.txt
        download_link_text (str): Text to display for download link.
    
        Examples:
        download_link(YOUR_DF, 'YOUR_DF.csv', 'Click here to download data!')
        download_link(YOUR_STRING, 'YOUR_STRING.txt', 'Click here to download your text!')
    
        """
        if isinstance(object_to_download,pd.DataFrame):
            object_to_download = object_to_download.to_csv(index=False)
    
        # some strings <-> bytes conversions necessary here
        b64 = base64.b64encode(object_to_download.encode()).decode()
    
        return f'<a href="data:file/txt;base64,{b64}" download="{download_filename}">{download_link_text}</a>'
    
    with header:
        st.title('Muscle activity prediction during level wlaking')
        st.text('This application is the product of my Master´s thesis. '     
                'This application was conceived to fulfill two objectives:'    
                '1. make signal processing simpler'    
                '2. make it easier to register'    
                'biomechanical variables in clinical practice')   
    with dataset:
    
        st.header('Information obtained from sEMG and IMU sensors')
        uploaded_file =st.file_uploader('Introduce your dataset here please', type = ['csv'],
                         accept_multiple_files=True)
        if uploaded_file is not None:
            dataframes = []
            for i in uploaded_file:
                df = pd.read_csv(i)
                dataframes.append(df)
                st.write(df.head())
    
    with form :
        st.header('Variable extraction')
        st.markdown('**What features would you like to obtain from the IMU?**')
        min_imu = st.checkbox('Minimum value')
        max_imu = st.checkbox('Maximum value')
        std_imu = st.checkbox('Standard deviation')
        final_imu = st.checkbox('Final value')
        init_imu = st.checkbox('Initial value')
        mean_imu = st.checkbox('Mean value')
        
        st.markdown('**What features would you like to obtain from the sEMG?**')
        time_col, freq_col = st.beta_columns(2)
         
    
        rms = time_col.checkbox('Root Mean Square')
        mav = time_col.checkbox('Mean Absolute Value')
        ssi = time_col.checkbox('Simple Square Integral')
        wfl = time_col.checkbox('Waveform Length')
        zc = time_col.checkbox('Zero Crossings')
        ssc = time_col.checkbox('Slope Sign Change')
        var = time_col.checkbox('Variances')
        wa = time_col.checkbox('Willison Amplitude')
        
        mdf = freq_col.checkbox('Median Frequency')
        mnf = freq_col.checkbox('Mean Frequency')
        se = freq_col.checkbox('Shannon Entropy')
        spe = freq_col.checkbox('Spectral Entropy')
        svde = freq_col.checkbox('Single Value Decomposition Entropy')
    
        st.header('In order to obtain the variables you´ve just selected I need the following information')
        fs, cols = st.beta_columns(2)
    
        gy1 = fs.text_input('Which columns correspond to your Gy_Shank? Right first', '3,15')
        r_labels = fs.text_input('Which columns of your dataframe correspond to the right sided imus?', '0,1,2,4,5,6,7,8,9,10,11')
        l_labels = fs.text_input('Which columns of your dataframe correspond to the left side and/or waist imus?','12,13,14,16,17,18,19,20,21,22,23,24,25,26,27,28,29')
        r_labels_emg = fs.text_input('Which columns of your dataframe correspond to the right sided muscles emg?', '30,31,32,33,34,35,36')
        l_labels_emg = cols.text_input('Which columns of your dataframe correspond to the left side muscle emg?','37,38,39,40,41,42,43')
        mode = cols.text_input('Which column of your dataframe presents the mode?','48')
        emg_fs = cols.selectbox('What´s the sampling frequency of your sEMG sensor', options= [250,512,1024,2048], index = 2)    
        imu_fs = cols.selectbox('What´s the sampling frequency of your IMU sensor', options= [250,512,1024,2048], index = 2) 
        #emg_cols = cols.text_input('Which columns of your dataset correspond to sEMG data?', '30,31,32,33,34,35,36,37,38,39,40,41,42,43')
        #imu_cols = cols.text_input('Which columns of your dataset correspond to IMU data?', '1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25,26,27,28,29')
    
        submit_button = st.form_submit_button(label='Submit')
        
        if submit_button:
            if len(dataframes) != 0 :
                try:
                    list_emg_time = {'wa':wa,'var':var,'rms':rms,'mav':mav,'wfl':wfl,'zc':zc,'ssc':ssc, 'ssi':ssi }
                    list_emg_freq = {'mdf':mdf,'mf':mnf,'she':se,'spe':spe,'svde':svde}
                    list_imu = {'min':min_imu, 'max': max_imu,'std':std_imu, 'fin':final_imu, 'ini':init_imu,'mean': mean_imu}
                    emg_cols = r_labels_emg +','+ l_labels_emg
                    #st.write(emg_cols)
                    imu_cols = r_labels +','+ l_labels
        
                    #st.write(len(emg_list))
                #name = pd.read_csv(df[0])
                    names = dataframes[0].columns
                    names_series = pd.Series(names)
                #gy = names[gy]
                    featurename_emg = ['mav','ssi','var', 'rms','wfl','zc' ,'ssc','wa' ,
                   'mdf','mf' ,'she', 'spe', 'svde']
        
                    featurename_imu = ['min', 'max', 'mean', 'std', 'initial', 'final']
        
                    gy_step = gy1.split(',')
                    gy_list = [int(i) for i in gy_step]
                    right_shank = names[gy_list[0]]
                    left_shank = names[gy_list[1]]
                    
                    r_lab = r_labels.split(',')
                    l_lab = l_labels.split(',')
                    r_lab = [int(i) for i in r_lab]
                    l_lab = [int(i) for i in l_lab]
                    r_lab.append(gy_list[0])
                    l_lab.append(gy_list[1])
                    #n_emg_d = names_series[r_lab_emg]
                    #n_emg_i = names_series[l_lab_emg]
        
        
                    r_lab_emg = r_labels_emg.split(',')
                    l_lab_emg = l_labels_emg.split(',')
                    r_lab_emg = [int(i) for i in r_lab_emg]
                    #st.write(len(r_lab_emg))
                    #st.write(len(l_lab_emg))
                    l_lab_emg = [int(i) for i in l_lab_emg]
                    n_emg_d = names_series[r_lab_emg]
                    n_emg_i = names_series[l_lab_emg]
                    n_emg_d = list(n_emg_d)
                    n_emg_i = list(n_emg_i)
                    #st.write(n_emg_d)
                    #st.write(n_emg_i)
                    imu_list = imu_cols.split(',')
                    emg_list = emg_cols.split(',')
                    imu_list = [int(i) for i in imu_list]
                    emg_list = [int(i) for i in emg_list]
                    imu_list = imu_list + gy_list
                    mode = int(mode)
                    mode = names[mode]
                    data = de.data_extraction_step(dataframes,names, emg_list, imu_list, mode)
                    emg_filt = data[0]
                    acc_filt = data[1]
                    acc_names = data[2]
                    muscle_names = data[3]
                    index_steps = data[4]
                    imu_features = dgs.acc_features_step(acc_filt,right_shank,left_shank, acc_names, index_steps, r_lab, l_lab)
                    
                    emg_features = dgs.emg_features_step(emg_filt, muscle_names, index_steps, 
                                      acc_filt[right_shank],
                                     acc_filt[left_shank], r_lab_emg, l_lab_emg) 
        
        
                    #lab_imu = dgs.create_labels_imu(acc_names, dgs.featurename_imu )
                    #lab_emg = dgs.create_labels_emg(muscle_names, dgs.featurename_emg)
                    lab_imu = dgs.create_labels_imu(acc_names, featurename_imu)
                    lab_emg = dgs.create_labels_emg(muscle_names, featurename_emg)
                    lab_emg_d = dgs.create_labels_emg(n_emg_d, featurename_emg)
                    lab_emg_i = dgs.create_labels_emg(n_emg_i, featurename_emg)
                    #st.write(lab_emg_d)
                    #st.write(lab_emg_i)
        
                    
                                  
                    dataframe_imu_step = dgs.create_dataframe_step(imu_features, lab_imu, 0,r_lab, l_lab, r_lab_emg, l_lab_emg)   
                    dataframe_emg_step = dgs.create_dataframe_step(emg_features, lab_emg, 2, r_lab, l_lab, lab_emg_d, lab_emg_i)       
        
                    results = pd.concat([dataframe_imu_step,dataframe_emg_step],axis = 1)
                    filtered_emg_time = [k for k, v in list_emg_time.items() if v is False]
                    filtered_emg_freq = [k for k, v in list_emg_freq.items() if v is False]
                    filtered_imu = [k for k, v in list_imu.items() if v is False]
                    selected = filtered_emg_time + filtered_emg_freq + filtered_imu
        
                    for i in selected:
                        results=results[results.columns.drop(list(results.filter(regex= i)))]
        
                    #r= results.filter(regex= selected).columns
                    st.write(results)
                    if results is not None:
                            st.header('Here you have your processed data ready! :)')
                            st.markdown('This are the option you selected:' + str(submit_button))
                            download = download_link(results, 'YOUR_DF.csv', 'Click here to download data!')
                            st.markdown(download, unsafe_allow_html=True)
                        #if st.button('Download Dataframe as CSV'):
                         #   if uploaded_file is not None:
                          #      download = download_link(results, 'YOUR_DF.csv', 'Click here to download data!')
                           #     st.markdown(download, unsafe_allow_html=True)
                    else:
                        st.markdown('Something went wrong')
            
                except:
                    st.markdown('Rellena correctamente toda la información y vuelve a pulsar "Submit"')
            else:
                st.write('Please upload a file')
            
        with model_application:
            st.header('In this second section of the app you will be able to apply a   machine learning algorithm')
            st.markdown('This model requires that you have collected data from: Shank/s,   X,Y')
            st.markdown('Please complete the following information:')
            r_shank_gy = st.text_input('Which columns of your dataset correspond to   the Right Shank IMU Gy?', '3')
            l_shank_gy = st.text_input('Which columns of your dataset correspond to   the Left Shank IMU Gy? ', '15')
            r_shank_rest = st.text_input('Which columns of your dataset correspond to   the rest of the your right shank imu?', '0,1,2,4,5')
            l_shank_rest = st.text_input('Which columns of your dataset correspond to   the rest of the your left shank imu?', '12,13,14,16,17')
            right_MG = st.text_input('Which columns of your dataset correspond to   the right gluteus maximus?','31')
            left_MG = st.text_input('Which columns of your dataset correspond to   the left gluteus maximus emg?', '38')
            right_SOL = st.text_input('Which columns of your dataset correspond to   the right soleus emg?', '32')
            left_SOL = st.text_input('Which columns of your dataset correspond to   the left soleus?', '39')
            mode = st.text_input('Which column of your dataset corresponds to   the mode?', '48')
            submit_button_model = st.form_submit_button(label='Submit')
            
            if submit_button_model:
                if len(dataframes) != 0 :
                    list_emg_time = {'wa':wa,'var':var,'rms':rms,'mav':mav,'wfl':wfl,'zc':zc,'ssc':ssc, 'ssi':ssi }
                    list_emg_freq = {'mdf':mdf,'mf':mnf,'she':se,'spe':spe,'svde':svde}
                    list_imu = {'min':min_imu, 'max': max_imu,'std':std_imu, 'fin':final_imu, 'ini':init_imu,'mean': mean_imu}
                    
                    r_imu_list = r_shank_rest.split(',')
                    r_imu_list = [int(i) for i in r_imu_list]
                    
                    l_imu_list = l_shank_rest.split(',')
                    l_imu_list = [int(i) for i in l_imu_list]
                    #emg_list = [int(i) for i in emg_list]
                    mode = int(mode)
                    r_gy = int(r_shank_gy)
                    l_gy = int(l_shank_gy)
                    right_MG = int(right_MG)
                    left_MG = int(left_MG)
                    right_SOL = int(right_SOL)
                    left_SOL = int(left_SOL)
                    imu_list = [r_gy] + r_imu_list + [l_gy] + l_imu_list
                    r_lab = [r_gy] + r_imu_list
                    l_lab = [l_gy] + l_imu_list
                    r_lab_emg = [right_MG , right_SOL]
                    l_lab_emg = [left_MG , left_SOL]
                    emg_list= [right_MG , left_MG , right_SOL, left_SOL]
                #name = pd.read_csv(df[0])
                    names = dataframes[0].columns
                    names_series = pd.Series(names)
        
                    n_emg_d = names_series[r_lab_emg]
                    n_emg_i = names_series[l_lab_emg]
                    n_emg_d = list(n_emg_d)
                    n_emg_i = list(n_emg_i)
                    
        
                #gy = names[gy]
                    featurename_emg = ['mav','ssi','var', 'rms','wfl','zc' ,'ssc','wa' ,
                   'mdf','mf' ,'she', 'spe', 'svde']
        
                    featurename_imu = ['min', 'max', 'mean', 'std', 'initial', 'final']
                
        
                    #gy = int(gy)
                    mode_lw = names[mode]
                    r_gy_ = names[r_gy]
                    l_gy_ = names[l_gy]
                    data = de.data_extraction_step(dataframes,names, emg_list, imu_list, mode_lw )
                    emg_filt = data[0]
                    acc_filt = data[1]
                    acc_names = data[2]
                    muscle_names = data[3]
                    index_steps = data[4]
                    #st.write(muscle_names)
                    lab_imu = dgs.create_labels_imu(acc_names, featurename_imu)
                    lab_emg = dgs.create_labels_emg(muscle_names, featurename_emg)
                    lab_emg_d = dgs.create_labels_emg(n_emg_d, featurename_emg)
                    lab_emg_i = dgs.create_labels_emg(n_emg_i, featurename_emg)
                    #st.write(lab_emg)
                    imu_features = dgs.acc_features_step(acc_filt,r_gy_,l_gy_, acc_names, index_steps, r_lab, l_lab)
                    emg_features = dgs.emg_features_step(emg_filt, muscle_names, index_steps, 
                                      acc_filt[r_gy_],
                                     acc_filt[l_gy_], r_lab_emg, l_lab_emg)           
                    #st.write(imu_features_s2s)
                    #st.write(emg_features_s2s)
                    dataframe_imu_step = dgs.create_dataframe_step(imu_features, lab_imu, 0,r_lab, l_lab, r_lab_emg, l_lab_emg) # num_var = [len(imu_list),len(emg_list)]
                    #st.write(imu_s2s_dataframe.head())
                    dataframe_emg_step = dgs.create_dataframe_step(emg_features, lab_emg, 1,r_lab, l_lab, lab_emg_d, lab_emg_i) #,num_var = [len(imu_list),len(emg_list)]
                    results = pd.concat([dataframe_imu_step,dataframe_emg_step],axis = 1)
                    #st.write(results.head())
                    r_shank = results.loc[:,results.columns.str.startswith(r_gy_)]
                    r_shank = r_shank.reindex(sorted(r_shank), axis = 1)
                    
                    r_shank_1 = results.loc[:,results.columns.str.startswith(names[r_imu_list[0]])]
                    r_shank_1 = r_shank_1.reindex(sorted(r_shank_1), axis = 1)
        
                    r_shank_2 = results.loc[:,results.columns.str.startswith(names[r_imu_list[1]])]
                    r_shank_2 = r_shank_2.reindex(sorted(r_shank_2), axis = 1)
        
                    r_shank_3 = results.loc[:,results.columns.str.startswith(names[r_imu_list[2]])]
                    r_shank_3 = r_shank_3.reindex(sorted(r_shank_3), axis = 1)
                              
                    r_shank_4 = results.loc[:,results.columns.str.startswith(names[r_imu_list[3]])]
                    r_shank_4 = r_shank_4.reindex(sorted(r_shank_4), axis = 1)
                    #st.write(r_shank_4)
                    r_shank_5 = results.loc[:,results.columns.str.startswith(names[r_imu_list[4]])]
                    r_shank_5 = r_shank_5.reindex(sorted(r_shank_5), axis = 1)
                    
                    l_shank = results.loc[:,results.columns.str.startswith(l_gy_)]
                    l_shank = l_shank.reindex(sorted(l_shank), axis = 1)
                    
                    l_shank_1 = results.loc[:,results.columns.str.startswith(names[l_imu_list[0]])]
                    l_shank_1 = l_shank_1.reindex(sorted(l_shank_1), axis = 1)  
                    
                    l_shank_2 = results.loc[:,results.columns.str.startswith(names[l_imu_list[1]])]
                    l_shank_2 = l_shank_2.reindex(sorted(l_shank_2), axis = 1)
                    
                    l_shank_3 = results.loc[:,results.columns.str.startswith(names[l_imu_list[2]])]
                    l_shank_3 = l_shank_3.reindex(sorted(l_shank_3), axis = 1)
                    
                    l_shank_4 = results.loc[:,results.columns.str.startswith(names[l_imu_list[3]])]
                    l_shank_4 = l_shank_4.reindex(sorted(l_shank_4), axis = 1)            
        
                    l_shank_5 = results.loc[:,results.columns.str.startswith(names[l_imu_list[4]])]
                    l_shank_5 = l_shank_5.reindex(sorted(l_shank_5), axis = 1)            
        
        
        
        
                    rmg = results.loc[:,results.columns.str.startswith(names[right_MG])]
                    rmg = rmg.reindex(sorted(rmg), axis = 1)            
                    
                    lmg = results.loc[:,results.columns.str.startswith(names[left_MG])]
                    lmg = lmg.reindex(sorted(lmg), axis = 1)              
                    
                    rsol = results.loc[:,results.columns.str.startswith(names[right_SOL])]
                    rsol = rsol.reindex(sorted(rsol), axis = 1)   
                    #st.write(rsol.head())
                    
                    lsol = results.loc[:,results.columns.str.startswith(names[left_SOL])]
                    lsol = lsol.reindex(sorted(lsol), axis = 1)     
                    #st.write(lsol.head())
                    
                    
                    pdList_r = [r_shank,r_shank_1,r_shank_2,r_shank_3,r_shank_4,r_shank_5, rmg, rsol]
                    pdList_l = [l_shank,l_shank_1,l_shank_2,l_shank_3,l_shank_4,l_shank_5, lmg, lsol]
                    
                    results_r = pd.concat(pdList_r,axis = 1)
                    results_l = pd.concat(pdList_l,axis = 1)
                    #st.write(results_r)
                    #st.write(results_l)
        
                    results_r = results_r.dropna(axis=0,how='all')
                    results_l = results_l.dropna(axis=0,how='all')
                    #st.write(results_r)
                    #st.write(results_l)
        
                    results_r = results_r.values
                    results_l = results_l.values
        
        
                    prediction_right = []
                    prediction_left = []
        
                    length_r = np.arange(0,len(results_r), 1)
                    pred_r = aplicar_modelo(results_r, length_r)
                    
                    length_l = np.arange(0,len(results_l), 1)
                    pred_l = aplicar_modelo(results_l, length_l)
                    
                    predict_df_r = pd.DataFrame(np.concatenate(pred_r), columns = ['Right_TA_rms','Right_BF_rms', 'Right_ST_rms', 'Right_VL_rms', 'Right_RF_rms'])
        
                    predict_df_l = pd.DataFrame(np.concatenate(pred_l), columns = ['Left_TA_rms','Left_BF_rms', 'Left_ST_rms', 'Left_VL_rms', 'Left_RF_rms'])
                    
                    st.markdown('Your muscle activity prediction for the right steps')
                    st.write(predict_df_r)
                    
                    st.markdown('Your muscle activity prediction for the left steps')
                    st.write(predict_df_l)
                    
                    prediction = pd.concat([predict_df_r, predict_df_l], axis = 1)
                #r= results.filter(regex= selected).columns
                    #st.write(results)
                    if prediction is not None:
                            st.header('Here you have your prediction ready! :)')
                            #st.markdown('This are the option you selected:' + str(submit_button))
                            download = download_link(prediction, 'YOUR_DF.csv', 'Click here to download data!')
        
                            st.markdown(download, unsafe_allow_html=True)
                        #if st.button('Download Dataframe as CSV'):
                         #   if uploaded_file is not None:
                          #      download = download_link(results, 'YOUR_DF.csv', 'Click here to download data!')
                           #     st.markdown(download, unsafe_allow_html=True)
                    else:
                        st.markdown('Something went wrong')

            else:
                st.write('Please upload a file')
        
        
        
        
    
    
    
    
    
    
