# -*- coding: utf-8 -*-
"""
Created on Sat May 29 19:26:19 2021

@author: cjove
"""


import streamlit as st
import pandas as pd
import base64
import data_extraction_web as de
import dataset_generation_step_web as dgs
import dataset_generation_s2s_web as dgs2s
import numpy as np
import pickle

def app():
    pickle_in = open('multioutput_regression_s2s_2.pkl', 'rb')
    regressor = pickle.load(pickle_in)

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

    header = st.beta_container()
    dataset = st.beta_container()
    model_application = st.form(key='second_step')
 

        
    with header:
        st.title('Muscle activity prediction during sit-to-stand')
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


    with model_application:
        st.header('In this second section of the app you will be able to apply a   machine learning algorithm')
        st.markdown('This model requires that you have collected data from: Waist, Right TA, Left_TA, Left_MG, Left_ST')
        st.markdown('Please complete the following information:')
        waist = st.text_input('Which columns of your dataset correspond to   the Waist IMU Gx?', '29')
        waist_rest = st.text_input('Which columns of your dataset correspond to   the rest of the waist imu?', '24,25,26,27,28')
        right_TA = st.text_input('Which columns of your dataset correspond to   the Right_TA emg?', '30')
        left_TA = st.text_input('Which columns of your dataset correspond to   the Left_TA emg?', '37')
        left_MG = st.text_input('Which columns of your dataset correspond to   the Left_MG emg?', '38')
        left_ST = st.text_input('Which columns of your dataset correspond to   the Left_ST emg?', '41')
        
        submit_button_model = st.form_submit_button(label='Submit')
        
        if submit_button_model:
            if len(dataframes) != 0 :
                try:
                    #list_emg_time = {'wa':wa,'var':var,'rms':rms,'mav':mav,'wfl':wfl,'zc':zc,'ssc':ssc, 'ssi':ssi }
                    #list_emg_freq = {'mdf':mdf,'mf':mnf,'she':se,'spe':spe,'svde':svde}
                    #list_imu = {'min':min_imu, 'max': max_imu,'std':std_imu, 'fin':final_imu, 'ini':init_imu,'mean': mean_imu}
                    imu_list = waist_rest.split(',')
                    #emg_list = emg_cols.split(',')
                    imu_list = [int(i) for i in imu_list]
                    #emg_list = [int(i) for i in emg_list]
                    waist = int(waist)
                    right_TA = int(right_TA)
                    left_TA = int(left_TA)
                    left_MG = int(left_MG)
                    left_ST = int(left_ST)
                    imu_list.append(waist)
                    emg_list= [right_TA, left_TA, left_MG, left_ST]
                #name = pd.read_csv(df[0])
                    names = dataframes[0].columns
                #gy = names[gy]
                    featurename_emg = ['mav','ssi','var', 'rms','wfl','zc' ,'ssc','wa' ,
                   'mdf','mf' ,'she', 'spe', 'svde']
        
                    featurename_imu = ['min', 'max', 'mean', 'std', 'initial', 'final']
                
        
                    #gy = int(gy)
                    gy = names[waist]
                    data = de.data_extraction(dataframes,names, emg_list, imu_list)
                    emg_filt = data[0]
                    acc_filt = data[1]
                    acc_names = data[2]
                    muscle_names = data[3]
                    #st.write(muscle_names)
                    lab_imu = dgs.create_labels_imu(acc_names, featurename_imu)
                    lab_emg = dgs.create_labels_emg(muscle_names, featurename_emg)
                    #st.write(lab_emg)
                    imu_features_s2s = dgs2s.acc_features_sit2stand(acc_filt, acc_names, gy)
                    emg_features_s2s = dgs2s.emg_features_sit2stand(emg_filt, muscle_names, acc_filt, gy)           
                    #st.write(imu_features_s2s)
                    #st.write(emg_features_s2s)
                    imu_s2s_dataframe = dgs2s.create_dataframe_s2s(imu_features_s2s, lab_imu, 0) # num_var = [len(imu_list),len(emg_list)]
                    #st.write(imu_s2s_dataframe.head())
                    emg_s2s_dataframe = dgs2s.create_dataframe_s2s(emg_features_s2s, lab_emg, 1) #,num_var = [len(imu_list),len(emg_list)]
                    results = pd.concat([imu_s2s_dataframe,emg_s2s_dataframe],axis = 1)
                    #st.write(results.head())
                    w = results.loc[:,results.columns.str.startswith(names[waist][:4])]
                    rta = results.loc[:,results.columns.str.startswith(names[right_TA])]
                    lta = results.loc[:,results.columns.str.startswith(names[left_TA])]
                    lmg = results.loc[:,results.columns.str.startswith(names[left_MG])]
                    lst = results.loc[:,results.columns.str.startswith(names[left_ST])]
        
                    pdList_r = [w,rta,lta]
                    pdList_l = [w,lmg,lst]
                    results_r = pd.concat(pdList_r,axis = 1)
                    results_l = pd.concat(pdList_l,axis = 1)
                    st.write(results_r)
                    st.write(results_l)
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
                except:
                    st.markdown('Rellena toda la información necesaria y vuelve a pulsar "Submit"')
            else:
                st.write('Please upload a file')