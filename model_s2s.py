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
    pickle_in = open('multioutput_regression_s2s_def.pkl', 'rb')
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
        st.title("Predicción de la actividad neuromuscular durante la transición de sedestación a bipedestación ")
        st.markdown('En esta sección de la aplicación, introduciendo en el formulario las columnas provenientes del componente Z del acelerómetro de la cadera, el componente X del giróscopo de la cadera y la actividad electromiográfica del tibial anterior y del bíceps femoral podrás predecir la actividad muscular (RMS o media cuadrática) de los siguientes 5 músculos: gemelo interno (GI), sóleo (SOL), semitendinoso (ST), vasto lateral (VL) y recto femoral (RF) durante la transición de sedestación a bipedestación')
        st.markdown('El algoritmo utilizado es Árboles extremadamente aleatorios con los siguientes parámetros: bootstrap = False, criterion = mae, max_depth = 10, min_samples_leaf = 1, min_samples_split = 5, n_estimators = 300')
        st.markdown('Esta aplicación asume que se han realizado registros bilaterales y que el primer movimiento registrado es la transición y que no existe en el registro movimientos con mayor velocidad angular de la cintura.')


    with dataset:

        st.header('Paso 1.')
        uploaded_file =st.file_uploader('Introduce aquí tu matriz de datos. Se aceptan múltiples archivos.', type = ['csv'], accept_multiple_files=True)
        if uploaded_file is not None:
            dataframes = []
            for i in uploaded_file:
                df = pd.read_csv(i)
                dataframes.append(df)
                st.write(df.head())


    with model_application:
        st.header('Paso 2. Indica a continuación qué columna pertenece a cada uno de los siguientes apartados')

        waist = st.text_input('¿Qué columna corresponde con el componente X del giróscopo en la cintura?', '29')
        waist_az = st.text_input('¿Qué columna corresponde con el componente Z del acelerómetro en la cintura?', '26')
        right_TA = st.text_input('¿Qué columna corresponde con el registro del tibal anterior derecho?', '30')
        left_TA = st.text_input('¿Qué columna corresponde con el registro del tibal anterior izquierdo?', '37')
        right_BF = st.text_input('¿Qué columna corresponde con el registro del bíceps femoral derecho?', '33')
        left_BF = st.text_input('¿Qué columna corresponde con el registro del bíceps femoral izquierdo?', '40')

        submit_button_model = st.form_submit_button(label='Submit')

        if submit_button_model:
            if len(dataframes) != 0 :
                try:
                    waist = int(waist)
                    waist_az = int(waist_az)
                    right_TA = int(right_TA)
                    left_TA = int(left_TA)
                    right_BF = int(right_BF)
                    left_BF = int(left_BF)
                    imu_list = [waist_az, waist]
                    emg_list= [right_TA, left_TA, right_BF, left_BF]
                    names = dataframes[0].columns
                    featurename_emg = ['mav','ssi','var', 'rms','wfl','zc' ,'ssc','wa' ,
                   'mdf','mf' ,'she', 'spe', 'svde']

                    featurename_imu = ['min', 'max', 'mean', 'std', 'initial', 'final']


                    gy = names[waist]
                    data = de.data_extraction(dataframes,names, emg_list, imu_list)
                    emg_filt = data[0]
                    acc_filt = data[1]
                    acc_names = data[2]
                    muscle_names = data[3]
                    lab_imu = dgs.create_labels_imu(acc_names, featurename_imu)
                    lab_emg = dgs.create_labels_emg(muscle_names, featurename_emg)
                    imu_features_s2s = dgs2s.acc_features_sit2stand(acc_filt, acc_names, gy)
                    emg_features_s2s = dgs2s.emg_features_sit2stand(emg_filt, muscle_names, acc_filt, gy)
                    imu_s2s_dataframe = dgs2s.create_dataframe_s2s(imu_features_s2s, lab_imu, 0) # num_var = [len(imu_list),len(emg_list)]
                    emg_s2s_dataframe = dgs2s.create_dataframe_s2s(emg_features_s2s, lab_emg, 1) #,num_var = [len(imu_list),len(emg_list)]
                    results = pd.concat([imu_s2s_dataframe,emg_s2s_dataframe],axis = 1)
                    waz_mean = results.loc[:,results.columns.str.startswith(str(names[waist_az])+'_mean')]
                    wgx_std = results.loc[:,results.columns.str.startswith(str(names[waist])+'_std')]
                    rta_ssc = results.loc[:,results.columns.str.startswith(str(names[right_TA])+'_ssc')]
                    rta_svde = results.loc[:,results.columns.str.startswith(str(names[right_TA])+'_svde')]
                    lta_ssc = results.loc[:,results.columns.str.startswith(str(names[left_TA])+'_ssc')]
                    lta_svde = results.loc[:,results.columns.str.startswith(str(names[left_TA])+'_svde')]
                    rbf_mav = results.loc[:,results.columns.str.startswith(str(names[right_BF])+'_mav')]
                    rbf_rms = results.loc[:,results.columns.str.startswith(str(names[right_BF])+'_rms')]
                    rbf_ssi = results.loc[:,results.columns.str.startswith(str(names[right_BF])+'_ssi')]
                    rbf_svde = results.loc[:,results.columns.str.startswith(str(names[right_BF])+'_svde')]
                    rbf_wfl = results.loc[:,results.columns.str.startswith(str(names[right_BF])+'_wfl')]
                    lbf_mav = results.loc[:,results.columns.str.startswith(str(names[left_BF])+'_mav')]
                    lbf_rms = results.loc[:,results.columns.str.startswith(str(names[left_BF])+'_rms')]
                    lbf_ssi = results.loc[:,results.columns.str.startswith(str(names[left_BF])+'_ssi')]
                    lbf_svde = results.loc[:,results.columns.str.startswith(str(names[left_BF])+'_svde')]
                    lbf_wfl = results.loc[:,results.columns.str.startswith(str(names[left_BF])+'_wfl')]

                    pdList_r = [waz_mean,wgx_std,rta_ssc,rta_svde,rbf_mav,rbf_rms,rbf_ssi,rbf_svde,rbf_wfl]
                    pdList_l = [waz_mean,wgx_std,lta_ssc,lta_svde,lbf_mav,lbf_rms,lbf_ssi,lbf_svde,lbf_wfl]
                    results_r = pd.concat(pdList_r,axis = 1)
                    results_l = pd.concat(pdList_l,axis = 1)
                    results_r = results_r.values
                    results_l = results_l.values


                    prediction_right = []
                    prediction_left = []

                    length_r = np.arange(0,len(results_r), 1)
                    pred_r = aplicar_modelo(results_r, length_r)

                    length_l = np.arange(0,len(results_l), 1)
                    pred_l = aplicar_modelo(results_l, length_l)

                    predict_df_r = pd.DataFrame(np.concatenate(pred_r), columns = ['Derecho_GI_rms','Derecho_SOL_rms', 'Derecho_ST_rms', 'Derecho_VL_rms', 'Derecho_RF_rms'])

                    predict_df_l = pd.DataFrame(np.concatenate(pred_l), columns = ['Izquierdo_GI_rms','Izquierdo_SOL_rms', 'Izquierdo_ST_rms', 'Izquierdo_VL_rms', 'Izquierdo_RF_rms'])

                    st.markdown('La predicción de la actividad muscular del lado derecho')
                    st.write(predict_df_r)

                    st.markdown('La predicción de la actividad muscular del lado izquierdo')
                    st.write(predict_df_l)

                    prediction = pd.concat([predict_df_r, predict_df_l], axis = 1)
                    if prediction is not None:
                            st.header('¡Aquí tienes tus datos procesados! :)')
                            download = download_link(prediction, 'datos_prediccion.csv', 'Pulsa aquí para descargar los datos')

                            st.markdown(download, unsafe_allow_html=True)
                    else:
                        st.markdown('Algo ha ido mal :(')
                except:
                    st.markdown('Rellena correctamente toda la información y vuelve a pulsar "Submit"')
            else:
                st.write('Por favor, introduce una matriz de datos')
