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
from PIL import Image


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
        st.title('Extracción de variables durante la marcha en superficie plana')
        st.markdown('Escoge aquellas variables que sean de tu interés e indica a que hemicuerpo y tipo de sensor corresponden las columnas de tu matriz de datos. Es fundamental que indiques correctamente que columna corresponde al componente Y del giróscopo de la tibia derecha e izquierda.')
        st.markdown('Esta aplicación asume que se han realizado registros bilaterales y que al menos se han recogido el componente Y del giróscopo de cada tibia y algun músculos bilateralmente para poder ser utilizada.')

        st.markdown("""
        El procedimiento interno de la aplicación es:
        * Detección de las zancadas en cada pierna
        * Cálculo de las variables para cada zancada
        * Cada fila del csv descargable representa las variables obtenidas en una zancada. Se presentan de forma conjunta los resultados del lado derecho e izquierdo.
        """)
        st.markdown('Para poder utilizar correctamente la aplicación es necesaria tener una columna en la matriz de datos que indique el movimiento que se esta relizando (columna "Modo").La columna modo es necesaria para garantizar que si se han realizado otras actividades distintas a la marcha en el registro introducido se discriminan los pasos correctamente. Para ello crea una columna con nombre "mode" y rellena con 1´s cada punto de la señal que correspoda a la marcha en superficie plana. Si todo el registro es de marcha en plano, rellena la columna de 1´s en su totalidad')

    with dataset:
        st.header('Paso 1.')
        uploaded_file =st.file_uploader('Introduce aquí tu matriz de datos', type = ['csv'],
                         accept_multiple_files=True)
        if uploaded_file is not None:
            dataframes = []
            for i in uploaded_file:
                df = pd.read_csv(i)
                dataframes.append(df)
                st.write(df.head())

    with form :
        st.header('Paso 2. Selecciona las variables de interés')
        st.markdown('**¿Qué variables quieres obtener de tu/s sensor/es inerciales?**')
        min_imu = st.checkbox('Minimum value')
        max_imu = st.checkbox('Maximum value')
        std_imu = st.checkbox('Standard deviation')
        final_imu = st.checkbox('Final value')
        init_imu = st.checkbox('Initial value')
        mean_imu = st.checkbox('Mean value')

        st.markdown('**¿Qué variables quieres obtener de tu/s sensor/es de electromiografía?**')
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

        st.header('Paso 3. Indica a continuación a que tipo de sensor y lado corresponde cada columna de la que quieras extraer variables')
        st.markdown('Introduce el valor numérico al que pertenece tu columna, siendo la primera 0. Separa los elementos por "," siguiendo el ejemplo de los valores que aparecen por defecto.')
        fs, cols = st.beta_columns(2)

        gy1 = fs.text_input('¿Qué columna corresponde al componente Y del giróscopo de la tibia? Derecho primero', '3,15')
        r_labels = fs.text_input('¿Qué columnas pertenecen a los sensores inerciales del lado derecho?', '0,1,2,4,5,6,7,8,9,10,11')
        l_labels = fs.text_input('¿Qué columnas pertenecen a los sensores inerciales del lado izquierdo y/o cintura? ','12,13,14,16,17,18,19,20,21,22,23,24,25,26,27,28,29')
        r_labels_emg = fs.text_input('¿Qué columnas pertenecen a los músculos del lado derecho?', '30,31,32,33,34,35,36')
        l_labels_emg = cols.text_input('¿Qué columnas pertenecen a los músculos del lado izquierdo?','37,38,39,40,41,42,43')
        mode = cols.text_input('¿Qué columna pertenece al modo?','48')
        #emg_fs = cols.selectbox('What´s the sampling frequency of your sEMG sensor', options= [250,512,1024,2048], index = 2)
        #imu_fs = cols.selectbox('What´s the sampling frequency of your IMU sensor', options= [250,512,1024,2048], index = 2)
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
                            st.header('¡Aquí tienes tus datos procesados! :)')
                            #st.markdown('This are the option you selected:' + str(submit_button))
                            download = download_link(results, 'datos_procesados.csv', 'Pulsa aquí para descargar los datos')
                            st.markdown(download, unsafe_allow_html=True)
                        #if st.button('Download Dataframe as CSV'):
                         #   if uploaded_file is not None:
                          #      download = download_link(results, 'YOUR_DF.csv', 'Click here to download data!')
                           #     st.markdown(download, unsafe_allow_html=True)
                    else:
                        st.markdown('Algo ha ido mal :(')

                except:
                    st.markdown('Rellena correctamente toda la información y vuelve a pulsar "Submit"')
            else:
                st.write('Por favor, introduce una matriz de datos')
