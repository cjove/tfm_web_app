# -*- coding: utf-8 -*-
"""
Created on Sun May 23 09:29:13 2021

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

from typing import Dict

def app():
    pickle_in = open('multioutput_regression_s2s_2.pkl', 'rb')
    regressor = pickle.load(pickle_in)

    header = st.beta_container()
    dataset = st.beta_container()
    form = st.form(key='my_form')
    #   form2 = st.form(key='second_step')
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
        st.title('Extracción de variables durante la transición de sedestación a bipedestación')
        st.text('Escoge aquellas variables que sean de tu interés e indica a que hemicuerpo y tipo de sensor corresponden las columnas de tu matriz de datos. Es fundamental que indiques correctamente que columna corresponde al componente X del giróscopo de la cintura')
        st.markdown('Esta aplicación asume que se han realizado registros bilaterales y que al menos se han recogido el componente X del giróscopo de la cintura y algun músculos bilateralmente para poder ser utilizada. También asume que el primer movimiento registrado es la transición y que no existe en el registro movimientos con mayor velocidad angular de la cintura')
        st.markdown("""
        El procedimiento interno de la aplicación es:
        * Detección de la transción de sedestación a bipedestación
        * Cálculo de las variables para cada transición
        * Cada fila del csv descargable representa las variables obtenidas en un levantamiento. Se presentan de forma conjunta los resultados del lado derecho e izquierdo.
        """)


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
        #emg_fs = fs.selectbox('What´s the sampling frequency of your sEMG sensor', options= [250,512,1024,2048], index = 2)
        #imu_fs = fs.selectbox('What´s the sampling frequency of your IMU sensor', options= [250,512,1024,2048], index = 2)
        emg_cols = cols.text_input('¿Qué columnas pertenecen a los sensores de electromiografía?', '30,31,32,33,34,35,36,37,38,39,40,41,42,43')
        imu_cols = cols.text_input('¿Qué columnas pertenecen a los sensores inerciales?', '1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25,26,27,28,29')
        gy = cols.text_input('¿Qué columna pertene al componente X de la cintura?', '29')

        submit_button = st.form_submit_button(label='Submit')

        if submit_button:
            if len(dataframes) != 0 :
                try:
                    list_emg_time = {'wa':wa,'var':var,'rms':rms,'mav':mav,'wfl':wfl,'zc':zc,'ssc':ssc, 'ssi':ssi }
                    list_emg_freq = {'mdf':mdf,'mf':mnf,'she':se,'spe':spe,'svde':svde}
                    list_imu = {'min':min_imu, 'max': max_imu,'std':std_imu, 'fin':final_imu, 'ini':init_imu,'mean': mean_imu}
                    imu_list = imu_cols.split(',')
                    emg_list = emg_cols.split(',')
                    imu_list = [int(i) for i in imu_list]
                    emg_list = [int(i) for i in emg_list]
                #name = pd.read_csv(df[0])
                    names = dataframes[0].columns
                #gy = names[gy]
                    featurename_emg = ['mav','ssi','var', 'rms','wfl','zc' ,'ssc','wa' ,
                   'mdf','mf' ,'she', 'spe', 'svde']

                    featurename_imu = ['min', 'max', 'mean', 'std', 'initial', 'final']
                #data = de.data_extraction(dataframes,names, emg_list, imu_list)
                #emg_filt = data[0]
                #acc_filt = data[1]
                #acc_names = data[2]
                #muscle_names = data[3]
                #index_steps = data[4]
                #st.write(emg_filt.head())
                #st.write(acc_filt.head())
                #st.write(acc_names)
                #st.write(muscle_names)

                    emg_time_feature = [k for k, v in list_emg_time.items() if v is True]
                    emg_freq_feature = [k for k, v in list_emg_freq.items() if v is True]
                    emg_features = emg_time_feature + emg_freq_feature
                    imu_feature = [k for k, v in list_imu.items() if v is True]

                    filtered_emg_time = [k for k, v in list_emg_time.items() if v is False]
                    filtered_emg_freq = [k for k, v in list_emg_freq.items() if v is False]
                    filtered_imu = [k for k, v in list_imu.items() if v is False]
                    selected = filtered_emg_time + filtered_emg_freq + filtered_imu
                #lab_imu = dgs.create_labels_imu(acc_names, imu_feature)
                #lab_emg = dgs.create_labels_emg(muscle_names, emg_features)
                #st.markdown(lab_imu)
                #st.markdown(lab_emg)

                    gy = int(gy)
                    gy = names[gy]
                    data = de.data_extraction(dataframes,names, emg_list, imu_list)
                    emg_filt = data[0]
                    acc_filt = data[1]
                    acc_names = data[2]
                    muscle_names = data[3]

                    lab_imu = dgs.create_labels_imu(acc_names, featurename_imu)
                    lab_emg = dgs.create_labels_emg(muscle_names, featurename_emg)
                    #st.write(lab_emg)
                    imu_features_s2s = dgs2s.acc_features_sit2stand(acc_filt, acc_names, gy)
                    emg_features_s2s = dgs2s.emg_features_sit2stand(emg_filt, muscle_names, acc_filt, gy)
                    #st.write(imu_features_s2s)
                    #st.write(emg_features_s2s)
                    imu_s2s_dataframe = dgs2s.create_dataframe_s2s(imu_features_s2s, lab_imu, 0) # num_var = [len(imu_list),len(emg_list)]
                    emg_s2s_dataframe = dgs2s.create_dataframe_s2s(emg_features_s2s, lab_emg, 1) #,num_var = [len(imu_list),len(emg_list)]
                    results = pd.concat([imu_s2s_dataframe,emg_s2s_dataframe],axis = 1)
                    #st.write(results.head())
                    filtered_emg_time = [k for k, v in list_emg_time.items() if v is False]
                    filtered_emg_freq = [k for k, v in list_emg_freq.items() if v is False]
                    filtered_imu = [k for k, v in list_imu.items() if v is False]
                    selected = filtered_emg_time + filtered_emg_freq + filtered_imu
                    #st.markdown(selected)
                    for i in selected:
                        results=results[results.columns.drop(list(results.filter(regex= i)))]

                #r= results.filter(regex= selected).columns
                    st.write(results)
                    if results is not None:
                            st.header('¡Aquí tienes tus datos procesados! :)')
                            st.markdown('This are the option you selected:' + str(submit_button))
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
