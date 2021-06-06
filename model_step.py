# -*- coding: utf-8 -*-
"""
Created on Sat May 29 19:26:39 2021

@author: cjove
"""
import streamlit as st

import streamlit as st
import pandas as pd
import base64
import data_extraction_web as de
import dataset_generation_step_web as dgs

import numpy as np
import pickle

from typing import Dict


def app():
    pickle_in = open('multioutput_regression_step_def.pkl', 'rb')
    regressor = pickle.load(pickle_in)

    header = st.beta_container()
    dataset = st.beta_container()
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
        st.title('Predicción de la actividad neuromuscular durante la marcha en superficie plana ')
        st.markdown('En esta sección de la aplicación, introduciendo en el formulario las columnas provenientes del componente Y del giróscopo ubicado en la tibia, la actividad electromiográfica de los tibiales anteriores y de los vastos laterales podrás predecir la actividad muscular (RMS o media cuadrática) de los siguientes 5 músculos: gemelo interno (GI), sóleo (SOL), bíceps femoral (BF), semitendinoso (ST) y recto femoral (RF) de ambas piernas')
        st.markdown('El algoritmo utilizado es Árboles extremadamente aleatorios con los siguientes parámetros: bootstrap = False, criterion = mse, max_depth = 90, min_samples_leaf = 1, min_samples_split = 5, n_estimators = 250.')
        st.markdown('Para poder utilizar correctamente la aplicación es necesario tener una columna en la matriz de datos que indique el movimiento que se esta relizando (columna "Modo").La columna modo es necesaria para garantizar que si se han realizado otras actividades distintas a la marcha en el registro introducido se discriminan los pasos correctamente. Para ello crea una columna con nombre "mode" y rellena con 1´s cada punto de la señal que correspoda a la marcha en superficie plana. Si todo el registro es de marcha en plano, rellena la columna de 1´s en su totalidad.')

    with dataset:

        st.header('Paso 1.')
        uploaded_file =st.file_uploader('Introduce aquí tu matriz de datos. Sólo se acepta un archivo por ejecución.', type = ['csv'],
                         accept_multiple_files=False)
        dataframes = []
        if uploaded_file is not None:
            dataframes = []

            df = pd.read_csv(uploaded_file)
            dataframes.append(df)
            st.write(df.head())

    with model_application:
        st.header('Paso 2. Indica a continuación qué columna pertenece a cada uno de los siguientes apartados')
        r_shank_gy = st.text_input('¿Qué columna corresponde con el componente Y del giróscopo en la tibia derecha?', '3')
        l_shank_gy = st.text_input('¿Qué columna corresponde con el componente Y del giróscopo en la tibia izquierda?', '15')
        right_TA = st.text_input('¿Qué columna corresponde con el tibial anterior de la pierna derecha?','31')
        left_TA = st.text_input('¿Qué columna corresponde con el tibial anterior de la pierna izquierda?', '38')
        right_VL = st.text_input('¿Qué columna corresponde con el vasto lateral de la pierna derecha?', '32')
        left_VL = st.text_input('¿Qué columna corresponde con el vasto lateral de la pierna izquierda?', '39')
        mode = st.text_input('¿Que columna corresponde al modo?', '48')
        submit_button_model = st.form_submit_button(label='Submit')

        if submit_button_model:
            if len(dataframes) != 0 :
                try:
                    mode = int(mode)
                    r_gy = int(r_shank_gy)
                    l_gy = int(l_shank_gy)
                    right_TA = int(right_TA)
                    left_TA = int(left_TA)
                    right_VL = int(right_VL)
                    left_VL = int(left_VL)
                    imu_list = [r_gy , l_gy]
                    r_lab = [r_gy]
                    l_lab = [l_gy]
                    r_lab_emg = [right_TA , right_VL]
                    l_lab_emg = [left_TA , left_VL]
                    emg_list= [right_TA , left_TA , right_VL, left_VL]
                    names = dataframes[0].columns
                    names_series = pd.Series(names)

                    n_emg_d = names_series[r_lab_emg]
                    n_emg_i = names_series[l_lab_emg]
                    n_emg_d = list(n_emg_d)
                    n_emg_i = list(n_emg_i)


                    featurename_emg = ['mav','ssi','var', 'rms','wfl','zc' ,'ssc','wa' ,
                   'mdf','mf' ,'she', 'spe', 'svde']

                    featurename_imu = ['min', 'max', 'mean', 'std', 'initial', 'final']


                    mode_lw = names[mode]
                    r_gy_ = names[r_gy]

                    l_gy_ = names[l_gy]
                    data = de.data_extraction_step(dataframes,names, emg_list, imu_list, mode_lw )
                    emg_filt = data[0]
                    acc_filt = data[1]
                    acc_names = data[2]
                    muscle_names = data[3]
                    index_steps = data[4]

                    lab_imu = dgs.create_labels_imu(acc_names, featurename_imu)
                    lab_emg = dgs.create_labels_emg(muscle_names, featurename_emg)
                    lab_emg_d = dgs.create_labels_emg(n_emg_d, featurename_emg)
                    lab_emg_i = dgs.create_labels_emg(n_emg_i, featurename_emg)
                    imu_features = dgs.acc_features_step(acc_filt,r_gy_,l_gy_, acc_names, index_steps, r_lab, l_lab)
                    emg_features = dgs.emg_features_step(emg_filt, muscle_names, index_steps,
                                      acc_filt[r_gy_],
                                     acc_filt[l_gy_], r_lab_emg, l_lab_emg)
                    dataframe_imu_step = dgs.create_dataframe_step(imu_features, lab_imu, 0,r_lab, l_lab, r_lab_emg, l_lab_emg) # num_var = [len(imu_list),len(emg_list)]
                    dataframe_emg_step = dgs.create_dataframe_step(emg_features, lab_emg, 1,r_lab, l_lab, lab_emg_d, lab_emg_i) #,num_var = [len(imu_list),len(emg_list)]
                    results = pd.concat([dataframe_imu_step,dataframe_emg_step],axis = 1)
                    r_shank = results.loc[:,results.columns.str.startswith(r_gy_)]
                    r_shank = r_shank.reindex(sorted(r_shank), axis = 1)

                    l_shank = results.loc[:,results.columns.str.startswith(l_gy_)]
                    l_shank = l_shank.reindex(sorted(l_shank), axis = 1)


                    rta = results.loc[:,results.columns.str.startswith(names[right_TA])]
                    rta = rta.reindex(sorted(rta), axis = 1)

                    lta = results.loc[:,results.columns.str.startswith(names[left_TA])]
                    lta = lta.reindex(sorted(lta), axis = 1)

                    rvl = results.loc[:,results.columns.str.startswith(names[right_VL])]
                    rvl = rvl.reindex(sorted(rvl), axis = 1)

                    lvl = results.loc[:,results.columns.str.startswith(names[left_VL])]
                    lvl = lvl.reindex(sorted(lvl), axis = 1)


                    pdList_r = [r_shank, rta, rvl]
                    pdList_l = [l_shank, lta, lvl]

                    results_r = pd.concat(pdList_r,axis = 1)
                    results_l = pd.concat(pdList_l,axis = 1)

                    results_r = results_r.dropna(axis=0,how='all')
                    results_l = results_l.dropna(axis=0,how='all')

                    results_r = results_r.values
                    results_l = results_l.values


                    prediction_right = []
                    prediction_left = []

                    length_r = np.arange(0,len(results_r), 1)
                    pred_r = aplicar_modelo(results_r, length_r)

                    length_l = np.arange(0,len(results_l), 1)
                    pred_l = aplicar_modelo(results_l, length_l)

                    predict_df_r = pd.DataFrame(np.concatenate(pred_r), columns = ['Derecho_GI_rms','Derecho_SOL_rms', 'Derecho_BF_rms', 'Derecho_ST_rms', 'Derecho_RF_rms'])

                    predict_df_l = pd.DataFrame(np.concatenate(pred_l), columns = ['Izquierdo_GI_rms','Izquierdo_SOL_rms', 'Izquierdo_BF_rms', 'Izquierdo_ST_rms', 'Izquierdo_RF_rms'])

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
