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
    pickle_in = open('multioutput_regression_step.pkl', 'rb')
    regressor = pickle.load(pickle_in)

    header = st.beta_container()
    dataset = st.beta_container()
    #form = st.form(key='my_form2')
    #processed = st.beta_container()
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
        st.markdown('En esta sección de la aplicación, introduciendo en el formulario las columnas provenientes de: X,Y,Z, podrás predecir la actividad muscular de los siguientes 5 músculos: 1,2,3,4,5')
        st.markdown('El algoritmo utilizado es:           con los siguientes parámetros:    ')
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

    with model_application:
        st.header('Paso 2. Indica a continuación qué columna pertenece a cada uno de los siguientes apartados')
        st.markdown('Recuerda: este modelo requiere que hayas recogido información de sensores inerciales en las tibias y .....')
        r_shank_gy = st.text_input('¿Qué columna corresponde con el componente Y del giróscopo en la tibia derecha?', '3')
        l_shank_gy = st.text_input('¿Qué columna corresponde con el componente Y del giróscopo en la tibia izquierda?', '15')
        r_shank_rest = st.text_input('¿Qué columnas corresponden con el resto de componentes del sensor inercial en la tibia derecha?', '0,1,2,4,5')
        l_shank_rest = st.text_input('¿Qué columnas corresponden con el resto de componentes del sensor inercial en la tibia izquierda?', '12,13,14,16,17')
        right_MG = st.text_input('¿Qué columna corresponde con el gemelo medial de la pierna derecha?','31')
        left_MG = st.text_input('¿Qué columna corresponde con el gemelo medial de la pierna izquierda?', '38')
        right_SOL = st.text_input('¿Qué columna corresponde con el sóleo de la pierna derecha?', '32')
        left_SOL = st.text_input('¿Qué columna corresponde con el soleo de la pierna izquierda?', '39')
        mode = st.text_input('¿Que columna corresponde al modo?', '48')
        submit_button_model = st.form_submit_button(label='Submit')

        if submit_button_model:
            if len(dataframes) != 0 :
                try:
                    #list_emg_time = {'wa':wa,'var':var,'rms':rms,'mav':mav,'wfl':wfl,'zc':zc,'ssc':ssc, 'ssi':ssi }
                    #list_emg_freq = {'mdf':mdf,'mf':mnf,'she':se,'spe':spe,'svde':svde}
                    #list_imu = {'min':min_imu, 'max': max_imu,'std':std_imu, 'fin':final_imu, 'ini':init_imu,'mean': mean_imu}

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

                    st.markdown('La predicción de la actividad muscular del lado derecho')
                    st.write(predict_df_r)

                    st.markdown('La predicción de la actividad muscular del lado izquierdo')
                    st.write(predict_df_l)

                    prediction = pd.concat([predict_df_r, predict_df_l], axis = 1)
                #r= results.filter(regex= selected).columns
                    #st.write(results)
                    if prediction is not None:
                            st.header('¡Aquí tienes tus datos procesados! :)')
                            #st.markdown('This are the option you selected:' + str(submit_button))
                            download = download_link(prediction, 'datos_prediccion.csv', 'Pulsa aquí para descargar los datos')

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
