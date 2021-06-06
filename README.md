# tfm_web_app
En este repositorio se encuentra todo el código desarrollado para la implementación a través de Streamlit resultante del trabajo final del Master de Bioestadistica y Bioinformatica UOC-UB realizado por Carlos Jove.

Se puede acceder a la aplicación en el siguiente enlace: https://share.streamlit.io/cjove/tfm_web_app/main/app.py. 

Contenido del repositorio de la aplicación web.

•	Imagen.jpg: contiene la imagen que se muestra en la página de inicio de la aplicación web.

•	app.py: archivo necesario para la creación de una aplicación web con aplicaciones internas.

•	multiapp.py: archivo necesario para la creación de una aplicación web con aplicaciones internas.

•	home.py: contiene la información que se presenta en la página de inicio de la aplicación web.

•	app_lw.py: contiene el código de la ventana “Extracción de variables durante la marcha en superficie plana”. 

•	app_s2s_2.py: contiene el código de la ventana “Extracción de variables durante la transición de sedestación a bipedestación”.

•	model_s2s.py: contiene el código de la ventana “Predicción de la actividad neuromuscular durante la transición de sedestación a bipedestación”.

•	model_step.py: contiene el código de la ventana “Predicción de la actividad neuromuscular durante la marcha en superficie plana”.

•	multioutput_regression_step_def.pkl: archivo que contiene el modelo entrenado para la predicción de la actividad muscular durante la marcha en superficie plana.

•	multioutput_regression_s2s_def.pkl: archivo que contiene el modelo entrenado para la predicción de la actividad muscular durante la transición de sedestación a bipedestación.

•	requirements.txt: contiene los requerimientos que necesita la aplicación web para funcionar.

•	data_extraction_web.py: contiene el código necesario para extraer la información de la matriz de datos facilitada por el usuario.

•	dataset_generation_s2s_web.py: contiene el código necesario para generar la matriz de datos de las variables extraídas de las señales introducidas por el usuario de la transición de sedestación a bipedestación.

•	dataset_generation_step_web.py: contiene el código necesario para generar la matriz de datos de las variables extraídas de las señales introducidas por el usuario de la marcha en superficie plana.

•	signal_processing.py: contiene el código necesario para realizar el filtrado de las señales introducidas por el usuario.

•	sit2stand_functions_web.py: contiene las funciones necesarias para la detección de la transición de sedestación a bipedestación de los datos introducidos por el usuario.

•	step_functions_web.py: contiene las funciones necesarias para la detección de los ciclos de la marcha en superficie plana de los datos introducidos por el usuario.

•	variable_extraction_s2s_web.py: contiene el código para realizar la extracción de las variables de la transición de sedestación a bipedestación de los datos introducidos por el usuario.

•	variable_extraction_step_web.py: contiene el código para realizar la extracción de las variables de la marcha en superficie plana de los datos introducidos por el usuario.
