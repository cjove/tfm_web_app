import streamlit as st
from multiapp import MultiApp
import app_lw, app_s2s_2, home, model_s2s, model_step

import pickle

app = MultiApp()

st.markdown("""
# Trabajo final de Máster - Carlos Jové Blanco
Esta aplicación web es el resultado del trabajo de final de Máster de la titulación "Máster en Bioestadísctica y Bioinformática" UOC-UB.
En ella podrás introducir tus datos recogidos durante la marcha en superficie plana o la transición de sedestación a bipedestación, escoger las variables que sean de tu interés y obtendrás el valor de la dichas variables de cada ciclo de la marcha o de cada transición.
También podrás introducir tus datos para predecir la actividad electrica de aquellos músculos que no has podido registrar.
Accede a cada uno de los apartados de la aplicación para obtener información más detallada.

""")

# Add all your application here
app.add_app("Home", home.app)
app.add_app("Extracción de variables durante la transición de sedestación a bipedestación", app_s2s_2.app)
app.add_app("Predicción de la actividad neuromuscular durante la transición de sedestación a bipedestación ", model_s2s.app)
app.add_app("Extracción de variables durante la marcha en superficie plana", app_lw.app)
app.add_app("Predicción de la actividad neuromuscular durante la marcha en superficie plana ", model_step.app)

# The main app
app.run()
