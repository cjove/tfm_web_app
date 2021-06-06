import streamlit as st
from PIL import Image

def app():
    st.title('Información adicional')
    st.markdown('Toda la información referente al procedimiento utilizado en la extracción de la señal y el modo en el que han sido entrenados los modelos puedes encontrarlo en este [repositorio](https://github.com/cjove/TFM_CarlosJove_UOC)')
    st.markdown('Por favor, antes de empezar a utilizar la aplicación, comprueba en la siguiente imagen la orientación de tus sensores inerciales respecto a la utilizada en este trabajo.')
    image = Image.open("Imagen.jpg")

    st.image(image = image , caption = "Referencias espaciales de los sensores inerciales")


    st.markdown('Cualquier tipo de feedback o corrección de errores es bienvenida en el correo: cjoveb@gmail.com')
