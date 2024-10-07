import streamlit as st
import gdown
import tensorflow as tf
import io 
from PIL import Image 
import numpy as np
import pandas as pd
import plotly.express as px

@st.cache_resource
def load_model():
    # Link do model
    # https://drive.google.com/file/d/1MC4ZT730DUtMQ4MUJ7Rj8dt3d5rpCImp/view?usp=drive_link
    url = 'https://drive.google.com/uc?id=1MC4ZT730DUtMQ4MUJ7Rj8dt3d5rpCImp'

    gdown.download(url, 'cataract_model.tflite')
    interpreter = tf.lite.Interpreter(model_path='cataract_model.tflite')
    interpreter.allocate_tensors()
    return interpreter
    

def load_image():
    uploaded_file = st.file_uploader('Upload eye image for analysis', type=['png', 'jpeg', 'jpg'])

    if uploaded_file is not None:
        image_data = uploaded_file.read()
        image = Image.open(io.BytesIO(image_data))

        st.image(image, caption='Image uploaded successfully', use_column_width=True)

        image = np.array(image, dtype=np.float32)
        image = image / 255.0  # Normalize image
        image = np.expand_dims(image, axis=0)  # add dimension batch

        return image
    
def forecast(interpreter, image):
    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()

    interpreter.set_tensor(input_details[0]['index'], image)

    interpreter.invoke()

    output_data = interpreter.get_tensor(output_details[0]['index'])

    # Classes de catarata: Immature ou Mature
    classes = ['Immature Cataract', 'Mature Cataract']

    # Criar dataframe para visualização das probabilidades
    df = pd.DataFrame()
    df['classes'] = classes
    df['probability (%)'] = 100 * output_data[0]

    # Exibir gráfico de barras com as probabilidades
    fig = px.bar(df, y='classes', x='probability (%)', orientation='h', text='probability (%)',
                 title='Probabilidade de catarata')
    st.plotly_chart(fig)

def main():
    st.set_page_config(
        page_title="Cataract Diagnosis"
    )
    st.write("# Cataract Diagnosis !")
    st.write("Upload an image of the eye to classify the cataract stage.")

    # Carregar o modelo
    interpreter = load_model()

    # Carregar a imagem
    image = load_image()

    # Fazer a previsão se a imagem foi carregada
    if image is not None:
        forecast(interpreter, image)

if __name__ == "__main__":
    main()
