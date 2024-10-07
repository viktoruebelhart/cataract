import streamlit as st
import gdown
import tensorflow as tf
import io 
from PIL import Image 
import numpy as np
import pandas as pd

@st.cache_resource
def load_model():
    # Link do modelo TFLite de catarata
    url = 'https://drive.google.com/uc?id=1NSsQconZZViIPqI5Z-2tWqK1AVXoN-0h'

    gdown.download(url, 'cataract_model.tflite', quiet=False)
    interpreter = tf.lite.Interpreter(model_path='cataract_model.tflite')
    interpreter.allocate_tensors()
    return interpreter
    

def load_image():
    uploaded_file = st.file_uploader('Faça upload da imagem do olho para análise', type=['png', 'jpeg', 'jpg'])

    if uploaded_file is not None:
        image_data = uploaded_file.read()
        image = Image.open(io.BytesIO(image_data))

        st.image(image, caption='Imagem carregada com sucesso', use_column_width=True)

        # Redimensionar a imagem para 416x416 pixels (dimensão esperada)
        image = image.resize((416, 416))
        image = np.array(image, dtype=np.float32)
        image = image / 255.0  # Normalizar a imagem
        image = np.expand_dims(image, axis=0)  # Adicionar a dimensão de batch

        # Exibir informações da imagem
        st.write(f"Dimensões da imagem processada: {image.shape}")
        st.write(f"Tipo de dado da imagem: {image.dtype}")

        return image

def forecast(interpreter, image):
    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()

    # Exibir as formas de entrada esperadas pelo modelo
    st.write(f"Forma esperada de entrada pelo modelo: {input_details[0]['shape']}")
    st.write(f"Tipo de entrada esperado pelo modelo: {input_details[0]['dtype']}")

    # Passar a imagem processada para o modelo
    interpreter.set_tensor(input_details[0]['index'], image)
    interpreter.invoke()

    output_data = interpreter.get_tensor(output_details[0]['index'])

    # A saída do modelo é uma probabilidade
    probability = output_data[0][0]

    # Definir classes com base na probabilidade
    if probability < 0.5:
        classification = 'Catarata Imatura'
    else:
        classification = 'Catarata Madura'

    # Exibir o resultado da classificação
    st.write(f"### Classificação: {classification}")
    st.write(f"### Probabilidade: {probability:.2%}")