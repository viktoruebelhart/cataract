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

    gdown.download(url, 'cataract_model.tflite')
    interpreter = tf.lite.Interpreter(model_path='cataract_model.tflite')
    interpreter.allocate_tensors()
    return interpreter
    

def load_image():
    uploaded_file = st.file_uploader('Faça upload da imagem do olho para análise', type=['png', 'jpeg', 'jpg'])

    if uploaded_file is not None:
        image_data = uploaded_file.read()
        image = Image.open(io.BytesIO(image_data))

        st.image(image, caption='Imagem carregada com sucesso', use_column_width=True)
        image = image.resize((224, 224))  # Exemplo de redimensionamento
        image = image.convert('RGB')
        image = np.array(image, dtype=np.float32)
        image = image / 255.0  # Normalizar a imagem
        image = np.expand_dims(image, axis=0)  # Adicionar a dimensão de batch

        return image
    
def forecast(interpreter, image):
    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()

    interpreter.set_tensor(input_details[0]['index'], image)

    interpreter.invoke()

    output_data = interpreter.get_tensor(output_details[0]['index'])

    # Como a saída é escalar, interpretamos:
    probability = output_data[0][0]

    # Definir classes
    if probability < 0.5:
        classification = 'Immature Cataract'
    else:
        classification = 'Mature Cataract'

    # Exibir o resultado
    st.write(f"### Classificação: {classification}")
    st.write(f"### Probabilidade: {probability:.2%}")

def main():
    st.set_page_config(
        page_title="Diagnóstico de Catarata"
    )
    st.write("# Diagnóstico de Catarata!")
    st.write("Carregue uma imagem do olho para classificar o estágio da catarata.")

    # Carregar o modelo
    interpreter = load_model()

    # Carregar a imagem
    image = load_image()

    # Fazer a previsão se a imagem foi carregada
    if image is not None:
        forecast(interpreter, image)

if __name__ == "__main__":
    main()