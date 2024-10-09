import streamlit as st
import gdown
import tensorflow as tf
import io
from PIL import Image
import numpy as np

@st.cache_resource
def load_model():
    # URL do Google Drive
    url = 'https://drive.google.com/uc?id=1hrx7hSRezwkXr34_Feq4ZLdvhlkS_R3n'
    gdown.download(url, 'cataract_model.h5', quiet=False)
    model = tf.keras.models.load_model('cataract_model.h5')
    return model

def load_image():
    uploaded_file = st.file_uploader('Carregue a imagem do olho para análise', type=['png', 'jpeg', 'jpg'])
    
    if uploaded_file is not None:
        image = Image.open(io.BytesIO(uploaded_file.read()))
        st.image(image, caption='Imagem carregada com sucesso', use_column_width=True)
        
        # Redimensionar a imagem para o formato 416x416
        image = image.resize((416, 416))
        image = np.array(image, dtype=np.float32) / 255.0  # Normalização
        image = np.expand_dims(image, axis=0)  # Expansão para incluir a dimensão de batch
        
        return image
    else:
        return None

def forecast(model, image):
    # Fazer previsão com o modelo carregado
    prediction = model.predict(image)
    probability = prediction[0][0]

    # Classificação com base na probabilidade
    if probability < 0.5:
        st.write("### Classificação: Catarata Imatura")
    else:
        st.write("### Classificação: Catarata Madura")
    
    st.write(f"### Probabilidade: {probability:.2%}")

def main():
    st.set_page_config(page_title="Diagnóstico de Catarata")
    st.write("# Diagnóstico de Catarata")
    st.write("Faça o upload de uma imagem para classificar o estágio da catarata.")
    
    # Carregar o modelo
    model = load_model()
    
    # Carregar a imagem do usuário
    image = load_image()
    
    # Se a imagem foi carregada, fazer a previsão
    if image is not None:
        forecast(model, image)

if __name__ == "__main__":
    main()
