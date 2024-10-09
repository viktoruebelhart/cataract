import streamlit as st
import gdown
import tensorflow as tf
import io
from PIL import Image
import numpy as np

# Função para carregar o modelo .h5
@st.cache_resource
def load_model():
    # Baixando o modelo do Google Drive
    #https://drive.google.com/file/d/1hrx7hSRezwkXr34_Feq4ZLdvhlkS_R3n/view?usp=drive_link
    url = 'https://drive.google.com/uc?id=1hrx7hSRezwkXr34_Feq4ZLdvhlkS_R3n'
    gdown.download(url, 'cataract_model.h5', quiet=False)
    
    # Carregando o modelo .h5
    model = tf.keras.models.load_model('cataract_model.h5')
    return model

# Função para carregar e processar a imagem
def load_image():
    uploaded_file = st.file_uploader('Carregue a imagem do olho para análise', type=['png', 'jpeg', 'jpg'])
    
    if uploaded_file is not None:
        image = Image.open(io.BytesIO(uploaded_file.read()))
        st.image(image, caption='Imagem carregada com sucesso', use_column_width=True)
        
        # Redimensionar a imagem para o formato correto (ex: 416x416)
        image = image.resize((416, 416))
        image = np.array(image, dtype=np.float32) / 255.0  # Normalizando a imagem
        image = np.expand_dims(image, axis=0)  # Adicionando a dimensão de batch
        
        return image
    else:
        st.warning("Por favor, faça o upload de uma imagem para análise.")
        return None

# Função para fazer a previsão usando o modelo Keras
def forecast(model, image):
    # Fazer a previsão
    prediction = model.predict(image)
    probability = prediction[0][0]

    # Classificação com base na probabilidade
    if probability < 0.5:
        st.success("### Classificação: Catarata Imatura")
    else:
        st.success("### Classificação: Catarata Madura")
    
    st.write(f"### Probabilidade: {probability:.2%}")

# Função principal da aplicação
def main():
    st.set_page_config(page_title="Diagnóstico de Catarata")
    st.write("# Diagnóstico de Catarata")
    st.write("Faça o upload de uma imagem para classificar o estágio da catarata.")
    
    # Carregar o modelo Keras (.h5)
    with st.spinner('Carregando o modelo...'):
        model = load_model()
    
    # Carregar a imagem do usuário
    image = load_image()
    
    # Se a imagem foi carregada, fazer a previsão
    if image is not None:
        forecast(model, image)

# Rodar a aplicação
if __name__ == "__main__":
    main()
