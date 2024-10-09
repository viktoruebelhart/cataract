import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image
import io

# Função para carregar o modelo
@st.cache_resource
def load_model():
    model = tf.keras.models.load_model('cataract_model.h5')
    return model

# Função para carregar e pré-processar a imagem
def load_and_preprocess_image(uploaded_file):
    image = Image.open(uploaded_file)
    image = image.resize((416, 416))  # Redimensione para 416x416
    image = np.array(image, dtype=np.float32) / 255.0  # Normalização
    image = np.expand_dims(image, axis=0)  # Expansão para incluir a dimensão de batch
    return image

# Função para prever a classe da imagem
def predict_class(model, image):
    prediction = model.predict(image)
    return prediction[0][0]

# Função principal do Streamlit
def main():
    st.set_page_config(page_title="Diagnóstico de Catarata")
    st.title("Diagnóstico de Catarata")
    st.write("Faça o upload de uma imagem para classificar o estágio da catarata.")

    # Carregar o modelo
    model = load_model()

    # Carregar a imagem do usuário
    uploaded_file = st.file_uploader("Carregue a imagem do olho para análise", type=['png', 'jpeg', 'jpg'])
    
    if uploaded_file is not None:
        image = load_and_preprocess_image(uploaded_file)
        st.image(Image.open(uploaded_file), caption='Imagem carregada com sucesso', use_column_width=True)

        # Fazer a previsão
        probability = predict_class(model, image)

        # Exibir a classificação
        if probability < 0.5:
            st.write("### Classificação: Catarata Imatura")
        else:
            st.write("### Classificação: Catarata Madura")
        
        st.write(f"### Probabilidade: {probability:.2%}")

if __name__ == "__main__":
    main()