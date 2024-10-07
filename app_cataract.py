import streamlit as st
import gdown
import tensorflow as tf
import io
from PIL import Image
import numpy as np

# Função para carregar o modelo TFLite
@st.cache_resource
def load_model():
    # URL do modelo catarata em TFLite
    url = 'https://drive.google.com/uc?id=1NSsQconZZViIPqI5Z-2tWqK1AVXoN-0h'
    gdown.download(url, 'cataract_model.tflite', quiet=False)
    interpreter = tf.lite.Interpreter(model_path='cataract_model.tflite')
    interpreter.allocate_tensors()
    return interpreter

# Função para carregar e processar a imagem
def load_image():
    uploaded_file = st.file_uploader('Carregue a imagem do olho para análise', type=['png', 'jpeg', 'jpg'])
    
    if uploaded_file is not None:
        image = Image.open(io.BytesIO(uploaded_file.read()))
        st.image(image, caption='Imagem carregada com sucesso', use_column_width=True)
        
        # Redimensionar a imagem para o formato 416x416x3
        image = image.resize((416, 416))
        image = np.array(image, dtype=np.float32) / 255.0  # Normalização
        image = np.expand_dims(image, axis=0)  # Expansão para incluir a dimensão de batch

        return image
    else:
        return None

# Função para realizar a previsão
def forecast(interpreter, image):
    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()

    interpreter.set_tensor(input_details[0]['index'], image)
    interpreter.invoke()
    
    output_data = interpreter.get_tensor(output_details[0]['index'])
    probability = output_data[0][0]  # Como a saída é escalar, interpretamos
    
    # Classificação com base na probabilidade
    if probability < 0.5:
        st.write("### Classificação: Catarata Imatura")
    else:
        st.write("### Classificação: Catarata Madura")
    
    st.write(f"### Probabilidade: {probability:.2%}")

# Função principal da aplicação
def main():
    st.set_page_config(page_title="Diagnóstico de Catarata")
    st.write("# Diagnóstico de Catarata")
    st.write("Faça o upload de uma imagem para classificar o estágio da catarata.")
    
    # Carregar o modelo TFLite
    interpreter = load_model()
    
    # Carregar a imagem do usuário
    image = load_image()
    
    # Se a imagem foi carregada, fazer a previsão
    if image is not None:
        forecast(interpreter, image)

if __name__ == "__main__":
    main()
