import streamlit as st
from PyPDF2 import PdfReader
from transformers import pipeline

# Cargar el modelo GPT-2 (o GPT-Neo si prefieres)
modelo = pipeline("text-generation", model="gpt2")  # Cambia a "EleutherAI/gpt-neo-125M" si prefieres GPT-Neo

def consultar_modelo_local(texto, pregunta):
    # Construye el prompt
    prompt = f"""
    Texto: {texto}

    Pregunta: {pregunta}

    Instrucción: Responde la pregunta usando solo la información proporcionada en el texto. Si no encuentras la respuesta en el texto, di "No encontré información relevante en el documento."
    """
    
    # Genera la respuesta usando el modelo local
    respuesta = modelo(
        prompt,
        max_length=150,  # Limita la longitud de la respuesta
        num_return_sequences=1,  # Devuelve solo una respuesta
        temperature=0.7,  # Controla la creatividad (valores bajos = más precisos)
        top_p=0.9,  # Filtra las respuestas menos probables
        do_sample=True  # Muestra una respuesta aleatoria pero controlada
    )
    return respuesta[0]['generated_text']

st.title("Chat con PDFs usando DeepSeek-R1")
st.write("¡Bienvenido! Sube un archivo PDF para comenzar.")

# Subir archivo PDF
uploaded_file = st.file_uploader("Sube un archivo PDF", type="pdf")

if uploaded_file is not None:
    st.write("Archivo PDF subido correctamente.")
    
    # Extraer texto del PDF
    pdf_reader = PdfReader(uploaded_file)
    text = ""
    for page in pdf_reader.pages:
        text += page.extract_text()
    
    st.write("Texto extraído del PDF:")
    st.write(text)

    # Lógica para chatear con el PDF
    st.write("### Chatea con el PDF")
    user_input = st.text_input("Escribe tu pregunta:")
    
    if user_input:
        respuesta = consultar_modelo_local(text, user_input)
        st.write(f"**Respuesta:** {respuesta}")
        from transformers import pipeline

# Cargar el modelo GPT-2
modelo = pipeline("text-generation", model="gpt2")
from transformers import pipeline


