# app.py
import os
import time
import streamlit as st
from dotenv import load_dotenv

from rag import cargar_vectores, cargar_documento, dividir_documentos

from langchain.chat_models import ChatOllama
from langchain.prompts import PromptTemplate

# =============================
# CONFIGURACIÓN INICIAL
# =============================
load_dotenv()
st.set_page_config(page_title="Asistente Legal RAG", page_icon="⚖️")
st.title("⚖️ Asistente Legal con RAG")
st.write("Consulta la Ley 599 y 906 de Colombia, o sube un documento legal adicional para incluirlo en tu consulta.")

# =============================
# SELECCIÓN DE MODELOS
# =============================
modelos_disponibles = {
    "Qwen 3 (4B)": "qwen3:4b",
    "Gemma 3 (4B)": "gemma3:4b",
    "Phi": "phi:latest",
    "LLaMA 3.2": "llama3.2:latest",
    "LLaMA 3": "llama3:latest"
}

modelos_seleccionados = st.multiselect("🧠 Selecciona hasta 2 modelos", list(modelos_disponibles.keys()), max_selections=2)

# =============================
# CARGAR PDF OPCIONAL
# =============================
uploaded_file = st.file_uploader("📄 (Opcional) Sube otro documento legal en PDF", type=["pdf"])

chunks_pdf_usuario = []
if uploaded_file:
    st.info(f"Procesando archivo adicional: {uploaded_file.name}")
    temp_path = os.path.join("data", uploaded_file.name)
    with open(temp_path, "wb") as f:
        f.write(uploaded_file.read())

    with st.spinner("📚 Leyendo y dividiendo el documento..."):
        docs = cargar_documento(temp_path)
        chunks_pdf_usuario = dividir_documentos(docs)

# =============================
# PREGUNTA DEL USUARIO
# =============================
pregunta = st.text_input("💬 Escribe tu pregunta sobre las leyes 599, 906 o el documento que subiste")

# =============================
# GENERADOR DE RESPUESTA STREAMING
# =============================
def responder_en_stream(model_id, contexto, pregunta):
    modelo = ChatOllama(model=model_id, temperature=0.2, top_p=0.9, max_tokens=500, streaming=True)

    prompt_template = PromptTemplate(
        input_variables=["context", "question"],
        template="""
Eres un asistente legal experto en el sistema jurídico colombiano, especializado en derecho penal, de tránsito y de policía.
Usa exclusivamente la siguiente información extraída de documentos legales para responder la consulta de forma clara y profesional.

Contexto legal:
{context}

Pregunta: {question}
Respuesta:"""
    )
    
    full_prompt = prompt_template.format(context="\n".join([doc.page_content for doc in contexto]), question=pregunta)
    
    response_generator = modelo.stream(full_prompt)
    return response_generator

# =============================
# RESPUESTA EN PANTALLA
# =============================
if pregunta and len(modelos_seleccionados) > 0:
    with st.spinner("🔎 Buscando respuestas..."):
        vectordb = cargar_vectores()
        retriever = vectordb.as_retriever(search_kwargs={"k": 6})
        documentos_legales = retriever.get_relevant_documents(pregunta)

        contexto_base = documentos_legales + chunks_pdf_usuario
        contexto_filtrado = [doc for doc in contexto_base if len(doc.page_content.strip()) > 100]

    if len(modelos_seleccionados) == 1:
        modelo_clave = modelos_seleccionados[0]
        modelo_id = modelos_disponibles[modelo_clave]

        st.markdown(f"#### 🤖 {modelo_clave}")
        respuesta_container = st.empty()
        texto_respuesta = ""

        start_time = time.time()
        for chunk in responder_en_stream(modelo_id, contexto_filtrado, pregunta):
            texto_respuesta += chunk.content
            respuesta_container.markdown(texto_respuesta)
        end_time = time.time()

        st.caption(f"🕒 Tiempo: {end_time - start_time:.2f} segundos")

    elif len(modelos_seleccionados) == 2:
        col1, col2 = st.columns(2)
        for i, modelo_clave in enumerate(modelos_seleccionados):
            modelo_id = modelos_disponibles[modelo_clave]
            columna = col1 if i == 0 else col2

            with columna:
                st.markdown(f"#### 🤖 {modelo_clave}")
                respuesta_container = st.empty()
                texto_respuesta = ""

                start_time = time.time()
                for chunk in responder_en_stream(modelo_id, contexto_filtrado, pregunta):
                    texto_respuesta += chunk.content
                    respuesta_container.markdown(texto_respuesta)
                end_time = time.time()

                st.caption(f"🕒 Tiempo: {end_time - start_time:.2f} segundos")