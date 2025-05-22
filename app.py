# app.py
import os
import streamlit as st
from dotenv import load_dotenv

from rag import cargar_vectores, cargar_documento, dividir_documentos

from langchain.chat_models import ChatOllama
from langchain.prompts import PromptTemplate
from langchain.chains.question_answering import load_qa_chain

# =============================
# CONFIGURACIÓN INICIAL
# =============================
load_dotenv()
st.set_page_config(page_title="Asistente Legal RAG", page_icon="⚖️")
st.title("⚖️ Asistente Legal con RAG")
st.write("Consulta la Ley 599 y 906 de Colombia, o sube un documento legal adicional para incluirlo en tu consulta.")

# =============================
# SELECCIÓN DE MODELO
# =============================
modelos_disponibles = {
    "Qwen 3 (4B)": "qwen3:4b",
    "Gemma 3 (4B)": "gemma3:4b",
    "Phi": "phi:latest",
    "LLaMA 3.2": "llama3.2:latest",
    "LLaMA 3": "llama3:latest"
}

modelo_seleccionado = st.selectbox("🧠 Selecciona el modelo de lenguaje", list(modelos_disponibles.keys()))
modelo_nombre = modelos_disponibles[modelo_seleccionado]

# =============================
# CARGAR PDF OPCIONAL DEL USUARIO
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

if pregunta:
    with st.spinner("🔎 Buscando respuesta..."):
        vectordb = cargar_vectores()
        retriever = vectordb.as_retriever(search_kwargs={"k": 6})
        documentos_legales = retriever.get_relevant_documents(pregunta)

        if chunks_pdf_usuario:
            documentos_pdf = chunks_pdf_usuario
        else:
            documentos_pdf = []

        contexto_completo = documentos_legales + documentos_pdf
        contexto_filtrado = [doc for doc in contexto_completo if len(doc.page_content.strip()) > 100]

        modelo = ChatOllama(
            model=modelo_nombre,
            temperature=0.2,
            top_p=0.9,
            max_tokens=500
        )

        prompt = PromptTemplate(
            input_variables=["context", "question"],
            template="""
Eres un asistente legal experto en el sistema jurídico colombiano, especializado en derecho penal, de tránsito y de policía.
Usa exclusivamente la siguiente información extraída de documentos legales para responder la consulta de forma clara y profesional.

Contexto legal:
{context}

Pregunta: {question}
Respuesta:"""
        )

        chain = load_qa_chain(modelo, chain_type="stuff", prompt=prompt)
        respuesta = chain.run(input_documents=contexto_filtrado, question=pregunta)

    st.markdown("### 📌 Respuesta:")
    st.write(respuesta)