# rag.py (versión actualizada con imports nuevos)
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.embeddings import OllamaEmbeddings
from langchain_community.vectorstores import Chroma

def cargar_documento(pdf_path):
    loader = PyPDFLoader(pdf_path)
    documentos = loader.load()
    return documentos

def dividir_documentos(docs, chunk_size=1000, chunk_overlap=200):
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap
    )
    return splitter.split_documents(docs)

def crear_vectores(chunks, persist_dir="chromadb", modelo_embedding="nomic-embed-text"):
    embedding = OllamaEmbeddings(model=modelo_embedding)
    vectordb = Chroma.from_documents(
        documents=chunks,
        embedding=embedding,
        persist_directory=persist_dir
    )
    vectordb.persist()
    return vectordb

def cargar_vectores(persist_dir="chromadb", modelo_embedding="nomic-embed-text"):
    embedding = OllamaEmbeddings(model=modelo_embedding)
    return Chroma(
        persist_directory=persist_dir,
        embedding_function=embedding
    )

if __name__ == "__main__":
    import os

    folder_path = "data"
    if not os.path.exists(folder_path):
        print("❌ No se encontró la carpeta 'data'")
        exit()

    pdf_files = [f for f in os.listdir(folder_path) if f.endswith(".pdf")]

    if not pdf_files:
        print("📭 No se encontraron archivos PDF en la carpeta 'data'")
        exit()

    print(f"📁 Se encontraron {len(pdf_files)} archivo(s) PDF para procesar:\n")
    for f in pdf_files:
        print(f"  📄 {f}")

    all_chunks = []

    for pdf_file in pdf_files:
        print(f"\n📂 Procesando: {pdf_file}")
        pdf_path = os.path.join(folder_path, pdf_file)

        try:
            docs = cargar_documento(pdf_path)
            print(f"✅ Documento cargado ({len(docs)} página(s))")

            chunks = dividir_documentos(docs)
            print(f"✅ Se generaron {len(chunks)} chunks")

            all_chunks.extend(chunks)

        except Exception as e:
            print(f"⚠️ Error procesando {pdf_file}: {e}")

    print("\n💾 Guardando todos los chunks en ChromaDB...")
    crear_vectores(all_chunks)
    print("✅ Base vectorial unificada creada y guardada en 'chromadb'")