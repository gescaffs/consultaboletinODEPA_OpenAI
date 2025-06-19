import streamlit as st
import os
import glob
import openai
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores import FAISS
from langchain.document_loaders import PyPDFLoader
from langchain.chains import RetrievalQA
from dotenv import load_dotenv

# Cargar API Key
load_dotenv()
openai.api_key = os.getenv("OPENAI_API_KEY")

# CSS para fondo
st.markdown(
    """
    <style>
    .stApp {
        background: url('img/arbol_fruta.png');
        background-size: cover;
        background-position: center;
    }
    .stTextInput textarea, .stButton>button {
        font-size: 1.1rem;
    }
    </style>
    """, unsafe_allow_html=True)

# Título
st.title("🌳 Asistente para Productores Agrícolas (RAG)")
st.subheader("Consulta boletines ODEPA de forma sencilla")
st.divider()

# Leer TODOS los PDFs en la carpeta /documentos/
pdf_files = glob.glob("documentos/*.pdf")

if not pdf_files:
    st.error("No se encontraron archivos PDF en la carpeta 'documentos/'. Agrega al menos uno.")
    st.stop()

# Cargar y combinar todas las páginas de todos los PDFs
all_pages = []

for pdf_file in pdf_files:
    loader = PyPDFLoader(pdf_file)
    pages = loader.load_and_split()
    all_pages.extend(pages)

# Embeddings
embeddings = OpenAIEmbeddings()
text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
texts = text_splitter.split_documents(all_pages)
db = FAISS.from_documents(texts, embeddings)
retriever = db.as_retriever(search_kwargs={"k": 3})

# LLM económico
llm = ChatOpenAI(model="gpt-3.5-turbo-0125", max_tokens=500)
qa_chain = RetrievalQA.from_chain_type(llm=llm, retriever=retriever, return_source_documents=True)

# Interfaz
user_question = st.text_area("📝 Haz tu pregunta:", height=120)
if st.button("Enviar") and user_question.strip():
    with st.spinner("Buscando respuesta..."):
        result = qa_chain.invoke({"query": user_question})
    st.divider()
    if result['result'].strip():
        st.success(result['result'])
    else:
        st.warning("ℹ️ Información no disponible por el momento.")
    st.divider()
    st.subheader("📁 Documentos fuente")
    with st.expander("Ver detalles"):
        for doc in result["source_documents"]:
            st.write(f"- {doc.metadata.get('source', 'Desconocido')}")
