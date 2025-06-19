
# Consulta boletín ODEPA — OpenAI + Streamlit + RAG

Este proyecto permite consultar boletines PDF de ODEPA usando un chatbot RAG (Retrieval-Augmented Generation) con OpenAI (GPT 3.5 turbo).

### Instrucciones para ejecutar en Streamlit Cloud

1. Crear repositorio en GitHub (ya creado por @gescaffs)
2. Subir este proyecto
3. En https://streamlit.io/cloud conectar el repo
4. En Secrets de Streamlit Cloud agregar:

```
OPENAI_API_KEY = "tu_api_key"
```

5. Los documentos PDF deben estar en la carpeta `/documentos/`
6. Deploy — ¡y listo!

### Requisitos

- Streamlit
- langchain
- openai
- python-dotenv
- faiss-cpu

