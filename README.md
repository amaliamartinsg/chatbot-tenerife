# Generador automático de infraestructura Azure con IA
# Chatbot de Turismo de Tenerife

Este proyecto implementa un asistente conversacional basado en inteligencia artificial para responder preguntas sobre turismo en Tenerife. Utiliza LangChain, OpenAI y Pinecone para procesar documentos, generar embeddings y realizar búsquedas semánticas, permitiendo respuestas precisas basadas únicamente en la información de los documentos cargados.

## Características principales
- **Entrenamiento personalizado:** Permite subir documentos en formatos `.pdf`, `.txt`, `.md`, `.docx`, `.html`, `.tf` para alimentar el modelo.
- **Chatbot contextual:** Responde preguntas sobre turismo de Tenerife usando solo la información de los documentos subidos.
- **Fuentes de respuesta:** Muestra los fragmentos y archivos utilizados para generar cada respuesta.
- **Interfaz web:** Desarrollada con Streamlit, con pestañas para entrenamiento y chat.
- **Integración con Pinecone:** Almacena y recupera embeddings para búsquedas semánticas eficientes.

## Estructura del proyecto
```
app.py                  # Interfaz principal Streamlit
aux_files/_utils.py     # Funciones de procesamiento, embeddings y consulta LLM
requirements.txt        # Dependencias del proyecto
README.md               # Este archivo
aux_files/              # Archivos auxiliares
	_utils.py
	...
data/                   # Documentos de ejemplo
	TENERIFE.pdf
	...
docs/                   # Documentos procesados y fragmentados
	turism/
		TENERIFE.txt
		...
```

## Instalación
1. Clona el repositorio y accede a la carpeta del proyecto.
2. Instala las dependencias:
	 ```powershell
	 python -m pip install -r requirements.txt
	 ```
3. Crea un archivo `.env` con tu clave de API de Pinecone y OpenAI:
	 ```env
	 PINECONE_API_KEY=tu_clave_pinecone
	 OPENAI_API_KEY=tu_clave_openai
	 ```
4. Ejecuta la aplicación:
	 ```powershell
	 streamlit run app.py
	 ```

## Uso
- **Entrenamiento:** Sube documentos relevantes en la pestaña "Entrenamiento".
- **Chatbot:** Realiza preguntas sobre turismo de Tenerife en la pestaña "Chatbot". El asistente solo responderá usando la información de los documentos subidos.

## Requisitos
- Python 3.8+
- Claves de API de Pinecone y OpenAI

## Créditos
Desarrollado por [Tu Nombre].

## Licencia
Este proyecto se distribuye bajo la licencia MIT.
