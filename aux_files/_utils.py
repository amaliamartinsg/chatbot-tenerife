import tempfile
import os
import fitz
import logging
from typing import List
from dotenv import load_dotenv
load_dotenv()

from pinecone import Pinecone
from streamlit.runtime.uploaded_file_manager import UploadedFile
from langchain import hub
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings
from langchain.prompts import PromptTemplate
from langchain_pinecone import PineconeVectorStore
from langchain_community.document_loaders import (
    TextLoader,
    UnstructuredMarkdownLoader
)
from langchain.tools import Tool
from langchain.agents import initialize_agent, AgentType
from langchain.agents import create_openai_functions_agent, AgentExecutor
from langchain.tools import StructuredTool
from langchain.chains.history_aware_retriever import create_history_aware_retriever
from langchain.chains.retrieval import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_openai import ChatOpenAI
from aux_files.aux_weather import get_weather, WeatherInput, get_date_info

# Genera el prompt de parámetros a partir de WeatherInput
def get_weatherinput_prompt():
    fields = WeatherInput.model_fields
    prompt_lines = ["Parámetros requeridos para la función Tiempo_Meteorologico:"]
    for name, field in fields.items():
        tipo = field.annotation.__name__ if hasattr(field.annotation, "__name__") else str(field.annotation)
        prompt_lines.append(f"- {name}: {tipo} — {field.description}")
    return "\n".join(prompt_lines)



LOG_FILE = os.path.join(os.path.dirname(__file__), '..', 'chatbot.log')
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S',
    handlers=[
        logging.FileHandler(LOG_FILE, encoding='utf-8'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

embeddings = OpenAIEmbeddings(model="text-embedding-3-small")

def ingest_docs(uploaded_files: List[UploadedFile], assistant_id: str, index_name, delete_existing_files=False):
    try:
        if not os.path.exists("docs"):
            os.makedirs("docs")
            logger.info("Carpeta 'docs' creada")

        # Verificar que existe la carpeta específica del índice
        index_dir = os.path.join("docs", index_name)
        if not os.path.exists(index_dir):
            os.makedirs(index_dir)
            logger.info(f"Carpeta '{index_dir}' creada")
        # Inicializar cliente de Pinecone
        pc = Pinecone(api_key=os.environ.get("PINECONE_API_KEY"))

        # Comprobar si el índice existe
        existing_indexes = [idx.name for idx in pc.list_indexes()]
        index_exists = index_name in existing_indexes

        if not index_exists:
            # Crear el índice si no existe
            logger.info(f"Creando nuevo índice: {index_name}")
            pc.create_index(
                name=index_name,
                dimension=1536,  # Dimensión para text-embedding-3-small
                metric="cosine",
                spec={
                    # "replicas": 1,
                    # "shard_size": 1000,
                    "serverless": {
                        "cloud": "aws",
                        "region": "us-east-1"  # o la región que prefieras
                    }
                }
            )

        all_documents = []

        # Procesar cada archivo subido
        for uploaded_file in uploaded_files:
            # Crear un archivo temporal para guardarlo
            with tempfile.NamedTemporaryFile(delete=False, suffix=os.path.splitext(uploaded_file.name)[1]) as temp_file:
                temp_file.write(uploaded_file.getvalue())
                temp_path = temp_file.name

            # Procesar el archivo después de cerrar el bloque 'with'
            try:
                file_documents = []
                if uploaded_file.name.endswith('.pdf'):
                    # Extraer texto y enlaces del PDF
                    pdf_doc = fitz.open(temp_path)
                    full_text = ""
                    for page_num in range(pdf_doc.page_count):
                        page = pdf_doc.load_page(page_num)
                        text = page.get_text()
                        full_text += f"\n--- Página {page_num + 1} ---\n{text}"
                        links = []
                        for link in page.get_links():
                            uri = link.get('uri')
                            if uri:
                                links.append(uri)
                        # Crear documento con texto y enlaces en metadatos
                        file_documents.append(type('Doc', (), {
                            'page_content': text,
                            'metadata': {
                                'filename': uploaded_file.name,
                                'filetype': uploaded_file.type,
                                'assistant_id': assistant_id,
                                'page': page_num + 1,
                                'links': links
                            }
                        }))
                    # Guardar el texto completo en docs/{index_name}/{filename}.txt
                    output_dir = os.path.join("docs", index_name)
                    os.makedirs(output_dir, exist_ok=True)
                    output_path = os.path.join(output_dir, f"{os.path.splitext(uploaded_file.name)[0]}.txt")
                    with open(output_path, "w", encoding="utf-8") as f:
                        f.write(full_text)
                    pdf_doc.close()
                elif uploaded_file.name.endswith('.md'):
                    loader = UnstructuredMarkdownLoader(temp_path)
                    file_documents = loader.load()
                elif uploaded_file.name.endswith(('.docx', '.txt', '.html', '.tf')):
                    loader = TextLoader(temp_path, encoding="utf-8")
                    file_documents = loader.load()
                else:
                    logger.warning(f"Tipo de archivo no soportado: {uploaded_file.name}")
                    continue

                logger.info(f"Cargados {len(file_documents)} documentos de {uploaded_file.name}")

                # Añadir metadatos del archivo original (solo para los que no son PDF)
                if not uploaded_file.name.endswith('.pdf'):
                    for doc in file_documents:
                        doc.metadata.update({
                            "filename": uploaded_file.name,
                            "filetype": uploaded_file.type,
                            "assistant_id": assistant_id
                        })

                all_documents.extend(file_documents)

            finally:
                # Asegurar la eliminación del archivo temporal
                os.unlink(temp_path)

        # Salir si no hay documentos
        if not all_documents:
            logger.warning("No se pudieron cargar documentos válidos")
            return

        # Dividir en chunks
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=200,
            add_start_index=True
        )
        documents = text_splitter.split_documents(all_documents)
        logger.info(f"Dividido en {len(documents)} chunks")

        # Definir tamaño del lote
        batch_size = 100
        total_batches = (len(documents) + batch_size - 1) // batch_size

        # Inicializar vectorstore con el índice existente
        vectorstore = PineconeVectorStore(index_name=index_name, embedding=embeddings)

        # Si se indica que se deben eliminar archivos existentes
        if delete_existing_files:
            # Obtener nombres de archivos a insertar
            filenames = [doc.metadata["filename"] for doc in all_documents if "filename" in doc.metadata]
            filenames = list(set(filenames))  # Eliminar duplicados

            if filenames:
                # Eliminar vectores con estos nombres de archivo
                vectorstore.delete(
                    filter={"filename": {"$in": filenames}}
                )
                logger.info(f"Eliminados documentos anteriores para los archivos: {', '.join(filenames)}")

        logger.info(f'Agregando {len(documents)} documentos a Pinecone en {total_batches} lotes')

        # Procesar por lotes
        for i in range(0, len(documents), batch_size):
            batch = documents[i:i+batch_size]
            end_idx = min(i+batch_size, len(documents))
            logger.info(f"Procesando lote {i//batch_size + 1}/{total_batches} (documentos {i+1}-{end_idx})")

            # Añadir documentos al índice existente
            vectorstore.add_documents(batch)

        logger.info("****Carga en el índice vectorial completada****")
        return True

    except Exception as e:
        logger.error(f"Error: {e}")
        return False


def get_docs_by_index(index_name: str, limit: int = 7, chunked = False):
    """
    Obtiene documentos de un índice específico en Pinecone.

    Args:
        index_name (str): Nombre del índice del cual obtener los documentos.
        limit (int): Número máximo de documentos a recuperar.
        chunked (bool): True si queremos los documentos fragmentados, False si no.

    Returns:
        list: Lista de documentos recuperados del índice.
    """
    try:
        pc = Pinecone(api_key=os.environ.get("PINECONE_API_KEY"))
        vectorstore = PineconeVectorStore(index_name=index_name, embedding=embeddings)
        docs = vectorstore.similarity_search("", k=limit)
        if chunked:
            return [doc.page_content for doc in docs]
        else:
            return docs
    except Exception as e:
        logger.error(f"Error al obtener documentos del índice {index_name}: {e}")
        return []


def run_llm_on_index(query: str, chat_history: list, index_name: str):
    """
    Ejecuta el modelo de lenguaje utilizando el índice especificado para responder consultas.
    """
    try:
        logger.info(f"[AGENTE] Nueva consulta recibida: '{query}' | Historial: {chat_history} | Índice: {index_name}")
        # Conexión al índice específico
        vectorstore = PineconeVectorStore(index_name=index_name, embedding=embeddings)

        # Configurar el LLM
        chat = ChatOpenAI(
            verbose=True,
            temperature=0.15,
            top_p=0.85,
            model='gpt-4o-mini', 
            max_tokens=4096
        )
        logger.info(f"[AGENTE] Parámetros del modelo: temperature={chat.temperature}, top_p={chat.top_p}, max_tokens={chat.max_tokens}")

        # Usar un prompt personalizado con instrucciones
        custom_prompt = PromptTemplate(
            input_variables=["context", "input"],
            template="""
            Eres un asistente experto en turismo de Tenerife. Responde siempre en español, de forma clara y concisa.
            # Si el contexto incluye información meteorológica, úsala para responder de manera precisa.
            # Si no sabes la respuesta, simplemente indícalo.
            # Si has utilizado información de los documentos proporcionados en el contexto para responder, incluye al final la frase exacta '*fuentes utilizadas:*' seguida de las fuentes utilizadas.
            # Si la respuesta es un saludo o no requiere información de los documentos, no incluyas la frase '*fuentes utilizadas:*' ni ninguna referencia a fuentes.
            Contexto:
            {context}
            Pregunta:
            {input}
            """
        )

        stuff_documents_chain = create_stuff_documents_chain(chat, custom_prompt)

        # Crear un retriever consciente del historial
        logger.info("[AGENTE] Descargando prompt de rephrase y configurando retriever...")
        rephrase_prompt = hub.pull("langchain-ai/chat-langchain-rephrase")
        history_aware_retriever = create_history_aware_retriever(
            llm=chat,
            retriever=vectorstore.as_retriever(),
            prompt=rephrase_prompt
        )

        # Crear la cadena de recuperación
        qa = create_retrieval_chain(
            retriever=history_aware_retriever,
            combine_docs_chain=stuff_documents_chain
        )

        # Tool principal: RAG
        def llm_tool_func(input, chat_history=None):
            logger.info(f"[AGENTE] Invocando QA chain con input: '{input}' y chat_history: {chat_history}")
            result = qa.invoke({"input": input, "chat_history": chat_history or []})
            logger.info(f"[AGENTE] Resultado QA chain: {result}")
            return result

        retriever_tool = Tool(
            name="Modelo_Lenguaje",
            func=llm_tool_func,
            description=(
                "Responde a preguntas sobre Tenerife utilizando el modelo de lenguaje y los documentos proporcionados."
            )
        )

        # Tool de la fecha
        date_tool = Tool(
            name="Info_Fechas",
            func=get_date_info,
            description=(
                "Obtiene la fecha y hora actual."
            )
        )
        
        # Tool del tiempo
        weather_tool = Tool(
            name="Tiempo_Meteorologico",
            func=get_weather,
            description=(
                get_weatherinput_prompt() +
                "\nObtiene el clima de una ciudad de Tenerife para una fecha concreta dada. "
                "Devuelve un JSON con la ciudad, fecha, temperatura_media (°C), "
                "precipitacion (%), humedad (%) y viento (km/h). "
                "Úsalo para dar al usuario una respuesta clara y natural en español."
            )
        )

        # Configuramos el agente
        system_prompt = """
            Eres un asistente que responde usando las herramientas disponibles.
            Si el usuario hace alguna referencia a fechas, SIEMPRE usa la herramienta `date_tool` para saber qué fecha es hoy y calcular la fecha en función de eso.
            Únicamente si el usuario pregunta qué tiempo hará en una zona concreta, usa la herramienta `weather_tool` (si hacen referencia a alguna fecha inclúyela en la petición).
            Si en la respuesta se añade algo de 'context', incluye al final la frase exacta '*fuentes utilizadas:*' seguida de las fuentes utilizadas.
            """
        logger.info("[AGENTE] Inicializando agente con herramientas...")
        agent = initialize_agent(
            tools=[retriever_tool, date_tool, weather_tool],
            llm=chat,
            agent=AgentType.OPENAI_FUNCTIONS,
            verbose=True,
            handle_parsing_errors=True,
            agent_kwargs={"system_message": system_prompt}
        )

        # Ejecutar el retriever_tool para obtener los documentos consultados
        logger.info("[AGENTE] Ejecutando retriever_tool para obtener documentos consultados...")
        retriever_output = retriever_tool.func(query, chat_history)
        logger.info(f"[AGENTE] Documentos consultados: {retriever_output}")
        context_docs = retriever_output.get('context', []) if isinstance(retriever_output, dict) else []

        # Ejecutar el agente para obtener la respuesta final
        logger.info("[AGENTE] Invocando agente para obtener respuesta final...")
        rag_result = agent.invoke({"input": query, "chat_history": chat_history})
        logger.info(f"[AGENTE] Respuesta final del agente: {rag_result}")

        return {
            "query": rag_result.get('input', query),
            "result": rag_result.get('output', ''),
            "source_documents": context_docs
        }
    except Exception as e:
        logger.error(f"Error al ejecutar consulta en índice {index_name}: {e}")
        raise ValueError(f"Error al ejecutar consulta en índice {index_name}: {e}")


def create_sources_string(source_urls):
    """
    Formatea las URLs de las fuentes para mostrarlas en la interfaz.
    """
    if not source_urls:
        return ""

    sources_list = list(set(source_urls))  # Eliminar duplicados
    sources_list.sort()
    sources_string = "Fuentes:\n"

    for i, source in enumerate(sources_list):
        sources_string += f"{i + 1}. {source}\n"

    return sources_string


if __name__ == "__main__":
    pass