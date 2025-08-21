# Informe Final: Chatbot de Turismo de Tenerife

## Diseño de la Solución
El proyecto implementa un asistente conversacional para turismo en Tenerife, basado en IA. Utiliza LangChain, OpenAI y Pinecone para procesar documentos, generar embeddings y realizar búsquedas semánticas. La interfaz web está desarrollada con Streamlit, permitiendo interacción en dos pestañas: "Chatbot" y "Entrenamiento". Los usuarios pueden subir documentos relevantes para alimentar el modelo y realizar consultas contextuales.

## Decisiones Técnicas
- **Frameworks y librerías:**
  - LangChain para la orquestación de agentes y herramientas.
  - OpenAI para generación de embeddings y respuestas LLM.
  - Pinecone para almacenamiento y recuperación de vectores.
  - Streamlit para la interfaz web interactiva.
- **Herramientas del agente:**
  - RAG: Recuperación de contexto documental.
  - Herramienta para consultar el tiempo en las diferentes zonas de Tenerife.
  - Herramienta para obtener la fecha actual y construir la respuesta fial teniendo en cuenta esta información.
  - Presentación de fuentes consultadas para poder mejorar la trazabilidad de la respuesta.

## Resultados
- El asistente responde preguntas sobre turismo de Tenerife usando únicamente la información de los documentos subidos. Aporta información recogida en los documentos y completa con información del clima cuando el usuario lo requiere. Utiliza el historial de la conversación para poder identificar la zona de la que se habla.
- El agente decide en función de la pregunta del usuario de qué herramienta obtener la respuesta.

## Limitaciones
- El sistema depende de la calidad y relevancia de los documentos subidos; si la información no está presente, el asistente no puede responder.
- La herramienta meteorológica está limitada a datos disponibles y puede no cubrir todas las fechas o zonas.
- El modelo LLM puede generar respuestas genéricas si el contexto documental es insuficiente.

## Mejora Futura
- Integrar una base de datos documental más robusta y actualizada.
- Añadir chatbot multimodal.
- Añadir respuestas en streaming.
- Añadir soporte para otros idiomas y ampliar el alcance geográfico.
- Añadir analítica de uso y feedback para mejorar el sistema de manera continua.
