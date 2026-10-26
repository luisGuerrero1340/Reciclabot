# ReciclaBot · Proyecto Fase 1

**Materia:**  Desarrollo de aplicaciones con Inteligencia Artificial
**Avance de proyecto 1** · Semana 2 
**Estudiante:** Luis Angel Guerrero

ReciclaBot es un asistente conversacional **basado en reglas**, desarrollado con **Python 3.10+ y NLTK**, que orienta a las personas sobre cómo separar correctamente los residuos domésticos (orgánicos, reciclables, no reciclables, de manejo especial y peligrosos), a dónde llevarlos y cómo generar menos basura.

> La información es orientativa. Las reglas de separación, los colores de los contenedores y los horarios de recolección dependen de cada municipio.

## Contenido del repositorio

| Archivo | Descripción |
|---|---|
| `Proyecto_Fase1_ReciclaBot.ipynb` | Notebook de Google Colab con la definición del problema, la arquitectura, la base de conocimiento, el motor de reglas, las 12 conversaciones de prueba y la exportación de evidencias. |
| `reciclabot.py` | El mismo código del notebook como script ejecutable fuera de Colab. |
| `requirements.txt` | Dependencias (`nltk`, `pandas`). |
| `evidencias/` | Carpeta sugerida para guardar el ZIP de evidencias y las capturas de pantalla de la ejecución en Colab. |

## Cómo ejecutarlo

**En Google Colab (recomendado):** abre el notebook en Colab (Archivo → Subir notebook, o desde GitHub) y usa *Entorno de ejecución → Ejecutar todas*. La última celda genera `Evidencias_ReciclaBot.zip` con el CSV y el TXT de las conversaciones.

**En tu computadora:**

```bash
pip install -r requirements.txt
python reciclabot.py
```

Para conversar manualmente cambia `ACTIVAR_CHAT = True` en la celda o en el script.

## Cómo funciona

1. **Normalización:** minúsculas, sin acentos ni espacios repetidos.
2. **Preprocesamiento con NLTK:** tokenización (`RegexpTokenizer`), eliminación de *stopwords* en español y *stemming* (`SnowballStemmer('spanish')`).
3. **Identificación de la intención:** cada intención temática recibe 1 punto por raíz coincidente y 2 por frase coincidente; gana la de mayor puntaje (en empate, la primera de la lista). Si ninguna coincide, se evalúan las reglas conversacionales con expresiones regulares; si tampoco, se responde `desconocida`.
4. **Respuesta:** se devuelve el texto asociado a la intención en la base de conocimiento.

## Intenciones (18)

| Tipo | Intenciones |
|---|---|
| Temáticas (12) | `organicos`, `plastico`, `papel_carton`, `vidrio`, `metal`, `electronicos`, `peligrosos`, `sanitarios`, `recoleccion`, `acopio`, `consejos`, `importancia` |
| Conversacionales (4) | `ayuda`, `despedida`, `agradecimiento`, `saludo` |
| Especiales (2) | `vacio`, `desconocida` |

## Pruebas

12 conversaciones distintas (30 turnos) con intención esperada declarada antes de ejecutar. Resultado de la versión actual: 30/30 turnos correctos. Las limitaciones observadas (sin contexto entre turnos, sin corrección ortográfica, consultas con dos temas, negaciones e información general por municipio) y las mejoras propuestas se documentan en la sección 7 del notebook y en el PDF del avance.

## Siguientes fases

Clasificación de intenciones con técnicas de PLN, clasificación de imágenes de residuos e integración en una interfaz web.
