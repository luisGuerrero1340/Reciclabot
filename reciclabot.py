# -*- coding: utf-8 -*-
"""
ReciclaBot - Proyecto Fase 1 (Desarrollo de aplicaciones con Inteligencia Artificial)
Chatbot basado en reglas con Python 3.10+ y NLTK para orientar sobre la separación
de residuos domésticos. Es el mismo código que el notebook.

Uso:  pip install -r requirements.txt
      python reciclabot.py
"""
import sys
assert sys.version_info >= (3, 10), 'Se requiere Python 3.10 o superior.'
# En Colab: %pip -q install nltk==3.9.1
import nltk
nltk.download('stopwords', quiet=True)   # lista de palabras vacías en español
print('Python:', sys.version.split()[0], '| NLTK:', nltk.__version__)

# =====================================================================
# BASE DE CONOCIMIENTO
# Intenciones temáticas: identificador, descripción, ejemplos de consulta,
# palabras/frases clave y respuesta. El orden define la prioridad en empates.
# =====================================================================
INTENCIONES_TEMATICAS = [
    {
        'intencion': 'organicos',
        'descripcion': 'Separación de residuos orgánicos (restos de comida y jardín)',
        'ejemplos': ['¿Dónde tiro las cáscaras de fruta?', 'Tengo restos de comida y café molido'],
        'claves': ['organico', 'organicos', 'cascara', 'cascaras', 'restos de comida', 'comida', 'fruta', 'frutas',
                   'verdura', 'verduras', 'cafe', 'cascaron', 'huesos', 'tortilla', 'jardin', 'pasto', 'hojas secas', 'composta'],
        'respuesta': 'Los residuos orgánicos (cáscaras, restos de comida, café, cascarón y residuos de jardín) van en el '
                     'contenedor de orgánicos, que normalmente se identifica con el color verde. Escúrrelos bien y no los '
                     'mezcles con plástico, papel ni bolsas. Si tienes espacio, puedes aprovecharlos para hacer composta. '
                     'Confirma el color del contenedor y el día de recolección con tu municipio.',
    },
    {
        'intencion': 'plastico',
        'descripcion': 'Separación de plásticos (PET, envases, bolsas, unicel)',
        'ejemplos': ['¿Cómo separo las botellas de plástico?', '¿El unicel se recicla?'],
        'claves': ['plastico', 'plasticos', 'pet', 'botella', 'botellas', 'botella de plastico', 'envase', 'envases',
                   'bolsa', 'bolsas', 'unicel', 'popote', 'popotes', 'tapa', 'tapas', 'tupper', 'polietileno', 'pvc'],
        'respuesta': 'Los plásticos reciclables (botellas PET, envases de HDPE como los de champú o detergente y las tapas) '
                     'van en inorgánicos reciclables: enjuágalos, aplástalos y sepáralos de otros materiales. Las bolsas, '
                     'los popotes y el unicel tienen poca aceptación en los centros de acopio; pregunta si los reciben y, '
                     'si no, deposítalos en inorgánicos no reciclables.',
    },
    {
        'intencion': 'papel_carton',
        'descripcion': 'Separación de papel y cartón',
        'ejemplos': ['Tengo cajas de cartón y periódicos viejos', '¿Dónde van las revistas?'],
        'claves': ['papel', 'carton', 'periodico', 'periodicos', 'revista', 'revistas', 'caja', 'cajas', 'cuaderno',
                   'cuadernos', 'libro', 'libros', 'hojas de papel', 'folleto', 'folletos'],
        'respuesta': 'El papel, el cartón, los periódicos, las revistas y las cajas van en inorgánicos reciclables, siempre '
                     'limpios y secos. Dobla o aplana las cajas para ahorrar espacio. El papel manchado de grasa (como las '
                     'cajas de pizza sucias), el papel encerado y las servilletas usadas no se reciclan: van en inorgánicos '
                     'no reciclables.',
    },
    {
        'intencion': 'vidrio',
        'descripcion': 'Separación de vidrio (botellas, frascos) y vidrio no reciclable',
        'ejemplos': ['¿Los frascos de vidrio se reciclan?', '¿Qué hago con un espejo roto?'],
        'claves': ['vidrio', 'vidrios', 'frasco', 'frascos', 'cristal', 'botella de vidrio', 'envase de vidrio', 'foco',
                   'focos', 'espejo', 'espejos', 'ceramica', 'loza', 'ventana'],
        'respuesta': 'Las botellas y los frascos de vidrio van en inorgánicos reciclables: enjuágalos y retira las tapas. '
                     'No los mezcles con focos, espejos, cerámica, loza ni vidrio de ventanas, que tienen otra composición '
                     'y no se reciclan con el vidrio común; envuélvelos para evitar cortaduras y deposítalos en inorgánicos '
                     'no reciclables (los focos ahorradores van a un punto de manejo especial).',
    },
    {
        'intencion': 'metal',
        'descripcion': 'Separación de metales (latas, aluminio, chatarra)',
        'ejemplos': ['¿Dónde tiro las latas de aluminio?', 'Tengo latas de hojalata y chatarra'],
        'claves': ['lata', 'latas', 'aluminio', 'metal', 'metales', 'hojalata', 'chatarra', 'cobre', 'fierro', 'hierro',
                   'acero', 'aerosol', 'aerosoles'],
        'respuesta': 'Las latas de aluminio y hojalata, las tapas metálicas y la chatarra pequeña van en inorgánicos '
                     'reciclables. Enjuaga y aplasta las latas para reducir volumen. Los aerosoles solo se reciclan si están '
                     'completamente vacíos; si contienen restos de pintura o químicos, trátalos como residuo peligroso.',
    },
    {
        'intencion': 'electronicos',
        'descripcion': 'Residuos electrónicos y pilas (manejo especial)',
        'ejemplos': ['Tengo pilas usadas y un celular viejo', '¿Dónde tiro un cargador que ya no sirve?'],
        'claves': ['pila', 'pilas', 'bateria', 'baterias', 'celular', 'celulares', 'telefono', 'computadora', 'laptop',
                   'cargador', 'cargadores', 'electronico', 'electronicos', 'cable', 'cables', 'television', 'televisor',
                   'pantalla', 'impresora', 'audifonos', 'tablet'],
        'respuesta': 'Las pilas, baterías, celulares, cargadores, cables y demás aparatos electrónicos son residuos de '
                     'manejo especial: no van en la basura común porque contienen metales pesados. Guárdalos en una bolsa o '
                     'caja aparte y llévalos a un punto de acopio de electrónicos (por ejemplo, jornadas como el '
                     'Reciclatrón en la Ciudad de México) o a los contenedores de pilas que hay en escuelas, plazas y '
                     'tiendas participantes.',
    },
    {
        'intencion': 'peligrosos',
        'descripcion': 'Residuos peligrosos (medicamentos, aceites, pinturas, químicos)',
        'ejemplos': ['Se me caducaron unos medicamentos', '¿Qué hago con el aceite de cocina usado?'],
        'claves': ['medicina', 'medicinas', 'medicamento', 'medicamentos', 'pastillas', 'jarabe', 'aceite', 'pintura',
                   'pinturas', 'solvente', 'solventes', 'quimico', 'quimicos', 'jeringa', 'jeringas', 'aguja', 'agujas',
                   'insecticida', 'termometro', 'gasolina', 'thinner'],
        'respuesta': 'Los medicamentos caducos, aceites, pinturas, solventes, jeringas y productos químicos son residuos '
                     'peligrosos: nunca los tires al drenaje ni a la basura común. Los medicamentos se entregan en los '
                     'contenedores del programa SINGREM que hay en farmacias; el aceite de cocina usado, en frascos '
                     'cerrados, en centros de acopio; y las pinturas, solventes y jeringas, en las jornadas de residuos '
                     'peligrosos de tu municipio.',
    },
    {
        'intencion': 'sanitarios',
        'descripcion': 'Residuos sanitarios e inorgánicos no reciclables',
        'ejemplos': ['¿Dónde van los pañales?', '¿El papel higiénico se recicla?'],
        'claves': ['pañal', 'pañales', 'papel higienico', 'toalla sanitaria', 'toallas sanitarias', 'toallitas',
                   'cubrebocas', 'colilla', 'colillas', 'cigarro', 'cigarros', 'chicle', 'chicles', 'servilleta',
                   'servilletas', 'curita', 'curitas', 'hisopo', 'hisopos', 'rastrillo'],
        'respuesta': 'Los pañales, el papel higiénico, las toallas sanitarias, las toallitas húmedas, los cubrebocas, las '
                     'colillas y los chicles son residuos sanitarios, es decir, inorgánicos no reciclables. Colócalos en '
                     'una bolsa cerrada e identificada y deposítalos en el contenedor de no reciclables; nunca los mezcles '
                     'con los orgánicos ni con los reciclables.',
    },
    {
        'intencion': 'recoleccion',
        'descripcion': 'Días y horarios de recolección',
        'ejemplos': ['¿Qué día pasa el camión de la basura?', '¿Cuál es el horario de recolección?'],
        'claves': ['recoleccion', 'recolectan', 'recogen', 'camion', 'camion de la basura', 'horario', 'horarios',
                   'cuando pasa', 'que dia pasa', 'calendario'],
        'respuesta': 'Los días y horarios de recolección dependen de tu colonia y municipio, y ReciclaBot no tiene acceso a '
                     'esa información. Consulta el calendario de recolección en el sitio web o las redes sociales de tu '
                     'alcaldía o municipio, o pregunta al personal del camión. En muchas localidades los orgánicos y los '
                     'inorgánicos se recogen en días distintos.',
    },
    {
        'intencion': 'acopio',
        'descripcion': 'Dónde entregar materiales reciclables',
        'ejemplos': ['¿Hay un centro de acopio cerca?', '¿Dónde puedo llevar mis reciclables?'],
        'claves': ['acopio', 'centro de acopio', 'centros de acopio', 'llevar', 'donde puedo llevar', 'entregar',
                   'reciclatron', 'punto limpio', 'punto de reciclaje', 'contenedor', 'contenedores', 'maquina'],
        'respuesta': 'Para entregar tus reciclables busca los centros de acopio, puntos limpios o jornadas de reciclaje de '
                     'tu municipio; además, algunos supermercados y plazas tienen máquinas o contenedores para PET, latas y '
                     'pilas. Lleva los materiales limpios, secos y separados por tipo. Revisa el directorio de centros de '
                     'acopio de tu localidad para ubicar el más cercano.',
    },
    {
        'intencion': 'consejos',
        'descripcion': 'Consejos para reducir y reutilizar',
        'ejemplos': ['Dame consejos para generar menos basura', '¿Cómo puedo reducir mi basura?'],
        'claves': ['consejo', 'consejos', 'tip', 'tips', 'reducir', 'reutilizar', 'reusar', 'menos basura', 'evitar',
                   'ahorrar', 'tres erres', '3r', 'sustentable', 'sostenible', 'recomendaciones'],
        'respuesta': 'Tres acciones sencillas: 1) Reduce: lleva bolsa y termo reutilizables y evita productos con exceso '
                     'de empaque. 2) Reutiliza: los frascos, las cajas y las bolsas pueden tener una segunda vida antes de '
                     'desecharse. 3) Separa desde casa: orgánicos, reciclables y no reciclables en contenedores distintos. '
                     'Comprar a granel y hacer composta reducen todavía más tu basura.',
    },
    {
        'intencion': 'importancia',
        'descripcion': 'Qué es el reciclaje y por qué es importante separar',
        'ejemplos': ['¿Por qué es importante reciclar?', '¿Qué es el reciclaje?'],
        'claves': ['importancia', 'importante', 'por que reciclar', 'para que sirve', 'que es reciclar',
                   'que es el reciclaje', 'beneficio', 'beneficios', 'medio ambiente', 'ambiente', 'contaminacion',
                   'planeta', 'por que separar'],
        'respuesta': 'Separar y reciclar reduce la cantidad de residuos que llegan a los rellenos sanitarios, permite '
                     'recuperar materiales como PET, aluminio, papel y vidrio, y disminuye la contaminación y el consumo de '
                     'recursos naturales. La separación correcta desde casa es el primer paso para que el reciclaje sea '
                     'posible: un material contaminado con restos de comida muchas veces ya no se puede recuperar.',
    },
]

# Reglas conversacionales: se evalúan con expresiones regulares sobre el texto
# normalizado, sólo cuando ninguna intención temática obtiene puntaje.
REGLAS_CONVERSACIONALES = [
    {
        'intencion': 'ayuda',
        'descripcion': 'Menú de temas disponibles',
        'ejemplos': ['¿Qué puedes hacer?', 'menú'],
        'patron': r'\b(ayuda|ayudame|menu|opciones|que puedes hacer|que sabes hacer|temas|instrucciones|como funcionas|reciclar|reciclaje|reciclo)\b',
        'respuesta': 'Puedo orientarte sobre cómo separar: orgánicos, plásticos, papel y cartón, vidrio, metales, '
                     'electrónicos y pilas, residuos peligrosos y residuos sanitarios. También sobre dónde llevar los '
                     'reciclables, la recolección y consejos para generar menos basura. Escribe "salir" para terminar.',
    },
    {
        'intencion': 'despedida',
        'descripcion': 'Cierre de la conversación',
        'ejemplos': ['Adiós', 'salir'],
        'patron': r'\b(adios|salir|hasta luego|hasta pronto|bye|chao|nos vemos|me voy)\b',
        'respuesta': '¡Hasta luego! Gracias por usar ReciclaBot. Recuerda: separar bien es reciclar mejor.',
    },
    {
        'intencion': 'agradecimiento',
        'descripcion': 'Agradecimiento del usuario',
        'ejemplos': ['Gracias', 'Perfecto, muy amable'],
        'patron': r'\b(gracias|te agradezco|muy amable|perfecto|excelente|genial)\b',
        'respuesta': '¡Con gusto! Cada residuo bien separado cuenta. ¿Tienes otra duda sobre reciclaje?',
    },
    {
        'intencion': 'saludo',
        'descripcion': 'Inicio de la conversación',
        'ejemplos': ['Hola', 'Buenos días'],
        'patron': r'^(hola|buenos dias|buenas tardes|buenas noches|buen dia|hey|que tal|saludos|buenas)\b',
        'respuesta': '¡Hola! Soy ReciclaBot, tu asistente para separar correctamente los residuos en casa. Pregúntame, '
                     'por ejemplo, "¿dónde tiro las pilas?" o "¿cómo separo el plástico?". Escribe "ayuda" para ver '
                     'todos los temas.',
    },
]

# Intenciones especiales: entrada vacía y consulta no reconocida.
RESPUESTAS_ESPECIALES = {
    'vacio': 'No escribiste ninguna consulta. Cuéntame qué residuo quieres separar o escribe "ayuda" para ver los temas.',
    'desconocida': 'No logré identificar tu consulta. Puedo ayudarte con orgánicos, plásticos, papel y cartón, vidrio, '
                   'metales, electrónicos, residuos peligrosos, residuos sanitarios, centros de acopio, recolección y '
                   'consejos. Intenta mencionar el material, por ejemplo: "¿dónde tiro los frascos de vidrio?".',
}

print(f'Intenciones temáticas: {len(INTENCIONES_TEMATICAS)} | conversacionales: {len(REGLAS_CONVERSACIONALES)} '
      f'| especiales: {len(RESPUESTAS_ESPECIALES)} | total: '
      f'{len(INTENCIONES_TEMATICAS) + len(REGLAS_CONVERSACIONALES) + len(RESPUESTAS_ESPECIALES)}')

import re
import unicodedata
from nltk.corpus import stopwords
from nltk.stem.snowball import SnowballStemmer
from nltk.tokenize import RegexpTokenizer

stemmer = SnowballStemmer('spanish')      # reduce las palabras a su raíz: botellas -> botell
tokenizador = RegexpTokenizer(r'\w+')     # separa palabras y descarta signos de puntuación


def normalizar(texto):
    """Convierte a minúsculas, elimina acentos y espacios repetidos."""
    texto = unicodedata.normalize('NFD', texto.lower().strip())
    texto = ''.join(c for c in texto if unicodedata.category(c) != 'Mn')
    return re.sub(r'\s+', ' ', texto)


try:
    STOPWORDS = {normalizar(p) for p in stopwords.words('spanish')}
except LookupError:   # respaldo si no se pudo descargar el corpus
    STOPWORDS = {'de', 'la', 'el', 'los', 'las', 'un', 'una', 'y', 'o', 'que', 'en', 'a', 'con', 'para', 'por',
                 'mi', 'mis', 'me', 'se', 'es', 'al', 'del', 'lo', 'le', 'su', 'sus', 'ya', 'como', 'donde', 'cuando'}


def preprocesar(texto):
    """Normaliza, tokeniza, elimina stopwords y aplica stemming (NLTK)."""
    tokens = tokenizador.tokenize(normalizar(texto))
    return [stemmer.stem(t) for t in tokens if t not in STOPWORDS]


# Precálculo de las claves: palabras sueltas -> raíz; frases -> texto normalizado.
for _intencion in INTENCIONES_TEMATICAS:
    _intencion['raices'] = {stemmer.stem(normalizar(c)) for c in _intencion['claves'] if ' ' not in c}
    _intencion['frases'] = [normalizar(c) for c in _intencion['claves'] if ' ' in c]


def puntuar(texto_normalizado, raices_usuario, intencion):
    """Puntaje = raíces coincidentes (1 punto) + frases coincidentes (2 puntos)."""
    puntos = sum(1 for r in raices_usuario if r in intencion['raices'])
    puntos += sum(2 for f in intencion['frases'] if f in texto_normalizado)
    return puntos


def identificar_intencion(texto):
    """Entrada del usuario -> (intención, respuesta, detalle del análisis)."""
    limpio = normalizar(texto)
    if not limpio:
        return 'vacio', RESPUESTAS_ESPECIALES['vacio'], {'raices': [], 'puntaje': 0}
    raices = preprocesar(texto)
    # 1) Intenciones temáticas: gana el mayor puntaje; en empate, la primera de la lista.
    puntajes = {i['intencion']: puntuar(limpio, raices, i) for i in INTENCIONES_TEMATICAS}
    mejor = max(INTENCIONES_TEMATICAS, key=lambda i: puntajes[i['intencion']])
    if puntajes[mejor['intencion']] > 0:
        return mejor['intencion'], mejor['respuesta'], {'raices': raices, 'puntaje': puntajes[mejor['intencion']]}
    # 2) Reglas conversacionales por expresión regular.
    for regla in REGLAS_CONVERSACIONALES:
        if re.search(regla['patron'], limpio):
            return regla['intencion'], regla['respuesta'], {'raices': raices, 'puntaje': 0}
    # 3) Consulta no reconocida.
    return 'desconocida', RESPUESTAS_ESPECIALES['desconocida'], {'raices': raices, 'puntaje': 0}


def responder(texto):
    """Devuelve un diccionario con la entrada, la intención identificada y la respuesta."""
    intencion, respuesta, detalle = identificar_intencion(texto)
    return {'entrada': texto, 'raices': detalle['raices'], 'puntaje': detalle['puntaje'],
            'intencion': intencion, 'respuesta': respuesta}


def ejecutar_pruebas(conversaciones, mostrar=True):
    """Ejecuta conversaciones [(entrada, intención esperada), ...] y devuelve los registros."""
    registros = []
    for numero, conversacion in enumerate(conversaciones, 1):
        if mostrar:
            print(f'\n=== Conversación {numero:02d} ===')
        for entrada, esperada in conversacion:
            salida = responder(entrada)
            correcto = salida['intencion'] == esperada
            registros.append({'conversacion': numero, 'entrada': entrada, 'raices': ' '.join(salida['raices']),
                              'esperada': esperada, 'intencion': salida['intencion'], 'correcto': correcto,
                              'respuesta': salida['respuesta']})
            if mostrar:
                print(f'Usuario   : {entrada!r}')
                print(f'ReciclaBot: {salida["respuesta"]}')
                print(f'Intención : {salida["intencion"]} | esperada: {esperada} | '
                      f'{"OK" if correcto else "REVISAR"}')
    return registros


# Resumen del diseño conversacional
print(f'{"Intención":<16}{"Ejemplo de entrada":<44}Respuesta (inicio)')
for i in INTENCIONES_TEMATICAS + REGLAS_CONVERSACIONALES:
    print(f'{i["intencion"]:<16}{i["ejemplos"][0]:<44}{i["respuesta"][:55]}...')
for nombre, respuesta in RESPUESTAS_ESPECIALES.items():
    print(f'{nombre:<16}{"(sin coincidencia)":<44}{respuesta[:55]}...')
print('\nEjemplo de preprocesamiento:', '¿Dónde tiro las cáscaras de fruta?', '->',
      preprocesar('¿Dónde tiro las cáscaras de fruta?'))

if __name__ == '__main__':
    fallos = []
    for intencion in INTENCIONES_TEMATICAS + REGLAS_CONVERSACIONALES:
        for ejemplo in intencion['ejemplos']:
            obtenida = identificar_intencion(ejemplo)[0]
            estado = 'OK' if obtenida == intencion['intencion'] else 'REVISAR'
            print(f'{estado:<8}{ejemplo!r:<48} -> {obtenida}')
            if estado != 'OK':
                fallos.append((ejemplo, intencion['intencion'], obtenida))
    print(f'\nEjemplos validados: {sum(len(i["ejemplos"]) for i in INTENCIONES_TEMATICAS + REGLAS_CONVERSACIONALES)} '
          f'| fallos: {len(fallos)}')
    assert not fallos, f'Ejemplos que requieren revisión: {fallos}'

    # Cada conversación es una lista de turnos (entrada del usuario, intención esperada).
    CONVERSACIONES = [
        [('Hola', 'saludo'), ('¿Dónde tiro las cáscaras de fruta?', 'organicos'), ('Gracias', 'agradecimiento')],
        [('Buenos días', 'saludo'), ('¿Cómo separo las botellas de plástico?', 'plastico'),
         ('¿Y las bolsas de plástico?', 'plastico'), ('Adiós', 'despedida')],
        [('Tengo cajas de cartón y periódicos viejos', 'papel_carton'),
         ('¿Qué hago con una caja de pizza sucia?', 'papel_carton')],
        [('¿Los frascos de vidrio se reciclan?', 'vidrio'), ('¿Y los focos?', 'vidrio'), ('salir', 'despedida')],
        [('¿DÓNDE TIRO LAS LATAS DE ALUMINIO?', 'metal'), ('gracias', 'agradecimiento')],
        [('Tengo pilas usadas y un celular viejo', 'electronicos'), ('¿Dónde las puedo llevar?', 'acopio')],
        [('Se me caducaron unos medicamentos', 'peligrosos'), ('¿Y el aceite de cocina usado?', 'peligrosos')],
        [('¿Dónde van los pañales y el papel higiénico?', 'sanitarios'), ('Hasta luego', 'despedida')],
        [('¿Qué día pasa el camión de la basura?', 'recoleccion'), ('¿Hay un centro de acopio cerca?', 'acopio')],
        [('¿Qué puedes hacer?', 'ayuda'), ('Dame consejos para generar menos basura', 'consejos'),
         ('Perfecto, gracias', 'agradecimiento')],
        [('   ', 'vacio'), ('¿Por qué es importante reciclar?', 'importancia')],
        [('¿Qué clima habrá mañana?', 'desconocida'), ('Quiero tirar plastco', 'desconocida'),
         ('¿Dónde tiro botellas de plástico y latas?', 'plastico')],
    ]
    print(f'Conversaciones: {len(CONVERSACIONES)} | turnos: {sum(len(c) for c in CONVERSACIONES)}')

    registros = ejecutar_pruebas(CONVERSACIONES)
    aciertos = sum(r['correcto'] for r in registros)
    print(f'\nResultado: {aciertos}/{len(registros)} turnos coinciden con la intención esperada.')
    print('Intenciones cubiertas:', len({r['intencion'] for r in registros}))

    import pandas as pd
    tabla = pd.DataFrame(registros)[['conversacion', 'entrada', 'raices', 'esperada', 'intencion', 'correcto']]
    try:
        display(tabla)
    except NameError:
        print(tabla.to_string(index=False))
    assert aciertos == len(registros), 'Hay turnos que requieren revisión.'

    import csv
    import zipfile
    from pathlib import Path
    from datetime import datetime, timezone

    carpeta = Path('evidencias_reciclabot')
    carpeta.mkdir(exist_ok=True)
    with (carpeta / 'pruebas.csv').open('w', newline='', encoding='utf-8-sig') as f:
        escritor = csv.DictWriter(f, fieldnames=list(registros[0]))
        escritor.writeheader()
        escritor.writerows(registros)
    lineas = ['ReciclaBot - evidencias de ejecución', 'Ejecución UTC: ' + datetime.now(timezone.utc).isoformat(),
              f'Python: {sys.version.split()[0]} | NLTK: {nltk.__version__}', '']
    for r in registros:
        lineas += [f"Conversación {r['conversacion']:02d} | Usuario: {r['entrada']!r}", 'ReciclaBot: ' + r['respuesta'],
                   f"Intención: {r['intencion']} | Esperada: {r['esperada']} | Correcto: {r['correcto']}", '']
    lineas.append(f'Resultado: {aciertos}/{len(registros)} turnos correctos.')
    (carpeta / 'conversaciones.txt').write_text('\n'.join(lineas), encoding='utf-8')
    with zipfile.ZipFile('Evidencias_ReciclaBot.zip', 'w', zipfile.ZIP_DEFLATED) as z:
        for archivo in carpeta.iterdir():
            z.write(archivo, arcname=archivo.name)
    print('Evidencias generadas: Evidencias_ReciclaBot.zip ->', [a.name for a in carpeta.iterdir()])

    DESCARGAR = False   # Cambia a True para descargar el ZIP desde Colab.
    if DESCARGAR:
        from google.colab import files
        files.download('Evidencias_ReciclaBot.zip')

    ACTIVAR_CHAT = False
    if ACTIVAR_CHAT:
        print('ReciclaBot: ¡Hola! Escribe tu consulta sobre residuos o "salir" para terminar.')
        while True:
            entrada = input('Tú: ')
            salida = responder(entrada)
            print(f'ReciclaBot [{salida["intencion"]}]: {salida["respuesta"]}')
            if salida['intencion'] == 'despedida':
                break
    else:
        print('Cambia ACTIVAR_CHAT a True para conversar manualmente.')
