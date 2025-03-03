import spacy
import re
from nltk.stem import PorterStemmer
import xml.etree.ElementTree as ET
import csv
stemmer = PorterStemmer() # Inicializa un stemmer de Porter para reducir palabras a su raíz

# Cargar el modelo en español de spaCy
nlp = spacy.load("es_core_news_sm") # Carga un modelo de procesamiento de lenguaje natural en español


# Dada la oración y el aspecto, extrae el segmento donde se encuentra el aspecto
def extraer_segmento(oracion,aspecto):
    """
    Extrae el segmento de una oración que contiene un aspecto dado. Combina varias funciones para identificar y retornar la porción relevante de texto.
    """
    dependencias = extraer_dependencias(oracion, aspecto) # Obtiene las dependencias sintácticas del aspecto en la oración
    lista = palabras_sin_repetir(dependencias) # Elimina las palabras repetidas de la lista de dependencias (función no definida aquí)
    segmento = combinacion(oracion,lista) # Combina las palabras en un segmento coherente (función no definida aquí)

    return convertir_lista_oracion(segmento) # Convierte la lista de palabras resultante en una oración (función no definida aquí)


def extraer_dependencias(texto_opinion, palabra_clave):
    """
    Extrae información sintáctica relacionada con una palabra clave en un texto. Identifica y devuelve frases relacionadas con el aspecto.
    """
    doc_analizado = nlp(texto_opinion) # Analiza el texto usando el modelo de spaCy
    frases_extraidas = [] # Inicializa una lista para guardar las frases extraídas

    for palabra_actual in doc_analizado: # Itera sobre cada palabra del texto analizado
        if palabra_actual.text == palabra_clave or palabra_actual.text == stemmer.stem(palabra_clave) : # Busca la palabra clave o su raíz

            palabras_asociadas = [] # Lista para guardar las palabras asociadas a la palabra clave

            # Extrae palabras a la izquierda y derecha de la palabra clave
            palabras_asociadas.extend(extraer_palabras_adyacentes(palabra_actual, 'left'))
            palabras_asociadas.append(palabra_actual.text)
            palabras_asociadas.extend(extraer_palabras_adyacentes(palabra_actual, 'right'))

            palabra_principal = palabra_actual.head # Encuentra la palabra principal (la raíz sintáctica) de la palabra clave

            # Extrae sujetos y objetos directos relacionados con la palabra principal
            palabras_asociadas.extend(extraer_palabras_adyacentes(palabra_principal, 'left'))
            palabras_asociadas.append(palabra_principal.text)
            palabras_asociadas.extend(extraer_palabras_adyacentes(palabra_principal, 'right'))

            # Agrega n-gramas, contexto y la frase extraída a la lista de frases
            frases_extraidas.append(palabras_asociadas)
            frases_extraidas.append(n_gramas(texto_opinion, palabra_clave, 3))
            frases_extraidas.append(extraccion_contexto(texto_opinion, palabra_clave))

    return frases_extraidas # Devuelve la lista de frases extraídas


# Funciones auxiliares para extraer dependencias

def extraer_palabras_adyacentes(token_actual, direccion):
    """
    Extrae las palabras a la izquierda o derecha de un token, incluyendo adjetivos relacionados con sustantivos.

    Args:
        token_actual: El token de referencia.
        direccion: 'left' para palabras a la izquierda, 'right' para palabras a la derecha.

    Returns:
        Una lista de palabras.
    """
    palabras = []
    adyacentes = token_actual.lefts if direccion == 'left' else token_actual.rights
    for token_adyacente in adyacentes:
        palabras.append(token_adyacente.text)
        if token_adyacente.dep_ == 'NOUN':
            for token_adjetivo in token_adyacente.lefts:
                if token_adjetivo.dep_ == 'ADJ':
                    palabras.append(token_adjetivo.text)
            for token_adjetivo in token_adyacente.rights:
                if token_adjetivo.dep_ == 'ADJ':
                    palabras.append(token_adjetivo.text)
    return palabras

def n_gramas(oracion, palabra, numero):
    """
    Extrae n-gramas (secuencias de n palabras) alrededor de una palabra clave en una oración.
    """
    palabras = limpiar_oracion(oracion).split() # Limpia la oración y la divide en palabras 
    index_palabra = [] # Inicializa una lista para guardar el índice de la palabra clave en la lista de palabras

    for index, word in enumerate(palabras): # Itera sobre las palabras para encontrar la palabra clave
        if word.lower() == palabra.lower() or word.lower() == stemmer.stem(palabra.lower()): # Busca la palabra clave o su raíz
            index_palabra = index
            break

    palabras_anteriores = palabras[max(0, index_palabra - numero):index_palabra] # Extrae las palabras anteriores a la palabra clave
    palabras_posteriores = palabras[index_palabra + 1:index_palabra + 1 + numero] # Extrae las palabras posteriores a la palabra clave

    return palabras_anteriores + [palabra] + palabras_posteriores # Devuelve una lista con las palabras anteriores, la palabra clave y las palabras posteriores


def extraccion_contexto(oracion, palabra_relevante):
    """
    Extrae el contexto de una palabra clave en una oración, incluyendo modificadores.
    """
    doc = nlp(oracion) # Analiza la oración con spaCy
    token_palabra_relevante = None # Inicializa la variable para guardar el token de la palabra clave
    for token in doc: # Busca el token de la palabra clave en la oración analizada
        if token.text.lower() == palabra_relevante:
            token_palabra_relevante = token
            break

    if not token_palabra_relevante: # Si no se encuentra la palabra clave
        return "Palabra relevante no encontrada en la oración"

    modificadores = [] # Inicializa una lista para guardar los modificadores de la palabra clave
    for token in doc: # Itera sobre los tokens de la oración analizada
        # Busca modificadores de la palabra clave (adjetivos, adverbios, etc.)
        if token.head == token_palabra_relevante and token.dep_ in ["amod", "advmod", "nsubj", "nummod", "det"]: 
            modificadores.append(token.text)

    return modificadores # Devuelve la lista de modificadores


def limpiar_oracion(oracion):

    # Usar una expresión regular para encontrar y reemplazar todos los caracteres que no sean letras
    oracion_limpia = re.sub(r'[^a-zA-Z\sáéíóúüÁÉÍÓÚÜñÑ]', ' ', oracion) 
    # Devolver la oración limpia
    return oracion_limpia

def palabras_sin_repetir(lista_de_listas):
    palabras = set()  # Utilizamos un conjunto para evitar palabras duplicadas  
    for sublista in lista_de_listas:
        for palabra in sublista:
            palabras.add(palabra) 

    return list(palabras)  # Convertimos el conjunto de palabras de nuevo a una lista y la devolvemos

def combinacion(oracion, lista_palabras):
    """
    Extrae las palabras de una oración que están presentes en una lista dada, 
    evitando duplicados y manejando algunas variaciones simples.

    Args:
        oracion: La oración de entrada como una cadena de texto.
        lista_palabras: Una lista de palabras a buscar en la oración.

    Returns:
        Una lista de palabras de la oración que se encontraron en la lista de palabras, sin duplicados.
    """
    resultado = [] # Inicializa una lista vacía para almacenar las palabras encontradas.

    for token in oracion.split(): # Itera sobre cada palabra (token) en la oración.
        if (token in lista_palabras) or (token[0:-1] in lista_palabras): # Verifica si el token o su forma sin el último carácter está en la lista de palabras.
            if not token in resultado: # Verifica si el token ya está en la lista de resultados para evitar duplicados.
                resultado.append(token) # Agrega el token a la lista de resultados.

    return resultado # Devuelve la lista de palabras encontradas.

def convertir_lista_oracion(lista_palabras):

    oracion = " ".join(lista_palabras)  # Unir las palabras con espacios
    return oracion



#///////////////////////////////////////////////////////////////////////////////////////////////////////



def process_xml_to_csv(xml_filename, csv_filename):
    """
    Procesa un archivo XML de reseñas y crea un archivo CSV, extrayendo solo la primera palabra del aspecto.
    """
    tree = ET.parse(xml_filename)
    root = tree.getroot()

    data = []
    for review in root.findall('Review'):
        for sentence in review.findall('sentences/sentence'):
            text = sentence.find('text').text
            for opinion in sentence.findall('Opinions/Opinion'):
                target = opinion.get('target')
                polarity = opinion.get('polarity')

                if target is not None:
                    aspect = target.split()[0] # Extrae la primera palabra
                else:
                    aspect = "GENERAL" # Para opiniones sin target específico

                data.append([text, aspect, 1 if polarity == 'positive' else 0])

    with open(csv_filename, 'w', newline='', encoding='utf-8') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(['texto', 'aspecto', 'polaridad'])
        writer.writerows(data)



