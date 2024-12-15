from colorama import Fore, Style
from Aspect_extraction.absapipeline import ABSAPipeline
import pandas as pd
from tqdm import tqdm
import numpy as np
from collections import Counter
from Aspect_extraction.absapipeline import metrics,predicted_bitmask


def ensamble_average(MODEL_ENSAMBLE, TRAINED_MODEL_ENSAMBLE, predict_data):
    """
    Realiza un ensamble de modelos mediante promediado de predicciones.

    Args:
        MODEL_ENSAMBLE: Lista de nombres o identificadores de los modelos.
        TRAINED_MODEL_ENSAMBLE: Lista de los modelos entrenados cargados. Debe coincidir en longitud con MODEL_ENSAMBLE.
        predict_data: Ruta al archivo CSV con los datos de prueba. Debe contener columnas 'text_tokens' y 'tags'.

    Returns:
        None. Imprime las métricas de evaluación.
    """

    print(f"\n{Fore.YELLOW}Cargando modelos...{Style.RESET_ALL}")

    num_models = len(MODEL_ENSAMBLE)
    if num_models != len(TRAINED_MODEL_ENSAMBLE):
        raise ValueError("El número de modelos y modelos entrenados debe coincidir.")

    all_predictions = []
    test_reviews = pd.read_csv(predict_data, sep=';')
    reviews = test_reviews['text_tokens']
    true_labels = test_reviews['tags'].tolist()

    for i in range(num_models):
        print(f"Procesando modelo {i+1} de {num_models}")
        pipeline = ABSAPipeline(MODEL_ENSAMBLE[i])
        pipeline.aspect_model.load_model(pipeline.aspect_model.model, TRAINED_MODEL_ENSAMBLE[i])
        model_predictions = []
        all_prob = []
        for rev in tqdm(reviews):
            tokens, aspects, prob_aspct = pipeline.predict_aspect(rev)
            all_prob.append(prob_aspct)
            model_predictions.append(predicted_bitmask(eval(rev), aspects))
        all_predictions.append(all_prob)
        

    predicted_labels_final = []
    for i in range(len(reviews)): # Iterar sobre cada revisión
        predicciones = []
        for pred in all_predictions:
            if len(pred[i]) >= len_elemnt(true_labels[i]):
                predicciones.append(pred[i][:len_elemnt(true_labels[i])])  # Recortar
            else:
                predicciones.append(np.pad(pred[i], (0, len_elemnt(true_labels[i]) - len(pred[i])), 'constant')) # Rellenar con ceros

        prediccion_final = np.mean(predicciones, axis=0)
        prediccion_final = np.where(prediccion_final >= 0.400, 1, 0)
        predicted_labels_final.append(prediccion_final)
    
    print(f"\n{Fore.CYAN}Métricas:{Style.RESET_ALL}")
    metrics(true_labels, predicted_labels_final)

def len_elemnt(elemnt):
    # Eliminar los corchetes y separar los elementos
    lista_str = elemnt[1:-1].split(',')
    # Convertir los elementos a enteros
    lista_int = [int(x.strip()) for x in lista_str]
    
    return len(lista_int)

def ensamble_max(MODEL_ENSAMBLE, TRAINED_MODEL_ENSAMBLE, predict_data):
    """
    Realiza un ensamble de modelos para extracción de aspectos mediante votación máxima.

    Args:
        MODEL_ENSAMBLE: Lista de modelos.
        TRAINED_MODEL_ENSAMBLE: Lista de rutas a los modelos entrenados.
        predict_data: Ruta al archivo CSV con los datos de prueba.

    Returns:
        None. Imprime las métricas del ensamble.
    """

    print(f"\n{Fore.YELLOW}Cargando modelos...{Style.RESET_ALL}")

    # Carga los datos
    test_reviews = pd.read_csv(predict_data, sep=';')
    reviews = test_reviews['text_tokens'].tolist()
    true_labels = test_reviews['tags'].tolist()

    # Lista para almacenar las predicciones de cada modelo
    model_predictions = []

    for i, (model, trained_model) in enumerate(zip(MODEL_ENSAMBLE, TRAINED_MODEL_ENSAMBLE)):
        print(f"Procesando modelo {i+1}...")
        pipeline = ABSAPipeline(model)
        pipeline.aspect_model.load_model(pipeline.aspect_model.model, trained_model)

        predicted_labels = []
        for rev in tqdm(reviews):
            tokens, aspects, prob_asp = pipeline.predict_aspect(rev)
            predicted_labels.append(predicted_bitmask(eval(rev), aspects))

        model_predictions.append(predicted_labels)

    # Ensamblaje por votación máxima
    predicted_labels_final = []
    num_models = len(model_predictions)

    for i in range(len(reviews)):
        suma_votos = np.sum(np.array(model_predictions[j][i]) for j in range(num_models)) # Convierte a array NumPy
        ensamble_votacion = suma_votos >= (num_models // 2 + 1)
        predicted_labels_final.append(ensamble_votacion.astype(int).tolist())

    print(f"\n{Fore.CYAN}Métricas:{Style.RESET_ALL}")
    metrics(true_labels, predicted_labels_final) # Asumiendo que metrics está definido en otro lugar


def ensamble_weighted_average(MODEL_ENSAMBLE, TRAINED_MODEL_ENSAMBLE, predict_data, model_weights):
    """
    Ensamble de modelos con promedio ponderado, manejando longitudes variables de predicciones.

    Args:
        MODEL_ENSAMBLE: Lista de nombres o identificadores de los modelos.
        TRAINED_MODEL_ENSAMBLE: Lista de los modelos entrenados cargados.
        predict_data: Ruta al archivo CSV con los datos de prueba.
        model_weights: Lista de pesos para cada modelo. Debe tener la misma longitud que MODEL_ENSAMBLE.
    """
    print(f"\n{Fore.YELLOW}Cargando modelos...{Style.RESET_ALL}")

    num_models = len(MODEL_ENSAMBLE)
    if num_models != len(TRAINED_MODEL_ENSAMBLE) or num_models != len(model_weights):
        raise ValueError("El número de modelos, modelos entrenados y pesos debe coincidir.")

    all_predictions = []
    test_reviews = pd.read_csv(predict_data, sep=';')
    reviews = test_reviews['text_tokens']
    true_labels = test_reviews['tags'].tolist()

    for i in range(num_models):
        print(f"Procesando modelo {i+1} de {num_models}")
        pipeline = ABSAPipeline(MODEL_ENSAMBLE[i])
        pipeline.aspect_model.load_model(pipeline.aspect_model.model, TRAINED_MODEL_ENSAMBLE[i])
        model_predictions = []
        for rev in tqdm(reviews):
            tokens, aspects, prob_asp = pipeline.predict_aspect(rev)
            model_predictions.append(predicted_bitmask(eval(rev), aspects))
        all_predictions.append(model_predictions)

    predicted_labels_final = []
    for i in range(len(reviews)): # Iterar sobre cada revisión
        min_len = min(len(pred[i]) for pred in all_predictions)
        predicciones = [pred[i][:min_len] for pred in all_predictions]
        
        # Promedio ponderado
        weighted_average = np.average(predicciones, axis=0, weights=model_weights)
        prediccion_final = np.where(weighted_average >= 0.5, 1, 0)
        predicted_labels_final.append(prediccion_final)

    print(f"\n{Fore.CYAN}Métricas:{Style.RESET_ALL}")
    metrics(true_labels, predicted_labels_final)


def predicted_bitmask(rev, aspects):
    # ... (Función original sin cambios) ...
    binary_list = []
    for palabra in rev:
        if palabra.lower() in aspects:
            binary_list.append(1)
        else:
            binary_list.append(0)
    return binary_list

