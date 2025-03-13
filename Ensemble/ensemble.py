import ast
import numpy as np
import pandas as pd
from tqdm import tqdm
from colorama import Fore, Style
from sklearn.ensemble import AdaBoostClassifier 
from sklearn.metrics import classification_report
from sklearn.ensemble import RandomForestClassifier 
from Aspect_extraction.absapipeline import ABSAPipeline
from Aspect_extraction.absapipeline import metrics,predicted_bitmask
import unicodedata

def ensamble_max(MODEL_ENSAMBLE, TRAINED_MODEL_ENSAMBLE, predict_data, list_model):

    reviews, true_labels = cargar_datos(predict_data)

    # Lista para almacenar las predicciones de cada modelo
    model_predictions = []

    for i, (model, trained_model) in enumerate(zip(MODEL_ENSAMBLE, TRAINED_MODEL_ENSAMBLE)):
        name_model = list_model[i]
        print(f"Procesando modelo {i+1}...")
        pipeline = ABSAPipeline(model)
        pipeline.aspect_model.load_model(pipeline.aspect_model.model, trained_model)

        predicted_labels = []
        for rev in tqdm(reviews):
            tokens, aspects, prob_asp = pipeline.predict_aspect(rev,name_model)
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


def ensamble_average(MODEL_ENSAMBLE, TRAINED_MODEL_ENSAMBLE, predict_data, list_model):

    reviews, true_labels = cargar_datos(predict_data)

    all_predictions = procesar_probabilidades(MODEL_ENSAMBLE, TRAINED_MODEL_ENSAMBLE, list_model, reviews)

    predicted_labels_final = []
    for i in range(len(reviews)): # Iterar sobre cada revisión
        predicciones = []
        for pred in all_predictions:
            if len(pred[i]) >= len_elemnt(true_labels[i]):
                predicciones.append(pred[i][:len_elemnt(true_labels[i])])  # Recortar
            else:
                predicciones.append(np.pad(pred[i], (0, len_elemnt(true_labels[i]) - len(pred[i])), 'constant')) # Rellenar con ceros

        prediccion_final = np.mean(predicciones, axis=0)
        print(prediccion_final)
        prediccion_final = np.where(prediccion_final >= 0.500, 1, 0)
        predicted_labels_final.append(prediccion_final)
        
    
    print(f"\n{Fore.CYAN}Métricas:{Style.RESET_ALL}")
    metrics(true_labels, predicted_labels_final)


def ensamble_weighted_average(MODEL_ENSAMBLE, TRAINED_MODEL_ENSAMBLE, predict_data, list_model, model_weights):

    reviews, true_labels = cargar_datos(predict_data)

    all_predictions = procesar_probabilidades(MODEL_ENSAMBLE, TRAINED_MODEL_ENSAMBLE, list_model, reviews)

    predicted_labels_final = []
    for i in range(len(reviews)): # Iterar sobre cada revisión
        predicciones = []
        for pred in all_predictions:
            if len(pred[i]) >= len_elemnt(true_labels[i]):
                predicciones.append(pred[i][:len_elemnt(true_labels[i])])  # Recortar
            else:
                predicciones.append(np.pad(pred[i], (0, len_elemnt(true_labels[i]) - len(pred[i])), 'constant')) # Rellenar con ceros
        
        # Promedio ponderado
        weighted_average = np.average(predicciones, axis=0, weights=model_weights)
        prediccion_final = np.where(weighted_average >= 0.5, 1, 0)
        predicted_labels_final.append(prediccion_final)

    print(f"\n{Fore.CYAN}Métricas:{Style.RESET_ALL}")
    metrics(true_labels, predicted_labels_final)


def ensamble_boosting(MODEL_ENSAMBLE, TRAINED_MODEL_ENSAMBLE, train_data, predict_data, list_model):

    print(f"\n{Fore.YELLOW}Cargando modelos...{Style.RESET_ALL}")

    reviews_train, true_labels_train = cargar_datos_version2(train_data)
    reviews_test, true_labels_test = cargar_datos_version2(predict_data)

    # caracteristicas_boosting = np.concatenate([bert_array, albert_array, electra_array], axis=1)
    # Crea el ensamble AdaBoost usando las predicciones de los modelos base como entrada
    ada_boost = AdaBoostClassifier(
        n_estimators=50, # Número de estimadores (ajusta según sea necesario)
        learning_rate=1.0, # Tasa de aprendizaje (ajusta según sea necesario)
        algorithm='SAMME', # Algoritmo para clasificación multiclase
        random_state=42
    )


    #Obtener salida de cada modelo para datos de entrenamiento
    # Lista para almacenar las predicciones de cada modelo para train
    model_predictions_train = []

    for i, (model, trained_model) in enumerate(zip(MODEL_ENSAMBLE, TRAINED_MODEL_ENSAMBLE)):
        name_model = list_model[i]
        print(f"Procesando modelo {i+1}...")
        pipeline = ABSAPipeline(model)
        pipeline.aspect_model.load_model(pipeline.aspect_model.model, trained_model)

        predicted_labels = []
        for rev in tqdm(reviews_train):
            tokens, aspects, prob_asp = pipeline.predict_aspect(rev,name_model)
            predicted_labels.append(predicted_bitmask(eval(rev), aspects))

        model_predictions_train.append(predicted_labels)                                                 #model_predictions_train
    model_predictions_train = vectorizar_salidas(model_predictions_train)  
    print(model_predictions_train.shape)

    # Entrena AdaBoost con las predicciones de los modelos base
    ada_boost.fit(model_predictions_train, true_labels_train)

    #Obtener salida de cada modelo para datos de test
    # Lista para almacenar las predicciones de cada modelo para test
    model_predictions_test = []

    for i, (model, trained_model) in enumerate(zip(MODEL_ENSAMBLE, TRAINED_MODEL_ENSAMBLE)):
        name_model = list_model[i]
        print(f"Procesando modelo {i+1}...")
        pipeline = ABSAPipeline(model)
        pipeline.aspect_model.load_model(pipeline.aspect_model.model, trained_model)

        predicted_labels = []
        for rev in tqdm(reviews_test):
            tokens, aspects, prob_asp = pipeline.predict_aspect(rev,name_model)
            predicted_labels.append(predicted_bitmask(eval(rev), aspects))

        model_predictions_test.append(predicted_labels)                                                #model_predictions_test
    model_predictions_test = vectorizar_salidas(model_predictions_test)  
    print(model_predictions_test.shape)


    # Luego, usa AdaBoost para combinar las predicciones
    predicciones_boosting = ada_boost.predict(model_predictions_test)

    print(f"\n{Fore.CYAN}Métricas:{Style.RESET_ALL}")
    # metrics(true_labels_test, predicciones_boosting) # Asumiendo que metrics está definido en otro lugar
    print(classification_report(true_labels_test, predicciones_boosting, 
                                target_names=[str(i) for i in range(2)], 
                                digits=4))


def ensamble_staking(MODEL_ENSAMBLE, TRAINED_MODEL_ENSAMBLE, train_data, predict_data, list_model):

    print(f"\n{Fore.YELLOW}Cargando modelos...{Style.RESET_ALL}")

    reviews_train, true_labels_train = cargar_datos_version2(train_data)
    reviews_test, true_labels_test = cargar_datos_version2(predict_data)
    
    stacker = RandomForestClassifier(random_state=42)

    #Obtener salida de cada modelo para datos de entrenamiento
    # Lista para almacenar las predicciones de cada modelo para train
    model_predictions_train = []

    for i, (model, trained_model) in enumerate(zip(MODEL_ENSAMBLE, TRAINED_MODEL_ENSAMBLE)):
        name_model = list_model[i]
        print(f"Procesando modelo {i+1}...")
        pipeline = ABSAPipeline(model)
        pipeline.aspect_model.load_model(pipeline.aspect_model.model, trained_model)

        predicted_labels = []
        for rev in tqdm(reviews_train):
            tokens, aspects, prob_asp = pipeline.predict_aspect(rev,name_model)
            predicted_labels.append(predicted_bitmask(eval(rev), aspects))

        model_predictions_train.append(predicted_labels)                                                 #model_predictions_train
    model_predictions_train = vectorizar_salidas(model_predictions_train)  

    # Entrena stacker con las predicciones de los modelos base
    stacker.fit(model_predictions_train, true_labels_train)


    #Obtener salida de cada modelo para datos de test
    # Lista para almacenar las predicciones de cada modelo para test
    model_predictions_test = []

    for i, (model, trained_model) in enumerate(zip(MODEL_ENSAMBLE, TRAINED_MODEL_ENSAMBLE)):
        name_model = list_model[i]
        print(f"Procesando modelo {i+1}...")
        pipeline = ABSAPipeline(model)
        pipeline.aspect_model.load_model(pipeline.aspect_model.model, trained_model)

        predicted_labels = []
        for rev in tqdm(reviews_test):
            tokens, aspects, prob_asp = pipeline.predict_aspect(rev,name_model)
            predicted_labels.append(predicted_bitmask(eval(rev), aspects))

        model_predictions_test.append(predicted_labels)                                                #model_predictions_test
    model_predictions_test = vectorizar_salidas(model_predictions_test)  



    # Luego, usa AdaBoost para combinar las predicciones
    predicciones_boosting = stacker.predict(model_predictions_test)

    print(f"\n{Fore.CYAN}Métricas:{Style.RESET_ALL}")
    # metrics(true_labels_test, predicciones_boosting) # Asumiendo que metrics está definido en otro lugar
    print(classification_report(true_labels_test, predicciones_boosting, 
                                target_names=[str(i) for i in range(2)], 
                                digits=4))
    

#////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

def cargar_datos(predict_data):
    print(f"\n{Fore.YELLOW}Cargando modelos...{Style.RESET_ALL}")

    # Carga los datos
    test_reviews = pd.read_csv(predict_data, sep=';')
    reviews = test_reviews['text_tokens'].tolist()
    true_labels = test_reviews['tags'].tolist()

    return reviews, true_labels

def cargar_datos_version2(train_data):
    train_reviews = pd.read_csv(train_data, sep=';')
    reviews_train = train_reviews['text_tokens'].tolist()
    true_labels_train = convertir_a_lista_enteros(train_reviews['tags'].tolist())  

    return reviews_train, true_labels_train

def procesar_probabilidades(MODEL_ENSAMBLE, TRAINED_MODEL_ENSAMBLE, list_model, reviews):
    num_models = len(MODEL_ENSAMBLE)
    if num_models != len(TRAINED_MODEL_ENSAMBLE):
        raise ValueError("El número de modelos y modelos entrenados debe coincidir.")

    all_predictions = []

    for i in range(num_models):
        print(f"Procesando modelo {i+1} de {num_models}")
        pipeline = ABSAPipeline(MODEL_ENSAMBLE[i])
        name_model = list_model[i]
        pipeline.aspect_model.load_model(pipeline.aspect_model.model, TRAINED_MODEL_ENSAMBLE[i])
        model_predictions = []
        all_prob = []
        for rev in tqdm(reviews):
            tokens, aspects, prob_aspct = pipeline.predict_aspect(rev,name_model)
            all_prob.append(prob_aspct)
            print(prob_aspct)
            model_predictions.append(predicted_bitmask(eval(rev), aspects))
        all_predictions.append(all_prob)
    return all_predictions

def len_elemnt(elemnt):
    # Eliminar los corchetes y separar los elementos
    lista_str = elemnt[1:-1].split(',')
    # Convertir los elementos a enteros
    lista_int = [int(x.strip()) for x in lista_str]
    
    return len(lista_int)


def remover_tildes(texto):
  """Remueve las tildes de un texto."""
  return ''.join(c for c in unicodedata.normalize('NFD', texto)
                 if unicodedata.category(c) != 'Mn')

def predicted_bitmask(rev, aspects):
    binary_list = []
    for palabra in rev:
        if remover_tildes(palabra.lower()) in aspects:
            binary_list.append(1)
        else:
            binary_list.append(0)
    return binary_list

def cal_weighted(w1,w2,w3):
    sum = w1 + 15 + w2 + w3 - 15 

    w1 = w1 + 15 / sum
    w2 = w2 / sum
    w3 = w3 - 15 / sum

    return w1, w2, w3

def vectorizar_salidas(model_outputs):
    vectorized_outputs = []
    for outputs in model_outputs:
        vectorized_outputs.append([int(bit) for sublist in outputs for bit in sublist])
    return np.array(vectorized_outputs).T

def convertir_a_lista_enteros(lista_de_cadenas):

    resultado = []
    
    for cadena in lista_de_cadenas:
        # Convierte la cadena a una lista utilizando ast.literal_eval
        lista = ast.literal_eval(cadena)
        
        # Agrega los elementos de la lista a la lista de resultados
        resultado.extend(lista)

    return np.array(resultado).T




