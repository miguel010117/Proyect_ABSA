from colorama import Fore, Style
from Aspect_extraction.absapipeline import ABSAPipeline
import pandas as pd
from tqdm import tqdm
import numpy as np
from Aspect_extraction.absapipeline import metrics,predicted_bitmask

def ensamble(MODEL_ENSAMBLE,TRAINED_MODEL_ENSABMLE,predict_data):
        print("\n" + Fore.YELLOW + "Cargando modelos..." + Style.RESET_ALL)

        modelos = MODEL_ENSAMBLE

        trained_models = TRAINED_MODEL_ENSABMLE

        pred_labels_model_1 = []
        pred_labels_model_2 = []
        pred_labels_model_3 = []
        pred_labels_model_4 = []
        pred_labels_model_5 = []
        pred_labels_model_6 = []
        pred_labels_model_7 = []

        predicted_labels_final = []

        for i in range(len(modelos)):
            print(i)
            pipeline = ABSAPipeline(modelos[i])
            pipeline.aspect_model.load_model(pipeline.aspect_model.model, trained_models[i])
            test_reviews = pd.read_csv(predict_data, sep=';')
            reviews = test_reviews['text_tokens']
            true_labels=test_reviews['tags'].tolist()
            predicted_labels = []
      
            for rev in tqdm(reviews):
                tokens, aspects = pipeline.predict_aspect(rev)
                predicted_labels.append(predicted_bitmask(eval(rev), aspects))

            if i == 0:
                pred_labels_model_1 = predicted_labels
            elif i == 1:
                pred_labels_model_2 = predicted_labels
            elif i == 2:
                pred_labels_model_3 = predicted_labels
            elif i == 3:
                pred_labels_model_4 = predicted_labels
            elif i == 4:
                pred_labels_model_5 = predicted_labels
            elif i == 5:
                pred_labels_model_6 = predicted_labels
            else:
                pred_labels_model_7 = predicted_labels
               
        
        for i in range(len(pred_labels_model_1)):
            
            predicciones = []
            predicciones.append(pred_labels_model_1[i])
            predicciones.append(pred_labels_model_2[i])
            predicciones.append(pred_labels_model_3[i])
            predicciones.append(pred_labels_model_4[i])
            # predicciones.append(pred_labels_model_5[i])
            # predicciones.append(pred_labels_model_6[i])
            # predicciones.append(pred_labels_model_7[i])
            # print(predicciones)
            prediccion_final = np.average(predicciones, axis=0)

            # Convertir a valores binarios (0 o 1)
            prediccion_final = np.where(prediccion_final >= 0.5, 1, 0)

            # print("Predicción final:", prediccion_final)
            predicted_labels_final.append(prediccion_final)

        print("\n" + Fore.CYAN + "Métricas:" + Style.RESET_ALL)
        metrics(true_labels, predicted_labels_final)