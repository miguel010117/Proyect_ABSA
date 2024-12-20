import pandas as pd
from Aspect_extraction.absapipeline import ABSAPipeline,metrics,predicted_bitmask
from colorama import Fore, Style
from Ensamble.ensamble import *


if __name__ == '__main__':
 
#VARIABLES
    # Modelos base
    BETO = r'F:/MIGUEL/Estudio/Tesis/Analisis_de_sentimiento/Model/bert-base-spanish-uncased'
    BERT = r'F:/MIGUEL/Estudio/Tesis/Analisis_de_sentimiento/Model/bert-base-multilingual-uncased'
    BERTIN_BASE = r'F:/MIGUEL/Estudio/Tesis/Analisis_de_sentimiento/Model/bertin-roberta-base-spanish'
    BERTIN_LARGE = r'F:/MIGUEL/Estudio/Tesis/Analisis_de_sentimiento/Model/bertin-roberta-large-spanish'
    ALBERT_BASE = r'F:/MIGUEL/Estudio/Tesis/Analisis_de_sentimiento/Model/albert-base-spanish'
    ALBERT_LARGE = r'F:/MIGUEL/Estudio/Tesis/Analisis_de_sentimiento/Model/albert-large-spanish'
    ALBERT_XX_LARGE = r'F:/MIGUEL/Estudio/Tesis/Analisis_de_sentimiento/Model/albert-xx-large-spanish'
    ELECTRA_SMALL = r'F:/MIGUEL/Estudio/Tesis/Analisis_de_sentimiento/Model/electra-small-discriminator'
    ELECTRA_BASE = r'F:/MIGUEL/Estudio/Tesis/Analisis_de_sentimiento/Model/electra-base-discriminator'
    GPT_2 = r'F:/MIGUEL/Estudio/Tesis/Analisis_de_sentimiento/Model/GPT-2'

    #Modelos entrenados
    BETO_TRAIN = r'F:/MIGUEL/Estudio/Tesis/Proyecto_ABSA/Model/beto-base-spanish/bert-base-spanish_epoch_3.pkl'
    BERT_TRAIN = r'F:/MIGUEL/Estudio/Tesis/Proyecto_ABSA/Model/bert-base-multilingual/bert-base-multilingual_epoch_2.pkl'
    BERTIN_BASE_TRAIN = r'F:/MIGUEL/Estudio/Tesis/Proyecto_ABSA/Model/bertin_base_spanish/bertin_base_spanish_epoch_2.pkl'
    BERTIN_LARGE_TRAIN = r'F:/MIGUEL/Estudio/Tesis/Proyecto_ABSA/Model/bertin_large_spanish/bertin_large_spanish_epoch_5.pkl'
    ALBERT_BASE_TRAIN = r'F:/MIGUEL/Estudio/Tesis/Proyecto_ABSA/Model/albert-base-spanish/albert-base-spanish_epoch_5.pkl'
    ALBERT_LARGE_TRAIN = r'F:/MIGUEL/Estudio/Tesis/Proyecto_ABSA/Model/albert_large_spanish/albert_large_spanish_epoch_4.pkl'
    ALBERT_XX_LARGE_TRAIN = r'F:/MIGUEL/Estudio/Tesis/Proyecto_ABSA/Model/albert_large_xx_spanish/albert_xx_large_spanish_epoch_3.pkl'
    ELECTRA_BASE_TRAIN = r'F:/MIGUEL/Estudio/Tesis/Proyecto_ABSA/Model/electra_base_spanish/electra_base_spanish_epoch_2.pkl'
    GPT_2_TRAIN = r'F:/MIGUEL/Estudio/Tesis/Proyecto_ABSA/electra_base_spanish_epoch_1.pkl'

    #Modelos pesos
    BETO_weighted = 0.85
    BERT_weighted =	0.84
    ALBERT_base_weighted = 0.84
    ALBERT_large_weighted =	0.86
    ALBERT_xx_large_weighted = 0.86
    BERTIN_base_weighted = 0.80
    BERTIN_large_weighted =	0.82
    GPT_2_weighted = 0.54
    ELECTRA_small_weighted = 0.80
    ELECTRA_base_weighted =	0.82

    #weighted = cal_weighted(0.50,0.20,0.10,0.10,0.10)

    #Modelo para entrenar
    modelo = BETO
    #Modelo entrenado 
    trained_model = BETO_TRAIN
    #Datos de entrenamiento
    train_data= "Data/dataset_train_without_duplicates.csv"
    #Datos para predecir
    predict_data= "Data/dataset_test_without_duplicates.csv"

    #Lista de modelos para ensamble
    MODEL_ENSAMBLE = [BETO,ALBERT_LARGE,ALBERT_XX_LARGE]

    #Lista de modelos entrenados para ensamble
    TRAINED_MODEL_ENSABMLE = [BETO_TRAIN,ALBERT_LARGE_TRAIN,ALBERT_XX_LARGE_TRAIN]
    
    
#FUNCIONES
    def presentacion():
        print(Style.BRIGHT + Fore.BLUE + "╔══════════════════════════════════════════════╗")
        print("║ " + Fore.WHITE + "    Bienvenido al Programa de Extracción    " + Fore.BLUE + " ║")
        print("║ " + Fore.WHITE + "            de Aspectos en ABSA             " + Fore.BLUE + " ║")
        print("╚══════════════════════════════════════════════╝" + Style.RESET_ALL)

    def menu_principal():
        print("\nPor favor, selecciona una opción:")
        print("  " + Fore.BLUE + "P" + Style.RESET_ALL + " - Para predecir aspectos.")
        print("  " + Fore.BLUE + "E" + Style.RESET_ALL + " - Para entrenar al modelo.")
        print("  " + Fore.BLUE + "S" + Style.RESET_ALL + " - Para predecir con ensamble.")
        return input("Ingrese su opción: ").strip().lower()

    def entrenamiento():
        print("\n" + Fore.YELLOW + "Inicializando pesos de los modelos pre-entrenados..." + 
        \
        Style.RESET_ALL)
        pipeline = ABSAPipeline(modelo)
        print(Fore.YELLOW + "Comenzando entrenamiento de la extracción de aspectos..." +
        \
         Style.RESET_ALL)
        pipeline.train_aspect_model(train_data,predict_data, 
                                    modelo, batch_size=8, 
                                    num_epochs=5)
        print(Fore.YELLOW + "¡Entrenamiento completado con éxito!" + Style.RESET_ALL)

    def predecir():
        print("\n" + Fore.YELLOW + "Cargando modelos..." + Style.RESET_ALL)
        pipeline = ABSAPipeline(modelo)
        pipeline.aspect_model.load_model(pipeline.aspect_model.model, trained_model)
        test_reviews = pd.read_csv(predict_data, sep=';')
        reviews = test_reviews['text_tokens']
        true_labels=test_reviews['tags'].tolist()
        predicted_labels = []

        for rev in reviews:
            tokens, aspects = pipeline.predict_aspect(rev)
            
            print("\n" + Fore.GREEN + "Review: " + Style.RESET_ALL, rev)
            print(Fore.GREEN + "Lista de aspectos: " + Style.RESET_ALL, aspects)
            
            predicted_labels.append(predicted_bitmask(eval(rev), aspects))
            

        print("\n" + Fore.CYAN + "Métricas:" + Style.RESET_ALL)
        metrics(true_labels, predicted_labels)



#MAIN
    presentacion()
    opcion = menu_principal()   
    if opcion == 'p':
        predecir()
    elif opcion == 'e':
        entrenamiento()
    elif opcion == 's':
        list_model = [ "BETO", "ALBERT_BASE", "ALBERT_BASE"]
        ensamble_weighted_average(MODEL_ENSAMBLE,TRAINED_MODEL_ENSABMLE,
                                  predict_data,list_model, [0.20, 0.40, 0.40])
    else:
        print(Fore.RED + "Opción no válida. Por favor, selecciona 'P' , 'E' o 'S'." + Style.RESET_ALL)
 