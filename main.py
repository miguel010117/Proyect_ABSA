import pandas as pd
from Aspect_extraction.absapipeline import ABSAPipeline,metrics,predicted_bitmask
from colorama import Fore, Style
from Ensamble.ensamble import *
from Logic.train_polarity import *


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


    BETO_TRAIN = r'F:/MIGUEL/Estudio/Tesis/Proyecto_ABSA/Model/2. Aspect/beto-base-spanish/bert-base-spanish_epoch_3.pkl'
    BERT_TRAIN = r'F:/MIGUEL/Estudio/Tesis/Proyecto_ABSA/Model/2. Aspect/bert-base-multilingual/bert-base-multilingual_epoch_2.pkl'
    BERTIN_BASE_TRAIN = r'F:/MIGUEL/Estudio/Tesis/Proyecto_ABSA/Model/2. Aspect/bertin_base_spanish/bertin_base_spanish_epoch_2.pkl'
    BERTIN_LARGE_TRAIN = r'F:/MIGUEL/Estudio/Tesis/Proyecto_ABSA/Model/2. Aspect/bertin_large_spanish/bertin_large_spanish_epoch_5.pkl'
    ALBERT_BASE_TRAIN = r'F:/MIGUEL/Estudio/Tesis/Proyecto_ABSA/Model/2. Aspect/albert-base-spanish/albert-base-spanish_epoch_5.pkl'
    ALBERT_LARGE_TRAIN = r'F:/MIGUEL/Estudio/Tesis/Proyecto_ABSA/Model/2. Aspect/albert_large_spanish/albert_large_spanish_epoch_4.pkl'
    ALBERT_XX_LARGE_TRAIN = r'F:/MIGUEL/Estudio/Tesis/Proyecto_ABSA/Model/2. Aspect/albert_large_xx_spanish/albert_xx_large_spanish_epoch_3.pkl'
    ELECTRA_SMALL_TRAIN = r'F:/MIGUEL/Estudio/Tesis/Proyecto_ABSA/Model/2. Aspect/electra_small_spanish/electra_small_spanish_epoch_4.pkl'
    ELECTRA_BASE_TRAIN = r'F:/MIGUEL/Estudio/Tesis/Proyecto_ABSA/Model/2. Aspect/electra_base_spanish/electra_base_spanish_epoch_2.pkl'
    GPT_2_TRAIN = r'F:/MIGUEL/Estudio/Tesis/Proyecto_ABSA/Model/2. Aspect/GPT_2/gpt_2_epoch_5.pkl'

    #Modelos entrenados
    model_base = {
    1: BETO,
    2: BERT,
    3: ALBERT_BASE,
    4: BERTIN_BASE,
    5: ELECTRA_BASE
    }

    #Modelos entrenados
    model_train = {
    1: BETO_TRAIN,
    2: BERT_TRAIN,
    3: ALBERT_BASE_TRAIN,
    4: BERTIN_BASE_TRAIN,
    5: ELECTRA_BASE_TRAIN
}
    
    # Datos para entrenar
    train_data_aspect = "Data/SemEval_Train_Aspect.csv"
    # Datos para entrenar
    train_data_polarity = "Data/SemEval_Train_Polarity.csv"
    # Datos para predecir
    predict_data_aspect = "Data/SemEval_Test_Aspect2.csv"
    # Datos para predecir
    predict_data_polarity = "Data/SemEval_Train_Polarity.csv"


#FUNCIONES
    def presentacion():
        print(Style.BRIGHT + Fore.BLUE + "╔══════════════════════════════════════════════╗")
        print("║ " + Fore.WHITE + "    Bienvenido al Programa de Extracción    " + Fore.BLUE + " ║")
        print("║ " + Fore.WHITE + "            de Aspectos en ABSA             " + Fore.BLUE + " ║")
        print("╚══════════════════════════════════════════════╝" + Style.RESET_ALL)

    def menu_principal():
        print("\nPor favor, selecciona una opción:")
        print("  " + Fore.BLUE + "1" + Style.RESET_ALL + " - Entrenar Modelos")
        print("  " + Fore.BLUE + "2" + Style.RESET_ALL + " - Predecir Aspectos")
        print("  " + Fore.BLUE + "3" + Style.RESET_ALL + " - Clasificar Polaridad")
        return input("Ingrese su opción: ").strip().lower()
    
    def entrenamiento_aspect(modelo,train_data,predict_data,num_epochs, batch):
        print("\n" + Fore.YELLOW + "Inicializando pesos de los modelos pre-entrenados..." + Style.RESET_ALL)
        pipeline = ABSAPipeline(modelo)
        print(Fore.YELLOW + "Comenzando entrenamiento de la extracción de aspectos..." + Style.RESET_ALL)
        pipeline.train_aspect_model(train_data,predict_data, 
                                    modelo, batch_size=batch, 
                                    num_epochs=num_epochs)
        print(Fore.YELLOW + "¡Entrenamiento completado con éxito!" + Style.RESET_ALL)

    def entrenamiento_polarity(modelo,train_data,num_epochs, batch):
        print("\n" + Fore.YELLOW + "Inicializando pesos de los modelos pre-entrenados..." + Style.RESET_ALL)
        print(Fore.YELLOW + "Comenzando entrenamiento de la clasificacion de polaridad..." + Style.RESET_ALL)
        train_polarity(modelo,train_data,num_epochs, batch)

    def predecir_aspect(modelo,trained_model,predict_data):
        print("\n" + Fore.YELLOW + "Cargando modelos..." + Style.RESET_ALL)
        pipeline = ABSAPipeline(modelo)
        pipeline.aspect_model.load_model(pipeline.aspect_model.model, trained_model)
        test_reviews = pd.read_csv(predict_data, sep=';')
        reviews = test_reviews['text_tokens']
        true_labels=test_reviews['tags'].tolist()
        predicted_labels = []
        name = ""

        for rev in reviews:
            tokens, aspects, prob = pipeline.predict_aspect(rev,name)
            
            print("\n" + Fore.GREEN + "Review: " + Style.RESET_ALL, rev)
            print(Fore.GREEN + "Lista de aspectos: " + Style.RESET_ALL, aspects)
            
            predicted_labels.append(predicted_bitmask(eval(rev), aspects))

        print("\n" + Fore.CYAN + "Métricas:" + Style.RESET_ALL)
        metrics(true_labels, predicted_labels)


#MAIN
    presentacion()

    opcion = menu_principal() 
    if opcion == '1':
        print("\nPor favor, selecciona el tipo de modelo:")
        print("  " + Fore.BLUE + "1" + Style.RESET_ALL + " - Aspecto")
        print("  " + Fore.BLUE + "2" + Style.RESET_ALL + " - Polarida")
        valor = input("Ingrese su opción: ").strip().lower()

        if valor == "1":
            print("\nPor favor, selecciona el modelo a entrenar:")
            print("  " + Fore.BLUE + "1" + Style.RESET_ALL + " - Beto")
            print("  " + Fore.BLUE + "2" + Style.RESET_ALL + " - Bert")
            print("  " + Fore.BLUE + "3" + Style.RESET_ALL + " - Albert_base")
            print("  " + Fore.BLUE + "4" + Style.RESET_ALL + " - Bertin_base")
            print("  " + Fore.BLUE + "5" + Style.RESET_ALL + " - Electra_base")
            num_modelo = input("Ingrese su opción: ").strip().lower()

            print("\nPor favor, selecciona el batch deseado")
            batch = int(input("Ingrese el valor del batch: ").strip().lower())

            print("\nPor favor, selecciona el numero de epochs")
            epochs = int(input("Ingrese el valor del epochs: ").strip().lower())

            if num_modelo == "1":
                entrenamiento_aspect(model_base[1],train_data_aspect,predict_data_aspect,epochs, batch)

            elif num_modelo == "2":
                entrenamiento_aspect(model_base[2],train_data_aspect,predict_data_aspect,epochs, batch)

            elif num_modelo == "3":
                entrenamiento_aspect(model_base[3],train_data_aspect,predict_data_aspect,epochs, batch)

            elif num_modelo == "4":
                entrenamiento_aspect(model_base[4],train_data_aspect,predict_data_aspect,epochs, batch)

            elif num_modelo == "5":
                entrenamiento_aspect(model_base[5],train_data_aspect,predict_data_aspect,epochs, batch)

        if valor == "2":
            print("\nPor favor, selecciona el modelo a entrenar:")
            print("  " + Fore.BLUE + "1" + Style.RESET_ALL + " - Beto")
            print("  " + Fore.BLUE + "2" + Style.RESET_ALL + " - Bert")
            print("  " + Fore.BLUE + "3" + Style.RESET_ALL + " - Albert_base")
            print("  " + Fore.BLUE + "4" + Style.RESET_ALL + " - Bertin_base")
            print("  " + Fore.BLUE + "5" + Style.RESET_ALL + " - Electra_base")
            num_modelo = input("Ingrese su opción: ").strip().lower()

            print("\nPor favor, selecciona el batch deseado")
            batch = int(input("Ingrese el valor del batch: ").strip().lower())

            print("\nPor favor, selecciona el numero de epochs")
            epochs = int(input("Ingrese el valor del epochs: ").strip().lower())

            if num_modelo == "1":
                entrenamiento_polarity(model_base[1],train_data_polarity,epochs, batch)

            elif num_modelo == "2":
                entrenamiento_polarity(model_base[2],train_data_polarity,epochs, batch)

            elif num_modelo == "3":
                entrenamiento_polarity(model_base[3],train_data_polarity,epochs, batch)

            elif num_modelo == "4":
                entrenamiento_polarity(model_base[4],train_data_polarity,epochs, batch)

            elif num_modelo == "5":
                entrenamiento_polarity(model_base[5],train_data_polarity,epochs, batch)



    elif opcion == '2':
        print("\nPor favor, selecciona el metodo de clasificacion de aspecto")
        print("  " + Fore.BLUE + "1" + Style.RESET_ALL + " - Individual")
        print("  " + Fore.BLUE + "2" + Style.RESET_ALL + " - Ensamble por votacion maxima")
        valor = input("Ingrese su opción: ").strip().lower()

        if valor == "1":
            print("\nPor favor, selecciona el modelo para clasificar:")
            print("  " + Fore.BLUE + "1" + Style.RESET_ALL + " - Beto")
            print("  " + Fore.BLUE + "2" + Style.RESET_ALL + " - Bert")
            print("  " + Fore.BLUE + "3" + Style.RESET_ALL + " - Albert_base")
            print("  " + Fore.BLUE + "4" + Style.RESET_ALL + " - Bertin_base")
            print("  " + Fore.BLUE + "5" + Style.RESET_ALL + " - Electra_base")
            num_modelo = input("Ingrese su opción: ").strip().lower()

            if num_modelo == "1":
                predecir_aspect(model_base[1],model_train[1],predict_data_aspect)

            elif num_modelo == "2":
                predecir_aspect(model_base[2],model_train[2],predict_data_aspect)

            elif num_modelo == "3":
                predecir_aspect(model_base[3],model_train[3],predict_data_aspect)

            elif num_modelo == "4":
                predecir_aspect(model_base[4],model_train[4],predict_data_aspect)

            elif num_modelo == "5":
                predecir_aspect(model_base[5],model_train[5],predict_data_aspect)


        if valor == "2":

            MODEL_ENSAMBLE = []
            TRAINED_MODEL_ENSABMLE = []
            list_model = []

            print("\nPor favor, selecciona los modelos para combinar en el ensamble (introduce los números separados por comas)")
            print("  " + Fore.BLUE + "1" + Style.RESET_ALL + " - Beto")
            print("  " + Fore.BLUE + "2" + Style.RESET_ALL + " - Bert")
            print("  " + Fore.BLUE + "3" + Style.RESET_ALL + " - Albert_base")
            print("  " + Fore.BLUE + "4" + Style.RESET_ALL + " - Bertin_base")
            print("  " + Fore.BLUE + "5" + Style.RESET_ALL + " - Electra_base")
            entrada = input("Modelos seleccionados: ").strip().lower()
            numeros = [int(x.strip()) for x in entrada.split(',')]

            for num in numeros:
                if 1 <= num <= 5:
                    # #Lista de modelos para ensamble
                    MODEL_ENSAMBLE.append(model_base[num])
                    # #Lista de modelos entrenados para ensamble
                    TRAINED_MODEL_ENSABMLE.append(model_train[num])
                    #Lista de modelos
                    list_model.append("")
                else:
                    print(f"Número {num} inválido. Debe estar entre 1 y 5.")

            ensamble_max(MODEL_ENSAMBLE,TRAINED_MODEL_ENSABMLE,predict_data_aspect,list_model)
   







    elif opcion == '3':
        print("\nPor favor, selecciona el modelo")
        print("  " + Fore.BLUE + "1" + Style.RESET_ALL + " - Beto")
        print("  " + Fore.BLUE + "2" + Style.RESET_ALL + " - Bert")
        valor = input("Ingrese su opción: ").strip().lower()