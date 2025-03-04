import pandas as pd
from Aspect_extraction.absapipeline import ABSAPipeline,metrics,predicted_bitmask
from colorama import Fore, Style
from Ensamble.ensamble import *
from Logic.train_polarity import *
from Test.test_polarity import *


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

    # Modelos entrenados para extraccion de aspectos
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

    # Modelos entrenados para clasificacion de polaridad
    BETO_TRAIN_POLARITY = r'F:/MIGUEL/Estudio/Tesis/Proyecto_ABSA/Model/3. Polarity/Beto_Fine_turned/'
    BERT_TRAIN_POLARITY = r'F:/MIGUEL/Estudio/Tesis/Proyecto_ABSA/Model/3. Polarity/Bert_Fine_turned/'
    BERTIN_BASE_TRAIN_POLARITY = r'F:/MIGUEL/Estudio/Tesis/Proyecto_ABSA/Model/3. Polarity/Bertin_Base_Fine_turned/Bertin_Base.pth'
    BERTIN_LARGE_TRAIN_POLARITY = r'F:/MIGUEL/Estudio/Tesis/Proyecto_ABSA/Model/3. Polarity/Bertin_Large_Fine_turned/Bertin_Large.pth'
    ALBERT_BASE_TRAIN_POLARITY = r'F:/MIGUEL/Estudio/Tesis/Proyecto_ABSA/Model/3. Polarity/Albert_Base_Fine_turned/'
    ALBERT_LARGE_TRAIN_POLARITY = r'F:/MIGUEL/Estudio/Tesis/Proyecto_ABSA/Model/3. Polarity/'
    ALBERT_XX_LARGE_TRAIN_POLARITY = r'F:/MIGUEL/Estudio/Tesis/Proyecto_ABSA/Model/3. Polarity/'
    ELECTRA_SMALL_TRAIN_POLARITY= r'F:/MIGUEL/Estudio/Tesis/Proyecto_ABSA/Model/3. Polarity/Electra_Small_Fine_turned/Electra_Small.pth'
    ELECTRA_BASE_TRAIN_POLARITY = r'F:/MIGUEL/Estudio/Tesis/Proyecto_ABSA/Model/3. Polarity/Electra_Base_Fine_turned/Electra_Base.pth'
    GPT_2_TRAIN = r'F:/MIGUEL/Estudio/Tesis/Proyecto_ABSA/Model/GPT_2_Fine_turned/GPT_2.pth'

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
    
    #Modelos entrenados
    model_train_polarity = {
    1: BETO_TRAIN_POLARITY,
    2: BERT_TRAIN_POLARITY,
    3: ALBERT_BASE_TRAIN_POLARITY,
    4: BERTIN_BASE_TRAIN_POLARITY,
    5: ELECTRA_BASE_TRAIN_POLARITY
}
    
    # Datos para entrenar aspectos
    train_data_aspect = "Data/SemEval_Train_Aspect.csv"
    # Datos para entrenar polaridad
    train_data_polarity = "Data/SemEval_Train_Polarity.csv"
    # Datos para predecir aspectos
    predict_data_aspect = "Data/SemEval_Test_Aspect2.csv"
    # Datos para predecir polaridad
    predict_data_polarity_ventana = "Data/SemEval_Test_Polarity_Ventana.csv"
    predict_data_polarity_dep = "Data/SemEval_Test_Polarity_Dep.csv"
    predict_data_polarity_hibrido = "Data/SemEval_Test_Polarity_Hibrido.csv"

    predict_data_polarity = {
       1: predict_data_polarity_ventana,
       2: predict_data_polarity_dep,
       3: predict_data_polarity_hibrido
    }

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

    # ENTRENAMIENTO MODELOS ******************************************************************************************** 
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


    # CLASIFICACION DE ASPECTOS **********************************************************************************************
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
                    MODEL_ENSAMBLE.append(model_base[num])
                    TRAINED_MODEL_ENSABMLE.append(model_train[num])
                    list_model.append("")
                else:
                    print(f"Número {num} inválido. Debe estar entre 1 y 5.")

            ensamble_max(MODEL_ENSAMBLE,TRAINED_MODEL_ENSABMLE,predict_data_aspect,list_model)
   
    # CLASIFICACION DE POLARIDAD ********************************************************************************************
    elif opcion == '3':
        print("\nPor favor, selecciona el modelo para clasificar:")
        print("  " + Fore.BLUE + "1" + Style.RESET_ALL + " - Beto")
        print("  " + Fore.BLUE + "2" + Style.RESET_ALL + " - Bert")
        print("  " + Fore.BLUE + "3" + Style.RESET_ALL + " - Albert_base")
        print("  " + Fore.BLUE + "4" + Style.RESET_ALL + " - Bertin_base")
        print("  " + Fore.BLUE + "5" + Style.RESET_ALL + " - Electra_base")
        num_modelo = input("Ingrese su opción: ").strip().lower()

        print("\nPor favor, selecciona el metodo de seleccion de caracteristicas:")
        print("  " + Fore.BLUE + "1" + Style.RESET_ALL + " - Ventana de palabras")
        print("  " + Fore.BLUE + "2" + Style.RESET_ALL + " - Analisis de dependencias")
        print("  " + Fore.BLUE + "3" + Style.RESET_ALL + " - Enfoque hibrido")
        metodo = int(input("Ingrese su opción: ").strip().lower())

        if num_modelo == "1":
            predict_polarity(model_base[1],model_train_polarity[1],predict_data_polarity[metodo],1)

        elif num_modelo == "2":
            predict_polarity(model_base[2],model_train_polarity[2],predict_data_polarity[metodo],1)

        elif num_modelo == "3":
            predict_polarity(model_base[3],model_train_polarity[3],predict_data_polarity[metodo],1)

        elif num_modelo == "4":
            predict_polarity(model_base[4],model_train_polarity[4],predict_data_polarity[metodo],2)

        elif num_modelo == "5":
            predict_polarity(model_base[5],model_train_polarity[5],predict_data_polarity[metodo],2)


##############################################################################################################################

