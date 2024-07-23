from torch.utils.data import DataLoader
from colorama import Fore, Style
from aspect_extraction import AspectTermExtraction, DatasetLoaderAspect
import pandas as pd
from sklearn.metrics import classification_report
from feature_extraction import ExtractFeauresByWindows
from prettytable import PrettyTable
from colorama import Fore, Style
import re


class ABSAPipeline:
    def __init__(self, aspect_model):
        self.aspect_model = AspectTermExtraction(model=aspect_model, tokenizer=aspect_model, seed=50)

    def train_aspect_model(self, train_file, model_type, batch_size, num_epochs):
        train_ds_aspect = DatasetLoaderAspect(pd.read_csv(train_file, sep=';'), model_type)
        aspect_train_loader = DataLoader(train_ds_aspect, batch_size=batch_size,
                                         collate_fn=self.aspect_model.create_mini_batch)
        self.aspect_model.train(aspect_train_loader, num_epochs)

    def predict_aspect(self, text):
        max_length = 512  # Longitud máxima de la secuencia admitida por BERT
        if len(text) > max_length:
            text = text[:max_length]  
        return self.aspect_model.predict(text, self.aspect_model.model)

if __name__ == '__main__':

#VARIABLES
    BETO = r'F:/MIGUEL/Estudio/Tesis/Sentiment_Analisis/Model/bert-base-spanish-uncased'
    modelo = BETO
    predict_data= "Data/dataset_test_without_duplicates.csv"
    train_data= "Data/dataset_train_without_duplicates.csv"
    trained_model=r'F:/MIGUEL/Estudio/Programación/VSC_Proyects/Proyectos/Proyecto_sofia/bert_ATE_spanish_epoch_3.pkl'

#FUNCIONES
    def presentacion():
        print(Style.BRIGHT + Fore.BLUE + "╔══════════════════════════════════════════════╗")
        print("║ " + Fore.WHITE + "    Bienvenido al Programa de Extracción    " + Fore.BLUE + " ║")
        print("║ " + Fore.WHITE + "            de Aspectos en ABSA             " + Fore.BLUE + " ║")
        print("╚══════════════════════════════════════════════╝" + Style.RESET_ALL)

    def menu_principal():
        print("\nPor favor, selecciona una opción:")
        print("  " + Fore.BLUE + "O" + Style.RESET_ALL + " - Para predecir los aspectos de una oracion.")
        print("  " + Fore.BLUE + "P" + Style.RESET_ALL + " - Para predecir aspectos.")
        print("  " + Fore.BLUE + "E" + Style.RESET_ALL + " - Para entrenar al modelo.")
        return input("Ingrese su opción: ").strip().lower()

    def entrenamiento():
        print("\n" + Fore.YELLOW + "Inicializando pesos de los modelos pre-entrenados..." + Style.RESET_ALL)
        pipeline = ABSAPipeline(modelo)
        print(Fore.YELLOW + "Comenzando entrenamiento de la extracción de aspectos..." + Style.RESET_ALL)
        pipeline.train_aspect_model(train_data, modelo, batch_size=8, num_epochs=3)
        print(Fore.YELLOW + "¡Entrenamiento completado con éxito!" + Style.RESET_ALL)

    def predecir():
        print("\n" + Fore.YELLOW + "Cargando modelos..." + Style.RESET_ALL)
        pipeline = ABSAPipeline(modelo)
        pipeline.aspect_model.load_model(pipeline.aspect_model.model, trained_model)
        windows_creator = ExtractFeauresByWindows(predict_data)
        test_reviews = pd.read_csv(predict_data, sep=';')
        reviews = test_reviews['text_tokens'][:1].to_list()
        true_labels=test_reviews['tags'][:1].tolist()
        aspectos = {}
        predicted_labels = []
        tabla_resultados = PrettyTable()
        tabla_resultados.field_names = ["Aspecto", "Parte de la reseña donde aparece"]

        for rev in reviews:
            tokens, aspects = pipeline.predict_aspect(rev)
            print("\n" + Fore.GREEN + "Review: " + Style.RESET_ALL, rev)
            print(Fore.GREEN + "Lista de aspectos: " + Style.RESET_ALL, aspects)
            
            predicted_labels.append(predicted_bitmask(rev, aspects))
            
            for aspect in aspects:
                aspect_cleaned = windows_creator.clean_text(aspect)
                part_of_review = windows_creator.create_new_sentence(windows_creator.clean_text(rev), aspect_cleaned.strip(), 5)
                
                if part_of_review is not None:
                    print(Fore.GREEN + 'Aspecto: '+ Style.RESET_ALL, aspect, 
                        Fore.GREEN + ' Parte de la reseña donde aparece: '+ Style.RESET_ALL, part_of_review)
                    
                    if aspect not in aspectos:
                        aspectos[aspect] = {'total': 1}
                    else:
                        aspectos[aspect]['total'] += 1
                    tabla_resultados.add_row([aspect, part_of_review])
                    
        print("\n" + Fore.CYAN + "Resultados:" + Style.RESET_ALL)
        print(tabla_resultados)
        print("\n" + Fore.CYAN + "Métricas:" + Style.RESET_ALL)
        print(true_labels)
        print(predicted_labels)
        metrics(true_labels, predicted_labels)

    def predecir_oracion(sentence):
            print("\n" + Fore.YELLOW + "Cargando modelos..." + Style.RESET_ALL)
            pipeline = ABSAPipeline(modelo)
            pipeline.aspect_model.load_model(pipeline.aspect_model.model, trained_model)
            tokens, aspects =pipeline.predict_aspect(sentence)
            print("\n" + Fore.GREEN + "Review: " + Style.RESET_ALL, sentence)
            print(Fore.GREEN + "Lista de aspectos: " + Style.RESET_ALL, aspects)

            return aspects

    def predicted_bitmask(rev,aspects):
            binary_list = []
            tokens = re.findall(r'\w+|[^\w\s]', rev)
            for palabra in tokens:
                if(palabra.lower() in aspects):
                    binary_list.append(1)
                else:
                    binary_list.append(0)
            return binary_list
          
    def metrics(true_labels, predicted_labels):
        true = []
        predicted = []
        for string in true_labels:
            tags = string[1:-1].split(',')
            for value in tags:
                true.append(int(value))
        for string in predicted_labels:
            for s in string:
                predicted.append(s)
        print(classification_report(true, predicted, target_names=[str(i) for i in range(2)]))

#MAIN
    presentacion()
    opcion = menu_principal()   
    if opcion == 'p':
        predecir()
    elif opcion == 'o':
        sentence= input("Ingrese la oracion: ")
        predecir_oracion(sentence)
    elif opcion == 'e':
        entrenamiento()
    else:
        print(Fore.RED + "Opción no válida. Por favor, selecciona 'P' o 'E'." + Style.RESET_ALL)
 