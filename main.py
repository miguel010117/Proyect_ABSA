from pathlib import Path
from colorama import Fore, Style
from Aspect_extraction.absapipeline import ABSAPipeline,metrics,predicted_bitmask
from colorama import Fore, Style
from Ensamble.ensamble import *
from Logic.train_polarity import *
from Test.test_polarity import *

# ================= CONFIGURACIÓN =================
class Config:
    BASE_MODELS = {
        'beto': Path("Model/1. Base/bert-base-spanish-uncased/"),
        'bert': Path("Model/1. Base/bert-base-multilingual-uncased"),
        'albert_base': Path("Model/1. Base/albert-base-spanish"),
        'bertin_base': Path("Model/1. Base/bertin-roberta-base-spanish"),
        'electra_base': Path("Model/1. Base/electra-base-discriminator"),
        # Agrega más modelos aquí
    }
    
    TRAINED_MODELS = {
        'beto': {
            'aspect': Path("Model/2. Aspect/beto-base-spanish/bert-base-spanish_epoch_3.pkl"),
            'polarity': Path("Model/3. Polarity/Beto_Fine_turned/")
        },
        'bert': {
            'aspect': Path("Model/2. Aspect/bert-base-multilingual/bert-base-multilingual_epoch_2.pkl"),
            'polarity': Path("Model/3. Polarity/Bert_Fine_turned/")
        },
        'albert_base': {
            'aspect': Path("Model/2. Aspect/albert-base-spanish/albert-base-spanish_epoch_5.pkl"),
            'polarity': Path("Model/3. Polarity/Albert_Base_Fine_turned/")
        },
        'bertin_base': {
            'aspect': Path("Model/2. Aspect/bertin_base_spanish/bertin_base_spanish_epoch_2.pkl"),
            'polarity': Path("Model/3. Polarity/Bertin_Base_Fine_turned/Bertin_Base.pth")
        },
        'electra_base': {
            'aspect': Path("Model/2. Aspect/electra_base_spanish/electra_base_spanish_epoch_2.pkl"),
            'polarity': Path("Model/3. Polarity/Electra_Base_Fine_turned/Electra_Base.pth")
        },
    }
    
    DATA_PATHS = {
        'train': {
            'aspect': "Data/SemEval_Train_Aspect.csv",
            'polarity': "Data/SemEval_Train_Polarity.csv"
        },
        'test': {
            'aspect': "Data/SemEval_Test_Aspect.csv",
            'polarity': {
                'ventana': "Data/SemEval_Test_Polarity_Ventana.csv",
                'dep': "Data/SemEval_Test_Polarity_Dep.csv",
                'hibrido': "Data/SemEval_Test_Polarity_Hibrido.csv"
            }
        }
    }

# ================= FUNCIONES PRINCIPALES =================
def mostrar_menu():
    print(Fore.BLUE + "\n=== MENÚ PRINCIPAL ===" + Style.RESET_ALL)
    print("1. Entrenar modelo")
    print("2. Predecir aspectos")
    print("3. Clasificar polaridad")
    print("4. Salir")
    return input("Seleccione una opción: ").strip()

def train_model():
    print(Fore.YELLOW + "\n=== ENTRENAMIENTO ===" + Style.RESET_ALL)
    print("Seleccione el tipo de modelo:")
    print("1. Aspectos")
    print("2. Polaridad")
    tipo = input("Opción: ").strip()

    name_model = list(Config.BASE_MODELS.keys()) # Obtenemos una lista de los nombres de los modelos
    
    print("\nSeleccione el modelo:")
    for i, (nombre, _) in enumerate(Config.BASE_MODELS.items(), 1):
        print(f"{i}. {nombre.capitalize()}")
    modelo = int(input("Opción: ").strip())
  
    batch, epoch = obtener_parametros()

    if tipo == '1':
        train_aspect(Config.BASE_MODELS[name_model[modelo-1]],Config.DATA_PATHS['train']['aspect'],Config.DATA_PATHS['test']['aspect'],epoch, batch)

    elif tipo == '2':
        train_polarity(Config.BASE_MODELS[name_model[modelo-1]],Config.DATA_PATHS['train']['polarity'], epoch, batch)

    print(Fore.GREEN + "\nEntrenamiento completado!" + Style.RESET_ALL)

def predecir_aspectos():
    print(Fore.YELLOW + "\n=== PREDICCIÓN DE ASPECTOS ===" + Style.RESET_ALL)
    name_model = list(Config.BASE_MODELS.keys()) # Obtenemos una lista de los nombres de los modelos
    print("\nPor favor, selecciona el metodo de clasificacion de aspecto")
    print("  " + Fore.BLUE + "1" + Style.RESET_ALL + " - Individual")
    print("  " + Fore.BLUE + "2" + Style.RESET_ALL + " - Ensamble por votacion maxima")
    tipo = input("Ingrese su opción: ").strip().lower()

    if tipo == '1':
        print("\nSeleccione el modelo:")
        for i, (nombre, _) in enumerate(Config.TRAINED_MODELS.items(), 1):
            print(f"{i}. {nombre.capitalize()}")
        modelo = int(input("Opción: ").strip())
        
        predecir_aspect(Config.BASE_MODELS[name_model[modelo-1]],Config.TRAINED_MODELS[name_model[modelo-1]]['aspect'],Config.DATA_PATHS['test']['aspect'])

    elif tipo == '2':
        MODEL_ENSAMBLE = []
        TRAINED_MODEL_ENSABMLE = []
        list_model = []
        name_model = list(Config.BASE_MODELS.keys()) # Obtenemos una lista de los nombres de los modelos

        print("\nPor favor, selecciona los modelos para combinar en el ensamble (introduce los números separados por comas)")
        for i, (nombre, _) in enumerate(Config.TRAINED_MODELS.items(), 1):
            print(f"{i}. {nombre.capitalize()}")
        modelo = input("Opción: ").strip()
        numeros = [int(x.strip()) for x in modelo.split(',')]

        for num in numeros:
            if 1 <= num <= 5:
                MODEL_ENSAMBLE.append(Config.BASE_MODELS[name_model[num-1]])
                TRAINED_MODEL_ENSABMLE.append(Config.TRAINED_MODELS[name_model[num-1]]['aspect'])
                list_model.append("")
            else:
                print(f"Número {num} inválido. Debe estar entre 1 y 5.")

        ensamble_max(MODEL_ENSAMBLE,TRAINED_MODEL_ENSABMLE,Config.DATA_PATHS['test']['aspect'],list_model)
        
    print(Fore.GREEN + "\nPredicción completada!" + Style.RESET_ALL)

def clasificar_polaridad():
    print(Fore.YELLOW + "\n=== CLASIFICACIÓN DE POLARIDAD ===" + Style.RESET_ALL)
    print("Seleccione el método:")
    print("1. Ventana de palabras")
    print("2. Análisis de dependencias")
    print("3. Enfoque híbrido")
    metodo = int(input("Opción: ").strip())
    name_metodo = ['ventana', 'dep', 'hibrido']
    name_model = list(Config.BASE_MODELS.keys()) # Obtenemos una lista de los nombres de los modelos
    
    print("\nSeleccione el modelo:")
    for i, (nombre, _) in enumerate(Config.BASE_MODELS.items(), 1):
        print(f"{i}. {nombre.capitalize()}")
    modelo = int(input("Opción: ").strip())

    predict_polarity(Config.BASE_MODELS[name_model[modelo-1]],Config.TRAINED_MODELS[name_model[modelo-1]]['polarity'],Config.DATA_PATHS['test']['polarity'][name_metodo[metodo-1]],name_model[modelo-1])

    print(Fore.GREEN + "\nClasificación completada!" + Style.RESET_ALL)

def train_aspect(modelo,train_data,predict_data,num_epochs, batch):
    print("\n" + Fore.YELLOW + "Inicializando pesos de los modelos pre-entrenados..." + Style.RESET_ALL)
    pipeline = ABSAPipeline(modelo)
    print(Fore.YELLOW + "Comenzando entrenamiento de la extracción de aspectos..." + Style.RESET_ALL)
    pipeline.train_aspect_model(train_data,predict_data, 
                                modelo, batch_size=batch, 
                                num_epochs=num_epochs)
    print(Fore.YELLOW + "¡Entrenamiento completado con éxito!" + Style.RESET_ALL)

def train_polarity(modelo,train_data,num_epochs, batch):
    print("\n" + Fore.YELLOW + "Inicializando pesos de los modelos pre-entrenados..." + Style.RESET_ALL)
    print(Fore.YELLOW + "Comenzando entrenamiento de la clasificacion de polaridad..." + Style.RESET_ALL)
    train_polarity_model(modelo,train_data,num_epochs, batch)

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

def obtener_parametros():
    """Obtiene y valida los parámetros de entrenamiento (batch y epochs)."""
    while True:
        try:
            batch = int(input("\nIngrese el valor del batch: ").strip())
            epochs = int(input("Ingrese el valor del epochs: ").strip())
            if batch <= 0 or epochs <= 0:
                raise ValueError("Batch y epochs deben ser números positivos.")
            return batch, epochs
        except ValueError as e:
            print(f"Error: {e}. Por favor, ingrese valores válidos.")

# ================= PROGRAMA PRINCIPAL =================
def main():
    print(Style.BRIGHT + Fore.BLUE + "╔══════════════════════════════════════════════╗")
    print("║ " + Fore.WHITE + "    Bienvenido al Programa de Extracción    " + Fore.BLUE + " ║")
    print("║ " + Fore.WHITE + "            de Aspectos en ABSA             " + Fore.BLUE + " ║")
    print("╚══════════════════════════════════════════════╝" + Style.RESET_ALL)
    
    while True:
        opcion = mostrar_menu()
        
        if opcion == '1':
            train_model()
        elif opcion == '2':
            predecir_aspectos()
        elif opcion == '3':
            clasificar_polaridad()
        elif opcion == '4':
            print(Fore.RED + "\nSaliendo del programa..." + Style.RESET_ALL)
            break
        else:
            print(Fore.RED + "\nOpción inválida. Intente nuevamente." + Style.RESET_ALL)

if __name__ == "__main__":
    main()