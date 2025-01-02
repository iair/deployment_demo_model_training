import sys
import os
# Agregar la raíz del proyecto al PYTHONPATH
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from src.data.data_loader import load_data
from src.data.data_processor import process_data
import pandas as pd
from src.data.data_splitter import split_data
from src.model.trainer import train_model
from src.model.evaluator import evaluate_model
from src.model.saver import save_model

def main():
    
    # Cargar los datos
    data = load_data(file_path = "data/raw/diabetes.csv")
    
    # Procesar los datos
    processed_data , target = process_data(
                                  df=data, 
                                  columns_to_impute=['BloodPressure', 'SkinThickness', 'BMI','Glucose','Insulin'],
                                  target_column='Outcome'
                                  )
    # qsna=processed_data.shape[0]-processed_data.isnull().sum(axis=0)
    # qna=processed_data.isnull().sum(axis=0)
    # ppna=round(100*(processed_data.isnull().sum(axis=0)/processed_data.shape[0]),2)
    # aux= {'datos sin NAs en q': qsna, 'Na en q': qna ,'Na en %': ppna}
    # na=pd.DataFrame(data=aux)
    # print(na.sort_values(by='Na en %',ascending=False).head(20))
    
    # Dividir los datos en conjuntos de entrenamiento y prueba
    X_train, X_test, y_train, y_test = split_data(processed_data,target_column='Outcome')
    # print(X_train.shape)
    # print(y_train.shape)
    # print(X_test.shape)
    # print(y_test.shape)

    # Entrenar el modelo
    model = train_model(X_train=X_train, y_train=y_train)
    # print(type(model))

    # Evaluar el modelo
    accuracy, precision, recall, f1, auc = evaluate_model(model, test_data=X_test, y_test=y_test)
    # print(f"Accuracy: {accuracy}")
    # print(f"Precision: {precision}")
    # print(f"Recall: {recall}")
    # print(f"F1: {f1}")
    # print(f"AUC: {auc}")

    # Guardar el modelo
    save_model(model, model_path="models/trained_model")
    
if __name__ == "__main__":
    main()