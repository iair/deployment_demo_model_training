from sklearn.model_selection import train_test_split
import pandas as pd

def split_data(data:pd.DataFrame,target_column: str, test_size=0.25, 
               random_state=42,stratify: bool = True,
               ) -> tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
    """
    Dividir los datos en conjuntos de entrenamiento y prueba

    Args:
        data (pd.DataFrame): Dataframe que contiene los datos a dividir
        target_column (str): Nombre de la columna objetivo
        test_size (float, optional): Proporción de los datos que quedan en el test. Defaults to 0.2.
        random_state (int, optional): Semilla para la aleatoriedad. Defaults to 42.

    Returns:
        Tupla: Una tupla que contiene los conjuntos de entrenamiento y prueba
        - X_train (pd.DataFrame): Conjunto de entrenamiento de las variables independientes
        - X_test (pd.DataFrame): Conjunto de prueba de las variables independientes
        - y_train (pd.Series): Conjunto de entrenamiento de la variable objetivo
        - y_test (pd.Series): Conjunto de prueba de la variable objetivo
    """
    
    X = data.drop(columns = target_column, axis=1)
    y = data[target_column]
    if stratify:
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=random_state, stratify=y)
    else:
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=random_state)
    
    #X_train = X_train.dropna()
    #y_train = y_train.loc[X_train.index]
    
    return X_train, X_test, y_train, y_test