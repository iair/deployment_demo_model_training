import xgboost as xgb
import pandas as pd

def train_model(X_train: pd.DataFrame, y_train:pd.Series,params: dict=None)-> xgb.sklearn.XGBClassifier:
    """
    Función para entrenar el modelo

    Args:
        X_train (pd.DataFrame): Dataframe con las variables independientes
        y_train (pd.Series): Serie con la variable objetivo
        params (dict, optional): Diccionario con los parámetros del modelo. Defaults to None.

    Returns:
        xgboost.sklearn.XGBClassifier: Modelo entrenado
    """
    
    if params is None:
        params = {'objective': 'binary:logistic', 
                  'eval_metric' : 'logloss',
                  'n_estimators': 300, 
                  'max_depth': 2, 
                  'learning_rate': 0.1, 
                  'random_state' : 42
                  }
    model = xgb.XGBClassifier(**params)
    model.fit(X_train, y_train)
    
    return model