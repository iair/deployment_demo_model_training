from datetime import datetime
import joblib
import xgboost

def save_model(model: xgboost.sklearn.XGBClassifier, model_path: str) -> None:
    """
    Guarda el modelo en un archivo usando joblib.

    Args:
        model (xgboost.sklearn.XGBClassifier): Modelo a guardar.
        model_path (str): Ruta donde guardar el modelo (sin extensión).
    """
    
    current_date = datetime.now().strftime("%Y-%m-%d")
    full_path = f"{model_path}_{current_date}.joblib"
    joblib.dump(model, full_path)
    print(f"Modelo guardado en {full_path}")