import pandas as pd
import numpy as np

def process_data(df: pd.DataFrame, columns_to_impute: list, target_column: str = None) -> tuple[pd.DataFrame, pd.Series]:
    """
    Procesa los datos:
    - Imputa los valores faltantes
    - Escalar las variables numéricas

    Args:
       data (pd.DataFrame): DataFrame con los datos a procesar.

    Returns:
       pd.DataFrame: DataFrame con los datos procesados
    """
    df[columns_to_impute] = df[columns_to_impute].replace(0, np.nan)
    df.loc[(df['Glucose'] == 0) & (df['SkinThickness'].isnull()), 'Glucose'] = np.nan
    df.loc[(df['Glucose'] == 0) & (df['BloodPressure'].isnull()), 'Glucose'] = np.nan
    df.loc[(df['Glucose'] == 0) & (df['BMI'].isnull()), 'Glucose'] = np.nan
    df.loc[(df['Insulin'] == 0) & (df['SkinThickness'].isnull()), 'Insulin'] = np.nan
    df.loc[(df['Insulin'] == 0) & (df['BloodPressure'].isnull()), 'Insulin'] = np.nan
    df.loc[(df['Insulin'] == 0) & (df['BMI'].isnull()), 'Insulin'] = np.nan
    df['Glucose'] = df['Glucose'].replace(0, np.nan)
    df.loc[(df['Insulin'] == 0) & (df['Glucose'].isnull()), 'Insulin'] = np.nan
    df['Insulin'] = df['Insulin'].replace(0, np.nan)
    
    target = df[target_column] if target_column else None
    
    return df, target
    
    
