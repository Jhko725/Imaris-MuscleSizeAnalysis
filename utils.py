import pandas as pd
import numpy as np
from numpy import float32, float64

def import_data(filepath):
    df_raw = pd.read_excel(filepath, skiprows = None, engine = "openpyxl")
    labels = df_raw.columns[[0, 3]]
    df = df_raw.iloc[1:, [1, 4]]
    df.columns = labels
    df.dropna(inplace = True)
    df = df.astype(float64)

    return df

def import_data_list(filepath_list):
    return list(map(import_data, filepath_list))

def df_columns_to_arrays(dataframe):
    n_cols = dataframe.shape[1]
    col_arrays = [np.array(dataframe.iloc[:, i], dtype = float32) for i in range(n_cols)]
    return col_arrays

class Statistic():

    def __init__(self, value, CI, name):
        self.value = value
        self.CI = tuple(CI)
        self.name = name

    def __str__(self):
        return f"{self.value :.2f} [{self.CI[0] :.2f}, {self.CI[1] :.2f}]"

    # convenience function to use with plt.bar y_err argument
    def errorbar(self):
        return np.array([self.value-self.CI[0], self.CI[1]-self.value])