import pandas as pd
import numpy as np
from pathlib import Path
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler

def load_dataset():
    dataset = pd.DataFrame()
    cwd = Path.joinpath(Path.cwd(), "dataset")
    csv_paths = Path.glob(cwd,"*.csv")
    for path in csv_paths:
        df = pd.read_csv(path)
        dataset = pd.concat([dataset, df], ignore_index=True)
    return dataset

if __name__ == '__main__':
    dataset = load_dataset()
    print(dataset)
