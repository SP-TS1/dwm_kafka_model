import pandas as pd
import numpy as np
from pathlib import Path
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import GridSearchCV, KFold, cross_val_score, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
import joblib

def load_dataset():
    # function to load and join all .csv files to prepare training dataset
    print('loading dataset ...')
    dataset = pd.DataFrame()
    cwd = Path.joinpath(Path.cwd(), "dataset")
    csv_paths = Path.glob(cwd, "*.csv")
    for path in csv_paths:
        df = pd.read_csv(path)
        dataset = pd.concat([dataset, df], ignore_index=True)
    print('dataset successfully loaded !!')
    return dataset


def create_pipeline(preprocessor):
    # function to generate pipeline for each X,Y,Z
    return Pipeline(
        steps=[("preprocessor", preprocessor),
               ("classifier", LinearRegression())]
    )


if __name__ == '__main__':
    dataset = load_dataset()
    targets = ['true_x', 'true_y', 'true_z']
    X = dataset.drop(targets, axis=1)

    numeric_features = list(X.columns)
    numeric_transformer = Pipeline(
        steps=[("imputer", SimpleImputer(strategy="mean")),
               ("scaler", StandardScaler())]
    )

    preprocessor = ColumnTransformer(
        transformers=[("numeric", numeric_transformer, numeric_features)])

    for target in targets:
        y = dataset[target]
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.33, random_state=99)
        pipe = create_pipeline(preprocessor)
        pipe.fit(X_train, y_train)
        model_name = f"{target[-1]}_predictor.joblib"
        joblib.dump(pipe, model_name)

    # X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=99)
    # pipeline.fit(X_train, y_train)
    # y_pred = pipeline.predict(X_test)
    # print(pipeline.score(y_test, y_pred))

    # cv = KFold(n_splits=10, shuffle=True, random_state=None)

    # scores = cross_val_score(
    #     pipeline, X, y, scoring='neg_median_absolute_error', cv=cv, n_jobs=-1)
    # scores = np.absolute(scores)
    # print(f'MAE: {round(np.mean(scores),3)} {round(np.std(scores),3)}')
