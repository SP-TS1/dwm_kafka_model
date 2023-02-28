import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import r2_score, mean_absolute_error, mean_absolute_percentage_error, mean_squared_error, max_error
import joblib


def load_dataset():
    # function to load and join all .csv files to prepare training dataset
    print('loading dataset ...')
    dataset = pd.DataFrame()
    cwd = Path.joinpath(Path.cwd(), "../dataset")
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


def report_result(target, y_test, y_pred, y_source):

    pred_r2 = r2_score(y_true=y_test, y_pred=y_pred)
    pred_mae = mean_absolute_error(y_true=y_test, y_pred=y_pred)
    pred_mape = mean_absolute_percentage_error(y_true=y_test, y_pred=y_pred)
    pred_mse = mean_squared_error(y_true=y_test, y_pred=y_pred)
    pred_max = max_error(y_true=y_test, y_pred=y_pred)

    source_r2 = r2_score(y_true=y_test, y_pred=y_source)
    source_mae = mean_absolute_error(y_true=y_test, y_pred=y_source)
    source_mape = mean_absolute_percentage_error(
        y_true=y_test, y_pred=y_source)
    source_mse = mean_squared_error(y_true=y_test, y_pred=y_source)
    source_max = max_error(y_true=y_test, y_pred=y_source)

    data = {
        'R_squared': [pred_r2, source_r2],
        'MAE': [pred_mae, source_mae],
        'MAPE': [pred_mape, source_mape],
        'MSE': [pred_mse, source_mse],
        'Max Error': [pred_max, source_max]
    }
    report_df = pd.DataFrame.from_dict(data, orient='index', columns=[
                                       f'{target[-1]}_predicted', f'{target[-1]}_source'])
    index_label = ['metrics']
    path = f'./../results/{target[-1]}_position.csv'
    report_df.to_csv(path, index_label=index_label)


def visualize_result(datadict):
    fig = plt.figure()
    ax = fig.add_subplot(projection='3d')

    ax.scatter(datadict['x_true'], datadict['y_true'],
               datadict['z_true'], marker='o', label='true position')
    ax.scatter(datadict['x_pred'], datadict['y_pred'],
               datadict['z_pred'], marker='^', label='pred position')
    ax.scatter(datadict['x_source'], datadict['y_source'],
               datadict['z_source'], marker='v', label='source position')
    ax.set_xlabel('X Position')
    ax.set_ylabel('Y Position')
    ax.set_zlabel('Z Position')
    ax.legend()
    for angle in range(0, 360):
        ax.view_init(30, angle)
        plt.draw()
        plt.pause(.001)
    # path = f'./../results/3D_plot.png'
    # plt.savefig(path, dpi=200)
    plt.close()


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

    datadict = {}

    for target in targets:
        y = dataset[target]
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.6, random_state=99)
        pipe = create_pipeline(preprocessor)
        pipe.fit(X_train, y_train)
        model_name = f"{target[-1]}_predictor.joblib"
        path = f"./../trained_model/{model_name}"
        joblib.dump(pipe, path)
        print(f"{model_name} is stored !!!")

        y_pred = pipe.predict(X_test)
        y_source = X_test[f'{target[-1]}']

        datadict[f'{target[-1]}_true'] = y_test
        datadict[f'{target[-1]}_pred'] = y_pred
        datadict[f'{target[-1]}_source'] = y_source

        report_result(target, y_test, y_pred, y_source)

    visualize_result(datadict)
