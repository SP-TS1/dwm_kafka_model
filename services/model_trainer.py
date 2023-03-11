import os
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import r2_score, mean_absolute_error, mean_absolute_percentage_error, mean_squared_error, max_error
import joblib
import imageio


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


def create_pipeline(preprocessor, model):
    # function to generate pipeline for each X,Y,Z in each selected model
    models = {"LinearRegression": LinearRegression(),
              "Ridge": Ridge(),
              "Lasso": Lasso(alpha=0.1)}

    return Pipeline(
        steps=[("preprocessor", preprocessor),
               ("classifier", models[model])]
    )


def report_result(target, y_test, y_pred, y_source):

    pred_r2 = round(r2_score(y_true=y_test, y_pred=y_pred), 3)
    pred_mae = round(mean_absolute_error(y_true=y_test, y_pred=y_pred), 3)
    pred_mape = round(mean_absolute_percentage_error(
        y_true=y_test, y_pred=y_pred), 3)
    pred_mse = round(mean_squared_error(y_true=y_test, y_pred=y_pred), 3)
    pred_max = round(max_error(y_true=y_test, y_pred=y_pred), 3)

    source_r2 = round(r2_score(y_true=y_test, y_pred=y_source), 3)
    source_mae = round(mean_absolute_error(y_true=y_test, y_pred=y_source), 3)
    source_mape = round(mean_absolute_percentage_error(
        y_true=y_test, y_pred=y_source), 3)
    source_mse = round(mean_squared_error(y_true=y_test, y_pred=y_source), 3)
    source_max = round(max_error(y_true=y_test, y_pred=y_source), 3)

    data = {
        'R_squared': [pred_r2, source_r2, round((pred_r2 - source_r2)/source_r2 * 100, 3)],
        'MAE': [pred_mae, source_mae, round((pred_mae - source_mae)/source_mae * 100, 3)],
        'MAPE': [pred_mape, source_mape, round((pred_mape - source_mape)/source_mape * 100, 3)],
        'MSE': [pred_mse, source_mse, round((pred_mse - source_mse)/source_mse * 100, 3)],
        'Max Error': [pred_max, source_max, round((pred_max - source_max)/source_max * 100, 3)]
    }
    report_df = pd.DataFrame.from_dict(data, orient='index', columns=[
                                       f'{target[-1]}_predicted', f'{target[-1]}_source', 'diff percentage'])
    index_label = ['metrics']
    path = f'./../results/{model}/{target[-1]}_position.csv'
    report_df.to_csv(path, index_label=index_label)


def visualize_result(datadict, model):
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
    for angle in range(0, 360, 2):
        ax.view_init(30, angle)
        plt.draw()
        plt.title(f"{model} Result")
        plt.savefig(f'./../results/{model}/img_{angle}.png',
                    transparent=False,
                    facecolor='white'
                    )
        plt.pause(.001)
    plt.close()

    frames = []
    for angle in range(0, 360, 2):
        image = imageio.v2.imread(f'./../results/{model}/img_{angle}.png')
        frames.append(image)
    imageio.mimsave(f'./../results/{model}/visualize.gif', frames, fps=20)

    for angle in range(0, 360, 2):
        if angle != 120:
            os.remove(f"./../results/{model}/img_{angle}.png")


if __name__ == '__main__':
    dataset = load_dataset()

    targets = ['true_x', 'true_y', 'true_z']
    X = dataset.drop(targets, axis=1)

    models = ['LinearRegression', 'Ridge', 'Lasso']

    numeric_features = list(X.columns)
    numeric_transformer = Pipeline(
        steps=[("imputer", SimpleImputer(strategy="mean")),
               ("scaler", StandardScaler())]
    )

    preprocessor = ColumnTransformer(
        transformers=[("numeric", numeric_transformer, numeric_features)])

    datadict = {}

    for model in models:
        for target in targets:
            y = dataset[target]
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=0.8, random_state=99)
            pipeline = create_pipeline(preprocessor, model)
            pipeline.fit(X_train, y_train)
            trained_model_name = f"{target[-1]}_predictor.joblib"
            path = f"./../trained_model/{model}/{trained_model_name}"
            joblib.dump(pipeline, path)
            print(f"{trained_model_name} is stored !!!")

            y_pred = pipeline.predict(X_test)
            y_source = X_test[f'{target[-1]}']

            datadict[f'{target[-1]}_true'] = y_test
            datadict[f'{target[-1]}_pred'] = y_pred
            datadict[f'{target[-1]}_source'] = y_source
            print(f"generating report for {target[-1]} ...")
            report_result(target, y_test, y_pred, y_source)

        print(f"generating testing result visualization for {model} ...\n")
        visualize_result(datadict, model)
