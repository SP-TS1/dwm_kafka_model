import joblib
import pandas as pd
from sklearn.pipeline import Pipeline

from utils import PREDICT_TOPIC, createConsumer, createProducer, prepareDataRow, serializePayload

def calculate_sma(pred_data: list[float]):
    return (pred_data.sum())/len(pred_data)

if __name__ == '__main__':
    # load trained model
    x_predictor: Pipeline = joblib.load("x_predictor.joblib")
    y_predictor: Pipeline = joblib.load("y_predictor.joblib")
    z_predictor: Pipeline = joblib.load("z_predictor.joblib")

    consumer = createConsumer()
    producer = createProducer()

    for msg in consumer:
        row = prepareDataRow(msg)
        if 'NaN' in row.values():
            continue
        else:
            # pred_x_list = []
            # pred_y_list = []
            # pred_z_list = []
            # x_ema_list = []
            # y_ema_list = []
            # z_ema_list = []
            # ema_size = 5

            data_row = pd.DataFrame.from_records([row])
            targets = ['true_x', 'true_y', 'true_z']
            X = data_row.drop(targets, axis=1)
            pred_x = float(x_predictor.predict(X))
            pred_y = float(y_predictor.predict(X))
            pred_z = float(z_predictor.predict(X))

            # pred_x_list.append(pred_x)
            # pred_y_list.append(pred_y)
            # pred_z_list.append(pred_z)
            # if len(pred_x_list == ema_size):
            #     sma = calculate_sma(pred_x_list)
            #     x_ema_list.append()
                

            payload = serializePayload(pred_x, pred_y, pred_z)
            print(f"{row['x']},{row['y']},{row['z']}","=>", payload)
            producer.send(topic=PREDICT_TOPIC, value=payload)
