import json
import joblib
import pandas as pd
from sklearn.pipeline import Pipeline
from services.utils import calculate_sma

from utils import PREDICT_TOPIC, createConsumer, createProducer, prepareDataRow, serializePayload


if __name__ == '__main__':
    # load trained model
    model = 'LinearRegression'
    x_predictor: Pipeline = joblib.load(
        f"./../trained_model/{model}/x_predictor.joblib")
    y_predictor: Pipeline = joblib.load(
        f"./../trained_model/{model}/y_predictor.joblib")
    z_predictor: Pipeline = joblib.load(
        f"./../trained_model/{model}/z_predictor.joblib")

    consumer = createConsumer()
    producer = createProducer()

    for msg in consumer:
        try:
            tagName = json.loads(msg.key).split('/')[2]
            row = prepareDataRow(msg)
        except:
            print("something went wrong in parsing data from Kafka in predictor")
        if 'NaN' in row.values():
            continue
        else:
            try:
                data_row = pd.DataFrame.from_records([row])
                targets = ['true_x', 'true_y', 'true_z']
                X = data_row.drop(targets, axis=1)
                pred_x = calculate_sma(float(x_predictor.predict(X)), 'X')
                pred_y = calculate_sma(float(y_predictor.predict(X)), 'Y')
                pred_z = calculate_sma(float(z_predictor.predict(X)), 'Z')

                payload = serializePayload(pred_x, pred_y, pred_z, tagName)
                print(f"{row['x']},{row['y']},{row['z']}", "=>", payload)
                producer.send(topic=PREDICT_TOPIC, value=payload)
            except:
                print("something went wrong in predicting position data in predictor")
