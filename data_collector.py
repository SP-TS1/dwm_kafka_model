import pandas as pd
import json
import argparse
import uuid
from kafka import KafkaConsumer

KAFKA_HOST = 'localhost:9092'
KAFKA_TOPIC = 'position'
JSON_FIELDS = ['x', 'y', 'z', 'quality']

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("records", type=int)
    parser.add_argument("true_x", type=float)
    parser.add_argument("true_y", type=float)
    parser.add_argument("true_z", type=float)
    args = parser.parse_args()

    dataset = pd.DataFrame()
    recived_msg_count = 0

    consumer = KafkaConsumer(
        KAFKA_TOPIC,
        auto_offset_reset='earliest',
        enable_auto_commit=True,
        bootstrap_servers=KAFKA_HOST
    )

    for msg in consumer:
        if (recived_msg_count < int(args.records)):
            json_string = json.loads(msg.value)
            message = json.loads(json_string)
            row = {}
            for field in JSON_FIELDS:
                value = message['position'][field]
                row[field] = value
            row['true_x'] = args.true_x
            row['true_y'] = args.true_y
            row['true_z'] = args.true_z
            if 'NaN' in row.values():
                continue
            else:
                data_row = pd.DataFrame.from_records([row])
                dataset = pd.concat([dataset, data_row])
                recived_msg_count += 1
                print(recived_msg_count, row)
        else:
            break

    id = uuid.uuid1()
    dataset.to_csv(f'./dataset/{id}.csv', index=False)
