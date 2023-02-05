import configparser
import pandas as pd
import json
import argparse
import uuid
from kafka import KafkaConsumer

# read config file
config = configparser.ConfigParser()
config.read('config.ini')

KAFKA_HOST = config["DEFAULT"]["KAFKA_HOST"]
KAFKA_TOPIC = config["DEFAULT"]["KAFKA_TOPIC"]
JSON_FIELDS = config["DEFAULT"]["JSON_FIELDS"].split(",")

if __name__ == '__main__':
    # create parser to take true position as arguments from user
    parser = argparse.ArgumentParser()
    parser.add_argument("records", type=int)
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
            # read values from read streaming
            for field in JSON_FIELDS:
                value = message['position'][field]
                row[field] = value
            # read values from config file
            for item in config["ENVIRONMENT"].items():
                row[item[0]] = float(item[1])
            if 'NaN' in row.values():
                continue
            else:
                data_row = pd.DataFrame.from_records([row])
                dataset = pd.concat([dataset, data_row])
                recived_msg_count += 1
                print(recived_msg_count, row)
        else:
            break

    id = uuid.uuid4()
    dataset.to_csv(f'./dataset/{id}.csv', index=False)
