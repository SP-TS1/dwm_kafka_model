import configparser
import json

from kafka import KafkaConsumer, KafkaProducer

# read config file
config = configparser.ConfigParser()
config.read('config.ini')

KAFKA_HOST = config["CONNECTION"]["KAFKA_HOST"]
KAFKA_TOPIC = config["CONNECTION"]["KAFKA_TOPIC"]
PREDICT_TOPIC = config["CONNECTION"]["PREDICT_TOPIC"]
JSON_FIELDS = config["CONNECTION"]["JSON_FIELDS"].split(",")

# function to parse json from Kafka stream and add node position
# from config file then convert to dataframe row


def prepareDataRow(msg) -> dict:
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
    return row

# function to turn predicted positions to json payload


def serializePayload(x, y, z):
    message = {"x": x, "y": y, "z": z}
    return json.dumps(message).encode('utf-8')

# initialize Kafka consumer with predefined config


def createConsumer():
    return KafkaConsumer(
        KAFKA_TOPIC,
        auto_offset_reset='latest',  # earliest
        enable_auto_commit=True,
        bootstrap_servers=KAFKA_HOST
    )

# initialize Kafka producer with predefined config


def createProducer():
    return KafkaProducer(bootstrap_servers=KAFKA_HOST)
