import configparser
import json

from kafka import KafkaConsumer, KafkaProducer
from numpy import mean

# read config file
config = configparser.ConfigParser()
config.read('config.ini')

KAFKA_HOST = config["CONNECTION"]["KAFKA_HOST"]
KAFKA_TOPIC = config["CONNECTION"]["KAFKA_TOPIC"]
PREDICT_TOPIC = config["CONNECTION"]["PREDICT_TOPIC"]
JSON_FIELDS = config["CONNECTION"]["JSON_FIELDS"].split(",")
ANCHORS_NAME = config["ANCHORS_NAME"]["names"].split(",")

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


def serializePayload(x, y, z, tagName):
    tag = {"name": tagName, "x": x, "y": y, "z": z}
    message = {"tags": [tag]}
    return json.dumps(message).encode('utf-8')

# initialize Kafka consumer with predefined config


def createConsumer():
    return KafkaConsumer(
        KAFKA_TOPIC,
        auto_offset_reset='latest',
        enable_auto_commit=True,
        bootstrap_servers=KAFKA_HOST
    )

# initialize Kafka producer with predefined config


def createProducer():
    return KafkaProducer(bootstrap_servers=KAFKA_HOST)


# function to calculate ema for predicted value of each position

N = 9  # EMA size
K = 2/(N+1)
INIT_X = 0
INIT_Y = 0
INIT_Z = 0
prev_sma_x = []
prev_sma_y = []
prev_sma_z = []


def calculate_sma(value: float, position: str) -> float:
    global INIT_X, INIT_Y, INIT_Z, prev_sma_x, prev_sma_y, prev_sma_z

    if position == 'X' and len(prev_sma_x) <= N:
        prev_sma_x.append(value)
        return value
    if position == 'Y' and len(prev_sma_y) <= N:
        prev_sma_y.append(value)
        return value
    if position == 'Z' and len(prev_sma_z) <= N:
        prev_sma_z.append(value)
        return value
    else:
        if position == 'X':
            prev_sma_x.append(value)
            prev_sma_x.pop(0)
            return mean(prev_sma_x)
        if position == 'Y':
            prev_sma_y.append(value)
            prev_sma_y.pop(0)
            return mean(prev_sma_y)
        if position == 'Z':
            prev_sma_z.append(value)
            prev_sma_z.pop(0)
            return mean(prev_sma_z)
