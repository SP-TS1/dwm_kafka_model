
import json
import numpy as np
from kafka import KafkaConsumer
from sklearn.metrics import mean_absolute_error
from utils import KAFKA_HOST
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

GRAPH_LENGTH = 100
index_list = [0]
source_data = {"x": [0], "y": [0], "z": [0]}
pred_data = {"x": [0], "y": [0], "z": [0]}
true_data = {"x": [0], "y": [0], "z": [0]}
empty_data = {"x": np.nan, "y": np.nan, "z": np.nan}

plt.style.use('seaborn-darkgrid')

SOURCE_TOPIC = 'position'
PREDICTS_TOPIC = 'predicted_position'


def append_list(datadict: dict, data: dict):
    for pos in ['x', 'y', 'z']:
        datadict[pos].append(data[pos])

    if len(datadict["x"]) > GRAPH_LENGTH:
        for pos in ['x', 'y', 'z']:
            datadict[pos].pop(0)
    return datadict


def pop_all(datadict: dict):
    for pos in ['x', 'y', 'z']:
        datadict[pos].pop(0)
    return datadict


def calculate_mae(data_series, true_value):
    return round(mean_absolute_error(y_true=[true_value for i in data_series], y_pred=data_series), 3)


def update(frame):
    for ax in range(3):
        axs[ax].cla()
        plot_subplot(ax, index_list, source_data, pred_data, true_data)
        axs[ax].legend(loc="upper left")
    fig.gca().relim()
    fig.gca().autoscale_view()


def plot_subplot(ax, index_list, source_data, pred_data, true_data):
    axis = ['x', 'y', 'z']

    idx = np.array(index_list)
    source_series = np.array(source_data[axis[ax]])
    source_mask = np.isfinite(source_series)
    pred_series = np.array(pred_data[axis[ax]])
    pred_mask = np.isfinite(pred_series)

    source_mae = calculate_mae(source_series[np.logical_not(
        np.isnan(source_series))], true_data[axis[ax]][-1])
    pred_mae = calculate_mae(pred_series[np.logical_not(
        np.isnan(pred_series))], true_data[axis[ax]][-1])

    # source
    axs[ax].plot(idx[source_mask], source_series[source_mask], 'g.-',
                 label=f'source [MAE={source_mae}]', linewidth=0.5, alpha=0.5)
    # predicted
    axs[ax].plot(idx[pred_mask], pred_series[pred_mask], '.-',
                 label=f'predicted [MAE={pred_mae}]', color="orange", linewidth=0.5, alpha=0.8)

    axs[ax].set_xlabel("data length")
    axs[ax].set_ylabel(f"{axis[ax]} position")


if __name__ == '__main__':
    # subscribe 2 topics
    topics = [SOURCE_TOPIC, PREDICTS_TOPIC]
    consumer = KafkaConsumer(
        auto_offset_reset='latest',
        enable_auto_commit=True,
        bootstrap_servers=KAFKA_HOST
    )
    consumer.subscribe(topics)
    fig, axs = plt.subplots(nrows=3, ncols=1, figsize=(6, 8))

    animation = FuncAnimation(fig, update, interval=500)
    plt.tight_layout()

    true_x = float(input('please enter true_x (float): '))
    true_y = float(input('please enter true_y (float): '))
    true_z = float(input('please enter true_z (float): '))

    true_data = {"x": [true_x], "y": [true_y], "z": [true_z]}

    first_insert = True

    while True:
        plt.pause(0.1)
        records = consumer.poll(timeout_ms=1000)
        for _, consumer_records in records.items():
            for msg in consumer_records:
                try:
                    # extract value from kafka message in json format
                    data = json.loads(msg.value)

                    # if values values again and parse to dict object in {x,y,z} format
                    if msg.topic == SOURCE_TOPIC:
                        data = json.loads(data)
                        position_data = data[SOURCE_TOPIC]
                    elif msg.topic == PREDICTS_TOPIC:
                        position_data = data['tags'][0]

                    print(msg.topic, ":", position_data)

                    if msg.topic == SOURCE_TOPIC:
                        source_data = append_list(source_data, position_data)
                        pred_data = append_list(pred_data, empty_data)
                    elif msg.topic == PREDICTS_TOPIC:
                        pred_data = append_list(pred_data, position_data)
                        source_data = append_list(source_data, empty_data)

                    # keep append index when new data are fed
                    index_list.append(index_list[-1] + 1)
                    true_data = append_list(
                        true_data, {"x": true_x, "y": true_y, "z": true_z})

                    # pop initial zero value from every dict and index_list
                    if first_insert:
                        index_list.pop(0)
                        source_data = pop_all(source_data)
                        pred_data = pop_all(pred_data)
                        true_data = pop_all(true_data)
                        first_insert = False

                    # pop head if data are exceed specified graph length to only keep tracking latest data
                    if len(index_list) > GRAPH_LENGTH:
                        index_list.pop(0)
                        true_data = pop_all(true_data)
                    print("\n")
                except:
                    print("something went wrong in parsing message from Kafka")
        continue
