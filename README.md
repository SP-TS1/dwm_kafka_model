# dwm_kafka_model
> A major limitation is that the model is *specific* to the environment in which the raw data is collected. Therefore, it is necessary to retrain the model every time the conditions change.

## System Overview
### Training phase
> ![image](https://user-images.githubusercontent.com/68238844/229580987-10a73a96-2b7a-4446-91c6-01e7e5e44ae3.png)
### Production phase
> ![image](https://user-images.githubusercontent.com/68238844/229581080-cf4779bb-7c69-4b45-b880-6c4b25bc965d.png)


## Services
* data_collector : collect samples with specified number of samples.
* model_trainer : pre-process .csv files to train model and visualize model evaluation result.
* predictor : predict true position X, Y, Z from receiving streaming features.
* visualize_realtime : compare predictions with known true position by showing MAE and line plot.
* utils : utilities for services in this project.

## User Manual
### Collecting Data (Training phase)
1. setup UWB networks and experiment enviroments eg. tag & anchor position, obstacle, measure position every nodes
2. set values in ENVIRONMENT section in `config.ini` file (everytime for each different experiments)
3. run `init-kafka-connector.sh` to start Kafka environment
4. run `data_collector.py` with `number_of_samples`[int] parameter
5. repeat each steps and change position of tag or anchors to collect variation of data

> To cover the entire range of data across the x, y and z axes, all 3D data should be collected in the data collection area.

### Predicting Real-time (Production phase)
1. the UWB network and Kafka environment need to be running to produce source data
2. configure the connection configuration eg. topic name, prediction topic name, tag and anchors name etc.
3. run `predictor.py` to predict true positions
4. to measure model's accuracy, setup known true position and run `visualize_realtime.py`

### Example `visualize_realtime`
In this test, we assign true position (X, Y, Z) equal to (4.10, 4.50, 0.80) in meter
> ![result_scene_1](https://user-images.githubusercontent.com/68238844/229591681-da78109a-bc72-4441-bfdc-414a9411470a.gif)
