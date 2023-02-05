import configparser
import subprocess
import os

# read config file to collect 
config = configparser.ConfigParser()
config.read('config.ini')

os.chdir("/usr/local/etc/kafka/")
subprocess.call("export CLASSPATH=/Users/max_sp/Downloads/kafka-connect-mqtt/target/kafka-connect-mqtt-1.1.0-package/kafka-connect-mqtt/*", shell=True)

print("changed directory to /usr/local/etc/kafka/")
start_cmd = """
    zookeeper-server-start -daemon zookeeper.properties && \
    while ! nc -z localhost 2181; do sleep 0.1; done && \
    kafka-server-start -daemon server.properties && \
    while ! nc -z localhost 9092; do sleep 0.1; done 
"""

subprocess.call(start_cmd, shell=True)

init_connector_cmd = """
    connect-standalone -daemon connect-standalone.properties connectors-properties/mqtt-position-connector.properties &&\
    while ! nc -z localhost 8083; do sleep 0.1; done
"""
subprocess.call(init_connector_cmd, shell=True)
print("... waiting for kafka-connector to be established ...")
subprocess.call("sleep 5", shell=True)
print("Successfully established kafka-mqtt-connector")

nodes = config["ANCHORS_NAME"]["names"].split(",")

for node in nodes:
    print(f"... waiting for anchor-{node}-connector to be established ...", end="  ")
    command = f"""
        curl -X POST \
        http://localhost:8083/connectors \
        -H 'Content-Type: application/json' \
        -d '{{ "name": "anchor-{node}-connector",
            "config" : {{
                "connector.class":"be.jovacon.kafka.connect.MQTTSourceConnector",
                "mqtt.topic":"dwm/node/{node}/uplink/config",
                "kafka.topic":"anchor_{node}",
                "mqtt.clientID":"my_client_id",
                "mqtt.broker":"tcp://172.20.10.8:1883",
                "errors.log.enable":"true",
                "errors.log.include.messages":"true",
                "key.converter":"org.apache.kafka.connect.json.JsonConverter",
                "key.converter.schemas.enable":"false",
                "value.converter":"org.apache.kafka.connect.json.JsonConverter",
                "value.converter.schemas.enable":"false"
            }}
        }}'
    """
    subprocess.call(command, shell=True)
    print("DONE")
print("\nSuccessfully establish servers and connectors !!")