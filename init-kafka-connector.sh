export CLASSPATH=/Users/max_sp/Downloads/kafka-connect-mqtt/target/kafka-connect-mqtt-1.1.0-package/kafka-connect-mqtt/*
echo "exported class path in ${PWD}"

cd ~/../../usr/local/etc/kafka/
echo "changed directory to /usr/local/etc/kafka/"

export CLASSPATH=/Users/max_sp/Downloads/kafka-connect-mqtt/target/kafka-connect-mqtt-1.1.0-package/kafka-connect-mqtt/*
echo "exported class path in ${PWD}"

# start zookeeper, kafka-server and kafka-mqtt-connector
zookeeper-server-start -daemon zookeeper.properties && \
while ! nc -z localhost 2181; do sleep 0.1; done && \
kafka-server-start -daemon server.properties --override advertised.listeners=PLAINTEXT://Sirapats-MacBook-Pro.local:9092 && \
while ! nc -z localhost 9092; do sleep 0.1; done && \
connect-standalone -daemon connect-standalone.properties connectors-properties/mqtt-position-connector.properties && \
while ! nc -z localhost 8083; do sleep 0.1; done
# sleep 3

echo "\nSuccessfully established servers and connectors\n"
