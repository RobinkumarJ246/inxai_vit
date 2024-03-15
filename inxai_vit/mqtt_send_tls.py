import paho.mqtt.client as mqtt

# Set up MQTT broker configuration
broker_address = "192.168.137.62"  # Replace with your broker's IP address
port = 8883  # Default MQTT TLS port
topic = "test/sample"

# Create an MQTT client instance
client = mqtt.Client()

# Set TLS parameters
client.tls_set(ca_certs="C:\Users\Robin\ca.crt")

# Connect to the MQTT broker with TLS
client.connect(broker_address, port=port)

# Publish a message
while True:
    message = input("Message: ")
    client.publish(topic, message)

# Disconnect from the broker
client.disconnect()