import paho.mqtt.client as mqtt

# MQTT broker configuration
broker_address = "localhost"  # Replace with your Raspberry Pi's IP address
port = 1883  # Default MQTT port
topic = "test/sample"

# Create an MQTT client instance
client = mqtt.Client()

# Connect to the MQTT broker
client.connect(broker_address, port=port)

# Publish a message
while True:
    message = input("Message : ")
    client.publish(topic, message)

# Disconnect from the broker
client.disconnect()