import paho.mqtt.client as mqtt

# MQTT broker configuration
broker_address = "localhost"  # Replace with your Raspberry Pi's IP address
port = 1884  # Default MQTT port
topic = "test/sample"

# Callback function when a message is received
def on_message(client, userdata, message):
    print("Received message on topic {}: {}".format(message.topic, str(message.payload.decode("utf-8"))))

# Create an MQTT client instance
client = mqtt.Client()

# Assign the on_message function to the client
client.on_message = on_message

# Connect to the MQTT broker
client.connect(broker_address, port=port)

# Subscribe to the topic
client.subscribe(topic)

# Loop to maintain network traffic and message handling
client.loop_forever()