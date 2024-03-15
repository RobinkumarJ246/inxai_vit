import paho.mqtt.client as mqtt

# Set up MQTT broker configuration
broker_address = "192.168.137.62"  # Replace with your broker's IP address
port = 8883  # Default MQTT TLS port
topic = "test/sample"

# Callback function when a message is received
def on_message(client, userdata, message):
    print("Received message on topic {}: {}".format(message.topic, str(message.payload.decode("utf-8"))))

# Create an MQTT client instance
client = mqtt.Client()

# Set TLS parameters
client.tls_set(ca_certs="C:\Users\Robin\ca.crt")

# Assign the on_message function to the client
client.on_message = on_message

# Connect to the MQTT broker with TLS
client.connect(broker_address, port=port)

# Subscribe to the topic
client.subscribe(topic)

# Loop to maintain network traffic and message handling
client.loop_forever()