import paho.mqtt.client as mqtt

# MQTT broker information
broker_address = "192.168.0.70"  # Replace with your broker's address
broker_port = 1883

# Topic to subscribe to
topic = "test/sample"

# Callback when the client connects to the broker
def on_connect(client, userdata, flags, rc):
    print("Client: Connected with result code "+str(rc))
    # Subscribe to the specified topic
    client.subscribe(topic)

# Callback when a message is received from the broker
def on_message(client, userdata, msg):
    print(f"Received message on topic '{msg.topic}': {msg.payload.decode()}")

# Create an MQTT client instance
client = mqtt.Client()

# Assign callback functions
client.on_connect = on_connect
client.on_message = on_message

# Connect to the broker
client.connect(broker_address, broker_port, 60)

# Loop to maintain the connection and process messages
client.loop_start()

# Keep the script running
try:
    while True:
        pass
except KeyboardInterrupt:
    print("Client: Disconnecting from the broker")
    # Disconnect from the broker when the loop is interrupted (e.g., by Ctrl+C)
    client.disconnect()
    print("Client: Disconnected")