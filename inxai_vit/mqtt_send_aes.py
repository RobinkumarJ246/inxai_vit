from cryptography.hazmat.primitives.ciphers import Cipher, algorithms, modes
from cryptography.hazmat.backends import default_backend
from cryptography.hazmat.primitives import padding
import base64
import paho.mqtt.client as mqtt
key = b'Sixteen byte key'

def aes_encrypt(message, key):
    backend = default_backend()
    iv = b'0123456789abcdef'  # Initialization vector (IV)
    cipher = Cipher(algorithms.AES(key), modes.CBC(iv), backend=backend)
    encryptor = cipher.encryptor()
    padder = padding.PKCS7(algorithms.AES.block_size).padder()
    padded_data = padder.update(message.encode()) + padder.finalize()
    ciphertext = encryptor.update(padded_data) + encryptor.finalize()
    return base64.b64encode(ciphertext).decode()

def aes_decrypt(ciphertext, key):
    backend = default_backend()
    iv = b'0123456789abcdef'  # Initialization vector (IV)
    cipher = Cipher(algorithms.AES(key), modes.CBC(iv), backend=backend)
    decryptor = cipher.decryptor()
    ciphertext = base64.b64decode(ciphertext.encode())
    decrypted_data = decryptor.update(ciphertext) + decryptor.finalize()
    unpadder = padding.PKCS7(algorithms.AES.block_size).unpadder()
    unpadded_data = unpadder.update(decrypted_data) + unpadder.finalize()
    return unpadded_data.decode()

# MQTT broker configuration
broker_address = "192.168.137.62"  # Replace with your Raspberry Pi's IP address
port = 1884  # Default MQTT port
topic = "test/sample"

# Create an MQTT client instance
client = mqtt.Client()

# Connect to the MQTT broker
client.connect(broker_address, port=port)

# Publish a message
while True:
    message = input("Message : ")
    encrypted_message = aes_encrypt(message, key)
    client.publish(topic, encrypted_message)

# Disconnect from the broker
client.disconnect()