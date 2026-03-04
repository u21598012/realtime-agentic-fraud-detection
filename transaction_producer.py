import time
import random
from kafka import KafkaProducer
import random

producer = KafkaProducer(
    bootstrap_servers='localhost:9092',
    value_serializer=lambda v: str(v).encode('utf-8')
)

topic = "transactions"

print("Producer started...")

while True:
    #randomly sample without replacement from test_fraud_detection_data.csv
    transaction = random.choice(open('test_fraud_detection_data.csv').readlines())
    producer.send(topic, transaction)
    print(f"Sent transaction: {transaction.strip()}")
    time.sleep(random.uniform(1, 5))