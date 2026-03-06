from fastapi import FastAPI
from kafka import KafkaConsumer
import threading
import csv
import os
from agentic_system import AgentState, Transaction_Inspector

app = FastAPI()

topic = "transactions"
output_file = "output.csv"

inspector = Transaction_Inspector()

CSV_COLUMNS = [
    "step",
    "type",
    "amount",
    "nameorig",
    "oldbalanceorg",
    "newbalanceorig",
    "namedest",
    "oldbalancedest",
    "newbalancedest",
    "isfraud",
    "isflaggedfraud",
    "type_encoded",
    "balancechangeorig",
    "balancechangedest",
    "errorbalanceorig",
    "errorbalancedest",
    "issameuser",
    "transactionsperuser",
    "fraudratioperuser",
]

OUTPUT_COLUMNS = CSV_COLUMNS + [
    "route",
    "decision",
    "predicted",
    "actual",
    "outcome_reason",
    "trace",
]


def _cast_value(key: str, value: str):
    if key in {"isfraud", "isflaggedfraud", "type_encoded", "step"}:
        try:
            return int(value)
        except Exception:
            return 0
    if key in {"issameuser"}:
        val = str(value).strip().lower()
        return val in {"true", "1", "yes", "y"}
    try:
        return float(value)
    except Exception:
        return value


def parse_transaction_line(line: str) -> dict | None:
    if not line:
        return None
    line = line.strip()
    if not line or line.lower().startswith("step,"):
        return None  # skip header lines
    try:
        row = next(csv.reader([line]))
    except Exception:
        return None
    if len(row) != len(CSV_COLUMNS):
        return None
    tx = {col: _cast_value(col, val) for col, val in zip(CSV_COLUMNS, row)}
    return tx


def write_output(row: dict):
    file_exists = os.path.exists(output_file)
    write_header = not file_exists or os.path.getsize(output_file) == 0
    with open(output_file, "a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=OUTPUT_COLUMNS)
        if write_header:
            writer.writeheader()
        writer.writerow(row)

def consume_messages():
    consumer = KafkaConsumer(
        topic,
        bootstrap_servers='localhost:9092',
        auto_offset_reset='earliest',
        group_id='fastapi-group',
        value_deserializer=lambda m: m.decode('utf-8').strip(),
    )

    print("Consumer started...")

    for message in consumer:
        raw_line = message.value
        tx = parse_transaction_line(raw_line)
        if tx is None:
            print(f"Skipped malformed or header line: {raw_line}")
            continue

        result = inspector.execute(tx)

        output_row = {
            **tx,
            "route": result.get("route"),
            "decision": result.get("decision"),
            "predicted": result.get("predicted"),
            "actual": result.get("actual"),
            "outcome_reason": result.get("outcome_reason", ""),
            "trace": "|".join(result.get("trace", [])),
        }

        write_output(output_row)

        print(f"Processed transaction: {tx} -> route={output_row['route']} decision={output_row.get('decision')} trace={output_row['trace']}")

@app.on_event("startup")
def start_consumer():
    thread = threading.Thread(target=consume_messages)
    thread.daemon = True
    thread.start()

@app.get("/")
def read_root():
    return {"status": "Consumer running"}