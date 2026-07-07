import sys
import asyncio
import json
import uuid
from pathlib import Path
from typing import Optional, Callable

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.config import KAFKA_BOOTSTRAP_SERVERS, KAFKA_TOPIC, KAFKA_CONSUMER_GROUP


class StreamingProducer:
    def __init__(self):
        self._producer = None

    async def start(self):
        try:
            from aiokafka import AIOKafkaProducer
            self._producer = AIOKafkaProducer(
                bootstrap_servers=KAFKA_BOOTSTRAP_SERVERS,
                value_serializer=lambda v: json.dumps(v).encode(),
            )
            await self._producer.start()
        except Exception as e:
            print(f"[streaming] Producer start failed: {e}")
            self._producer = None

    async def send(self, transaction: dict, topic: str = None) -> bool:
        if self._producer is None:
            return False
        topic = topic or KAFKA_TOPIC
        try:
            key = transaction.get("customer_id", str(uuid.uuid4())).encode()
            await self._producer.send(topic, key=key, value=transaction)
            return True
        except Exception as e:
            print(f"[streaming] Send failed: {e}")
            return False

    async def send_batch(self, transactions: list[dict], topic: str = None) -> int:
        sent = 0
        for tx in transactions:
            if await self.send(tx, topic):
                sent += 1
        return sent

    async def stop(self):
        if self._producer:
            await self._producer.stop()


class StreamingConsumer:
    def __init__(self, callback: Optional[Callable] = None):
        self._consumer = None
        self._running = False
        self._callback = callback
        self._results: list[dict] = []

    async def start(self):
        try:
            from aiokafka import AIOKafkaConsumer
            self._consumer = AIOKafkaConsumer(
                KAFKA_TOPIC,
                bootstrap_servers=KAFKA_BOOTSTRAP_SERVERS,
                group_id=KAFKA_CONSUMER_GROUP,
                value_deserializer=lambda v: json.loads(v.decode()),
                auto_offset_reset="latest",
            )
            await self._consumer.start()
            self._running = True
        except Exception as e:
            print(f"[streaming] Consumer start failed: {e}")
            self._consumer = None

    async def consume(self, timeout: float = 5.0) -> list[dict]:
        if self._consumer is None or not self._running:
            return []
        messages = []
        try:
            while True:
                msg_set = await self._consumer.getmany(timeout_ms=int(timeout * 1000), max_records=100)
                for _, msgs in msg_set.items():
                    for msg in msgs:
                        tx = msg.value
                        messages.append(tx)
                        if self._callback:
                            result = self._callback(tx)
                            self._results.append(result)
        except asyncio.TimeoutError:
            pass
        except Exception as e:
            print(f"[streaming] Consume error: {e}")
        return messages

    async def stop(self):
        self._running = False
        if self._consumer:
            await self._consumer.stop()

    def get_results(self) -> list[dict]:
        return list(self._results)

    def clear_results(self):
        self._results.clear()


streaming_producer = StreamingProducer()
