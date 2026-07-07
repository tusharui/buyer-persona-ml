from fastapi import APIRouter

from api.schemas import StreamPredictRequest, StreamPredictResponse
from src.streaming import streaming_producer
from src.config import KAFKA_TOPIC

router = APIRouter(tags=["stream"])


@router.post("/stream/predict", response_model=StreamPredictResponse)
async def stream_predict(req: StreamPredictRequest):
    if streaming_producer._producer is None:
        await streaming_producer.start()

    transactions = [t.model_dump() for t in req.transactions]
    sent = await streaming_producer.send_batch(transactions, KAFKA_TOPIC)

    return StreamPredictResponse(
        status="produced",
        message=f"Sent {sent}/{len(transactions)} transactions to Kafka topic '{KAFKA_TOPIC}'.",
    )


@router.post("/stream/connect")
async def stream_connect():
    await streaming_producer.start()
    return {"status": "connected", "broker": "localhost:9092"}


@router.post("/stream/disconnect")
async def stream_disconnect():
    await streaming_producer.stop()
    return {"status": "disconnected"}
