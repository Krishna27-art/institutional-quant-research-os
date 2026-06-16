"""
Data Gateway - Market data ingestion with Kafka producers
Phase 14 Implementation (moved up for data flow)
"""

from .nse.nse_gateway import NSEGateway
from .kafka_producer import KafkaDataProducer

__all__ = [
    'NSEGateway',
    'KafkaDataProducer',
]
