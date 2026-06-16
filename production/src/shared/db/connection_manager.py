"""
Connection Manager - Centralized database connection management
"""

import psycopg2
import redis
import clickhouse_connect
from typing import Optional, Dict
from dataclasses import dataclass
import logging
from tenacity import retry, stop_after_attempt, wait_exponential, retry_if_exception_type

logger = logging.getLogger(__name__)


@dataclass
class DatabaseConfig:
    """Database configuration"""
    host: str
    port: int
    database: str
    username: Optional[str] = None
    password: Optional[str] = None


class ConnectionManager:
    """Manage database connections"""
    
    def __init__(self):
        self.postgres_conn: Optional[psycopg2.extensions.connection] = None
        self.redis_client: Optional[redis.Redis] = None
        self.clickhouse_client: Optional[clickhouse_connect.driver.Client] = None
        self.timescale_conn: Optional[psycopg2.extensions.connection] = None
    
    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=0.1, min=0.1, max=0.5),
        retry=retry_if_exception_type(psycopg2.OperationalError),
        reraise=True
    )
    def _connect_postgres(self, config: DatabaseConfig) -> psycopg2.extensions.connection:
        return psycopg2.connect(
            host=config.host,
            port=config.port,
            database=config.database,
            user=config.username,
            password=config.password
        )

    def get_postgres_connection(self, config: DatabaseConfig) -> Optional[psycopg2.extensions.connection]:
        """Get or create PostgreSQL connection"""
        if self.postgres_conn is None or self.postgres_conn.closed:
            try:
                self.postgres_conn = self._connect_postgres(config)
            except Exception as e:
                logger.warning(f"Failed to connect to PostgreSQL: {e}. Falling back.")
                self.postgres_conn = None
        return self.postgres_conn
    
    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=0.1, min=0.1, max=0.5),
        reraise=True
    )
    def _connect_redis(self, host: str, port: int, db: int, password: Optional[str]) -> redis.Redis:
        client = redis.Redis(
            host=host,
            port=port,
            db=db,
            password=password,
            decode_responses=True
        )
        client.ping()
        return client

    def get_redis_connection(self, host: str = 'localhost', port: int = 6379,
                           db: int = 0, password: Optional[str] = None) -> Optional[redis.Redis]:
        """Get or create Redis connection"""
        if self.redis_client is None:
            try:
                self.redis_client = self._connect_redis(host, port, db, password)
            except Exception as e:
                logger.warning(f"Failed to connect to Redis: {e}. Falling back.")
                self.redis_client = None
        return self.redis_client
    
    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=0.1, min=0.1, max=0.5),
        reraise=True
    )
    def _connect_clickhouse(self, host: str, port: int, database: str,
                            username: Optional[str], password: Optional[str]) -> clickhouse_connect.driver.Client:
        return clickhouse_connect.get_client(
            host=host,
            port=port,
            database=database,
            username=username,
            password=password
        )

    def get_clickhouse_connection(self, host: str = 'localhost', port: int = 8123,
                                  database: str = 'default',
                                  username: Optional[str] = None,
                                  password: Optional[str] = None) -> Optional[clickhouse_connect.driver.Client]:
        """Get or create ClickHouse connection"""
        if self.clickhouse_client is None:
            try:
                self.clickhouse_client = self._connect_clickhouse(
                    host, port, database, username, password
                )
            except Exception as e:
                logger.warning(f"Failed to connect to ClickHouse: {e}. Falling back.")
                self.clickhouse_client = None
        return self.clickhouse_client
    
    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=0.1, min=0.1, max=0.5),
        retry=retry_if_exception_type(psycopg2.OperationalError),
        reraise=True
    )
    def _connect_timescale(self, config: DatabaseConfig) -> psycopg2.extensions.connection:
        return psycopg2.connect(
            host=config.host,
            port=config.port,
            database=config.database,
            user=config.username,
            password=config.password
        )

    def get_timescale_connection(self, config: DatabaseConfig) -> Optional[psycopg2.extensions.connection]:
        """Get or create TimescaleDB connection (uses PostgreSQL)"""
        if self.timescale_conn is None or self.timescale_conn.closed:
            try:
                self.timescale_conn = self._connect_timescale(config)
            except Exception as e:
                logger.warning(f"Failed to connect to TimescaleDB: {e}. Falling back.")
                self.timescale_conn = None
        return self.timescale_conn
    
    def close_all(self) -> None:
        """Close all connections"""
        if self.postgres_conn and not self.postgres_conn.closed:
            self.postgres_conn.close()
        if self.timescale_conn and not self.timescale_conn.closed:
            self.timescale_conn.close()
        if self.redis_client:
            self.redis_client.close()
        if self.clickhouse_client:
            self.clickhouse_client.close()
        
        self.postgres_conn = None
        self.timescale_conn = None
        self.redis_client = None
        self.clickhouse_client = None


# Global connection manager instance
connection_manager = ConnectionManager()
