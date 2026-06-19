import os
import logging
import psycopg2
from psycopg2.pool import SimpleConnectionPool
from psycopg2.extras import RealDictCursor
from contextlib import contextmanager

logger = logging.getLogger(__name__)

# Fetch database URL from environment variable
DATABASE_URL = os.getenv("DATABASE_URL", "")

_pool = None

def init_pool():
    global _pool
    if _pool is not None:
        return
    if not DATABASE_URL:
        logger.error("[Database] DATABASE_URL environment variable is empty! Cannot initialize PostgreSQL pool.")
        raise ValueError("DATABASE_URL environment variable is not set. Please configure it in your environment.")
    try:
        logger.info("[Database] Initializing Supabase/PostgreSQL Connection Pool...")
        _pool = SimpleConnectionPool(1, 20, dsn=DATABASE_URL)
        logger.info("[Database] Connection pool created successfully.")
    except Exception as e:
        logger.error(f"[Database] Failed to initialize connection pool: {e}")
        raise e

@contextmanager
def get_db_cursor():
    global _pool
    if _pool is None:
        init_pool()
    conn = _pool.getconn()
    try:
        with conn.cursor(cursor_factory=RealDictCursor) as cur:
            yield cur
        conn.commit()
    except Exception as e:
        conn.rollback()
        logger.error(f"[Database] Transaction rollback due to error: {e}")
        raise e
    finally:
        _pool.putconn(conn)
