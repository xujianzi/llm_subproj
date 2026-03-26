from __future__ import annotations

from contextlib import contextmanager
from typing import Generator, Optional

import psycopg2
import psycopg2.extras
from psycopg2.extensions import connection as PgConnection
from psycopg2.pool import SimpleConnectionPool

from Settings import get_settings

_pool: Optional[SimpleConnectionPool] = None


def _get_pool() -> SimpleConnectionPool:
    """懒初始化连接池（首次调用时建立）。"""
    global _pool
    if _pool is None:
        cfg = get_settings()
        _pool = SimpleConnectionPool(
            minconn=1,
            maxconn=10,
            dbname=cfg.db_name,
            user=cfg.db_user,
            password=cfg.db_password,
            host=cfg.db_host,
            port=cfg.db_port,
        )
    return _pool


def get_connection() -> PgConnection:
    """从连接池取出一个连接，调用方负责用完后调用 release_connection()。"""
    return _get_pool().getconn()


def release_connection(conn: PgConnection) -> None:
    """将连接归还连接池。"""
    _get_pool().putconn(conn)


@contextmanager
def db_conn() -> Generator[PgConnection, None, None]:
    """推荐用法：with db_conn() as conn，自动归还连接。

    示例：
        with db_conn() as conn:
            with conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor) as cur:
                cur.execute("SELECT * FROM my_table WHERE id = %s", (1,))
                rows = cur.fetchall()
    """
    conn = get_connection()
    try:
        yield conn
        conn.commit()
    except Exception:
        conn.rollback()
        raise
    finally:
        release_connection(conn)


def close_pool() -> None:
    """应用退出时关闭全部连接。"""
    global _pool
    if _pool is not None:
        _pool.closeall()
        _pool = None


if __name__ == "__main__":
    with db_conn() as conn:
        with conn.cursor() as cur:
            cur.execute("SELECT version();")
            print(cur.fetchone())
