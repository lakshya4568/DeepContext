import asyncio
import os
from urllib.parse import urlparse

import asyncpg
from dotenv import load_dotenv

load_dotenv()


async def init():
    db_type = os.getenv("DATABASE_TYPE", "postgres").lower()
    dsn = os.getenv("POSTGRES_DSN", "postgresql://postgres:postgres@127.0.0.1:5432/awems")

    if db_type == "sqlite":
        print("[+] DATABASE_TYPE is sqlite. No PostgreSQL configuration required.")
        print("DB_SETUP_SUCCESS=True")
        return

    parsed = urlparse(dsn)
    user = parsed.username or "postgres"
    password = parsed.password or "postgres"
    host = parsed.hostname or "127.0.0.1"
    port = parsed.port or 5432
    target_db = parsed.path.lstrip("/") or "awems"

    print(f"[*] Connecting to PostgreSQL at {host}:{port}...")
    try:
        # 1. Connect to postgres system database
        sys_conn = await asyncpg.connect(
            user=user, password=password, host=host, port=port, database="postgres"
        )
        exists = await sys_conn.fetchval("SELECT 1 FROM pg_database WHERE datname = $1", target_db)
        if not exists:
            print(f'[*] Creating database "{target_db}"...')
            await sys_conn.execute(f'CREATE DATABASE "{target_db}"')
            print(f'[+] Database "{target_db}" created successfully.')
        else:
            print(f'[+] Database "{target_db}" already exists.')
        await sys_conn.close()

        # 2. Connect to target database
        target_conn = await asyncpg.connect(
            user=user, password=password, host=host, port=port, database=target_db
        )
        try:
            await target_conn.execute("CREATE EXTENSION IF NOT EXISTS pgcrypto;")
            print('[+] Extension "pgcrypto" enabled.')
        except Exception as e_pgc:
            print(f"[!] pgcrypto notice: {e_pgc}")

        try:
            await target_conn.execute("CREATE EXTENSION IF NOT EXISTS vector;")
            print('[+] Extension "vector" (pgvector) enabled successfully.')
        except Exception as e_vec:
            print(f"[!] vector notice: {e_vec}")

        try:
            await target_conn.execute("CREATE EXTENSION IF NOT EXISTS pg_trgm;")
            print('[+] Extension "pg_trgm" enabled successfully.')
        except Exception as e_trgm:
            print(f"[!] pg_trgm notice: {e_trgm}")

        try:
            await target_conn.execute("CREATE EXTENSION IF NOT EXISTS pg_search;")
            print('[+] Extension "pg_search" (ParadeDB BM25) enabled successfully.')
        except Exception as e_search:
            print(f"[!] pg_search notice: {e_search}")

        await target_conn.close()
        print("DB_SETUP_SUCCESS=True")
    except Exception as e:
        print(f"DB_SETUP_ERROR={e}")


if __name__ == "__main__":
    asyncio.run(init())
