import asyncio
import asyncpg


async def main():
    print("==================================================")
    print(" INSPECTING DATABASE LISTENING ON PORT 5432")
    print("==================================================")
    conn = await asyncpg.connect("postgresql://postgres:postgres@127.0.0.1:5432/awems")
    
    version = await conn.fetchval("SELECT version();")
    print(f"\n[+] Active Engine: {version}")

    exts = await conn.fetch("SELECT extname, extversion FROM pg_extension ORDER BY extname;")
    print("\n[+] Currently Installed Extensions in 'awems':")
    for r in exts:
        print(f"    - {r['extname']}: v{r['extversion']}")

    avail = await conn.fetch(
        "SELECT name, default_version, installed_version, comment "
        "FROM pg_available_extensions "
        "WHERE name IN ('pg_search', 'pg_trgm', 'vector', 'pgcrypto', 'paradedb') "
        "ORDER BY name;"
    )
    print("\n[+] Available Key Extensions in this PostgreSQL engine:")
    for r in avail:
        status = f"INSTALLED (v{r['installed_version']})" if r['installed_version'] else "NOT INSTALLED"
        print(f"    - {r['name']} (default v{r['default_version']}): {status} — {r['comment']}")

    await conn.close()


if __name__ == "__main__":
    asyncio.run(main())
