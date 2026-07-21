# scripts/check_db.py
"""Smoke test: confirma que conexão async funciona."""

import asyncio

from sqlalchemy import text

from app.core.db import engine


async def main():
    async with engine.connect() as conn:
        result = await conn.execute(text("SELECT version()"))
        print(result.scalar())


if __name__ == "__main__":
    asyncio.run(main())
