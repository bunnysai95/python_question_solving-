import asyncio
import os
import traceback
from celery_app import celery_app

from datetime import datetime

from tortoise import Tortoise
from models import TaskStatus


async def _init_tortoise():
    # initialize Tortoise ORM inside the worker (only once)
    if not Tortoise._inited:
        from settings import settings
        await Tortoise.init(db_url=getattr(settings, "DB_URL", "sqlite://./db.sqlite3"), modules={"models": ["models"]})


async def _process_file_and_update(task_id: int, file_path: str):
    # update status
    task = await TaskStatus.get_or_none(id=task_id)
    if task:
        task.status = "working"
        task.started_at = datetime.utcnow()
        await task.save()

    try:
        # simple CSV processing: count rows
        import csv
        count = 0
        with open(file_path, newline="", encoding="utf-8") as fh:
            reader = csv.DictReader(fh)
            for _ in reader:
                count += 1

        if task:
            task.status = "success"
            task.finished_at = datetime.utcnow()
            task.error = f"imported:{count}"
            await task.save()
    except Exception as e:
        tb = traceback.format_exc()
        if task:
            task.status = "failed"
            task.error = f"{str(e)}\n\n{tb}"
            task.finished_at = datetime.utcnow()
            await task.save()


@celery_app.task(name="fastapi.process_etl_file")
def process_etl_file(task_id: int, file_path: str):
    """Celery task entrypoint. Runs an async worker that initializes Tortoise and processes the CSV.
    """
    try:
        async def _main():
            await _init_tortoise()
            await _process_file_and_update(task_id, file_path)

        asyncio.run(_main())
    except Exception:
        # if this fails at worker level, write a simple fallback to the DB file
        try:
            # best-effort: mark TaskStatus as failed
            import sqlite3
            from settings import settings
            db = getattr(settings, "DB_URL", "sqlite:///./db.sqlite3").replace("sqlite:///", "")
            conn = sqlite3.connect(db)
            cur = conn.cursor()
            cur.execute("UPDATE task_status SET status=?, error=?, finished_at=datetime('now') WHERE id=?", ("failed", "celery-exception", task_id))
            conn.commit()
            conn.close()
        except Exception:
            pass
