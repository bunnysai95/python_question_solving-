import asyncio
import csv
import traceback
from datetime import datetime
from typing import Any

from models import TaskStatus


async def _run_task_logic_from_records(records: list[dict]) -> Any:
    """Process a list of record dicts (simple example).
    Replace this with your domain logic (upserts, validation, transformations).
    """
    # simple example: count non-empty rows
    await asyncio.sleep(0)  # yield control
    imported = sum(1 for r in records if any(v not in (None, "") for v in r.values()))
    return {"imported": imported}


async def _run_task_logic_from_file(path: str) -> Any:
    """Stream-parse a CSV file at `path` and process rows in a memory-friendly way.
    Returns a summary dict.
    """
    count = 0
    # naive CSV streaming using built-in csv.reader
    with open(path, newline="", encoding="utf-8") as fh:
        reader = csv.DictReader(fh)
        batch = []
        BATCH_SZ = 500
        for row in reader:
            batch.append(row)
            count += 1
            if len(batch) >= BATCH_SZ:
                # replace with domain-specific batch processing
                await _run_task_logic_from_records(batch)
                batch = []
        if batch:
            await _run_task_logic_from_records(batch)
    return {"imported": count}


async def run_background_task(task_id: int | None, name: str, payload: dict | None = None, file_path: str | None = None) -> dict:
    """Run ETL logic and update TaskStatus. This function is async and intended for
    in-process testing or fallback; production worker uses Celery tasks (see celery_tasks.py).
    Provide either `payload` (records) or `file_path` (path to uploaded CSV).
    """
    if task_id is not None:
        task = await TaskStatus.get_or_none(id=task_id)
        if task:
            task.status = "working"
            task.started_at = datetime.utcnow()
            await task.save()
    else:
        task = await TaskStatus.create(name=name, status="working", started_at=datetime.utcnow())

    try:
        if file_path:
            result = await _run_task_logic_from_file(file_path)
        elif payload and isinstance(payload.get("records"), list):
            result = await _run_task_logic_from_records(payload.get("records", []))
        else:
            raise ValueError("no payload or file_path provided")

        if task:
            task.status = "success"
            task.finished_at = datetime.utcnow()
            await task.save()
        return {"ok": True, "result": result, "task_id": task.id if task else None}
    except Exception as exc:  # record failure and traceback
        tb = traceback.format_exc()
        if task:
            task.status = "failed"
            task.error = f"{str(exc)}\n\n{tb}"
            task.finished_at = datetime.utcnow()
            await task.save()
        return {"ok": False, "error": str(exc), "task_id": task.id if task else None}
