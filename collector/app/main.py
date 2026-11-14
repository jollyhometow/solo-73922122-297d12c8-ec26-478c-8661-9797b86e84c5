"""FastAPI service: continuous ingestion + query API.

Endpoints
---------
GET  /health                 liveness + counters
GET  /ingest/progress        per-file collection progress (offset/size)
POST /ingest/poll-now        force an immediate collection pass
GET  /events                 paginated events, filter by path / status /
                             status range / source code
GET  /events/{event_id}      single event with the raw JSON line
GET  /malformed              quarantined lines that failed JSON parsing

Field-structure audit (raw-line based, see app/audit.py)
--------------------------------------------------------
POST /schema-baselines            create a baseline for a tracked file
GET  /schema-baselines            list baselines
GET  /schema-baselines/{id}       get one baseline (frozen offset + fields)
POST /schema-baselines/{id}/check compare another physical file in one
                                  committed snapshot
"""

from __future__ import annotations

import logging
import os
import threading
from contextlib import asynccontextmanager
from typing import Optional

from fastapi import FastAPI, HTTPException, Query
from pydantic import BaseModel, Field

from . import audit, storage
from .ingest import CollectorLoop, poll_once

logging.basicConfig(
    level=os.getenv("LOG_LEVEL", "INFO"),
    format="%(asctime)s %(levelname)s %(name)s: %(message)s",
)
log = logging.getLogger("collector.api")

DB_PATH = os.getenv("DB_PATH", "/data/warehouse.db")
LOG_DIR = os.getenv("LOG_DIR", "/logs")
POLL_INTERVAL = float(os.getenv("POLL_INTERVAL", "0.5"))


class Health(BaseModel):
    status: str
    log_dir: str
    db_path: str
    poll_interval_seconds: float
    collector_running: bool
    total_polls: int
    total_ingested: int
    counts: dict


class FileProgress(BaseModel):
    source_code: str
    dev: int
    inode: int
    path: str
    offset: int
    size: Optional[int] = None
    complete: bool
    last_seen: str


class ProgressResponse(BaseModel):
    collector_running: bool
    total_polls: int
    total_ingested: int
    last_poll: Optional[dict] = None
    counts: dict
    files: list[FileProgress]


class Event(BaseModel):
    id: int
    ts: Optional[str]
    path: Optional[str]
    status: Optional[int]
    method: Optional[str]
    source_code: str
    dev: int
    inode: int
    byte_offset: int
    raw_line: str
    ingested_at: str


class EventPage(BaseModel):
    total: int
    limit: int
    offset: int
    next_offset: Optional[int]
    items: list[Event]


state: dict = {}


@asynccontextmanager
async def lifespan(app: FastAPI):
    os.makedirs(os.path.dirname(DB_PATH), exist_ok=True)
    conn = storage.connect(DB_PATH)
    storage.init_db(conn)

    loop = CollectorLoop(conn, LOG_DIR, interval=POLL_INTERVAL)
    thread = threading.Thread(target=loop.run_forever, name="collector",
                              daemon=True)
    thread.start()
    state.update(conn=conn, loop=loop, thread=thread)
    log.info("API ready; watching %s, db %s", LOG_DIR, DB_PATH)
    try:
        yield
    finally:
        log.info("shutdown requested, stopping collector loop...")
        loop.request_stop()
        thread.join(timeout=10)
        # A checkpoint persists WAL frames into the main db file on exit.
        try:
            conn.execute("PRAGMA wal_checkpoint(TRUNCATE)")
        except Exception:
            log.exception("checkpoint on shutdown failed")
        conn.close()
        log.info("shutdown complete")


app = FastAPI(
    title="Nginx 访问日志入仓 API",
    description=(
        "持续采集共享只读目录中的 JSON 行 Nginx 日志，按 (设备, inode) "
        "识别文件、行与续采偏移同事务提交，支持按路径与状态码分页查询故障请求。"
    ),
    version="1.0.0",
    lifespan=lifespan,
)


def _conn():
    return state["conn"]


@app.get("/health", response_model=Health, tags=["meta"])
def health():
    loop: CollectorLoop = state["loop"]
    return Health(
        status="ok",
        log_dir=LOG_DIR,
        db_path=DB_PATH,
        poll_interval_seconds=POLL_INTERVAL,
        collector_running=threading.main_thread().is_alive() and not loop._stop,
        total_polls=loop.total_polls,
        total_ingested=loop.total_ingested,
        counts=storage.counts(_conn()),
    )


@app.get("/ingest/progress", response_model=ProgressResponse, tags=["ingest"])
def progress():
    conn = _conn()
    loop: CollectorLoop = state["loop"]
    rows = storage.all_states(conn)
    # Current sizes from the live directory, read-only.
    current: dict[tuple[int, int], int] = {}
    try:
        for name in os.listdir(LOG_DIR):
            try:
                st = os.stat(os.path.join(LOG_DIR, name))
            except FileNotFoundError:
                continue
            current[(st.st_dev, st.st_ino)] = st.st_size
    except FileNotFoundError:
        pass

    files = []
    for row in rows:
        size = current.get((row["dev"], row["inode"]))
        files.append(FileProgress(
            source_code=row["source_code"],
            dev=row["dev"],
            inode=row["inode"],
            path=row["path"],
            offset=row["offset"],
            size=size,
            complete=(size is not None and row["offset"] >= size),
            last_seen=row["last_seen"],
        ))
    return ProgressResponse(
        collector_running=not loop._stop,
        total_polls=loop.total_polls,
        total_ingested=loop.total_ingested,
        last_poll=loop.last_result,
        counts=storage.counts(conn),
        files=files,
    )


@app.post("/ingest/poll-now", tags=["ingest"])
def poll_now():
    """Force an immediate scan/ingest pass (useful for tests/demos)."""
    loop: CollectorLoop = state["loop"]
    with loop.write_lock:
        return poll_once(_conn(), LOG_DIR)


@app.get("/events", response_model=EventPage, tags=["query"])
def list_events(
    path: Optional[str] = Query(None, description="精确路径，例如 /api/orders"),
    status: Optional[int] = Query(None, description="精确状态码，例如 500"),
    status_min: Optional[int] = Query(None, description="状态码下界，例如 400"),
    status_max: Optional[int] = Query(None, description="状态码上界，例如 599"),
    source_code: Optional[str] = Query(None, description="来源文件代号，例如 access.log.20261113T100000.log"),
    limit: int = Query(100, ge=1, le=1000),
    offset: int = Query(0, ge=0),
):
    """按路径、状态码分页取数。查故障请求示例：
    ``/events?status_min=500&status_max=599&limit=50``。"""
    return storage.query_events(
        _conn(),
        path=path,
        status=status,
        status_min=status_min,
        status_max=status_max,
        source_code=source_code,
        limit=limit,
        offset=offset,
    )


@app.get("/events/{event_id}", response_model=Event, tags=["query"])
def get_event(event_id: int):
    row = _conn().execute(
        "SELECT * FROM log_events WHERE id = ?", (event_id,)
    ).fetchone()
    if row is None:
        raise HTTPException(status_code=404, detail="event not found")
    return dict(row)


@app.get("/malformed", tags=["query"])
def list_malformed(limit: int = Query(100, ge=1, le=1000),
                   offset: int = Query(0, ge=0)):
    conn = _conn()
    total = conn.execute(
        "SELECT COUNT(*) c FROM malformed_lines"
    ).fetchone()["c"]
    rows = conn.execute(
        "SELECT * FROM malformed_lines ORDER BY id LIMIT ? OFFSET ?",
        (limit, offset),
    ).fetchall()
    return {"total": total, "limit": limit, "offset": offset,
            "items": [dict(r) for r in rows]}


# --------------------------------------------------------------------------- #
# Field-structure audit: baselines + drift checks against raw JSON lines.
# --------------------------------------------------------------------------- #

class BaselineCreateRequest(BaseModel):
    dev: int = Field(..., description="设备号 st_dev，与 /ingest/progress 一致")
    inode: int = Field(..., description="inode 号 st_ino，与 /ingest/progress 一致")
    baseline_id: Optional[str] = Field(
        None,
        description="用户指定的基线 ID；留空则自动生成。重复时返回 409。",
        min_length=1,
        max_length=128,
    )


class DriftCheckRequest(BaseModel):
    target_dev: int = Field(..., description="另一物理文件的设备号 st_dev")
    target_inode: int = Field(..., description="另一物理文件的 inode st_ino")


def _audit_guard(fn, *args, **kwargs):
    """Translate domain errors onto FastAPI HTTP responses."""
    try:
        return fn(*args, **kwargs)
    except audit.AuditError as exc:
        raise HTTPException(status_code=exc.status_code, detail=exc.detail)


@app.post("/schema-baselines", tags=["schema-audit"], status_code=201)
def create_schema_baseline(req: BaselineCreateRequest):
    """为已追踪日志建立字段结构基线。

    仅使用创建时**已提交、可解析的完整行**：至少一行，且每行顶层字段名与
    JSON 类型完全一致，否则 422 且不留任何记录。基线保存后冻结当时的续采
    偏移，文件追加或服务重启都不会改变它。
    """
    return _audit_guard(
        audit.create_baseline,
        DB_PATH,
        dev=req.dev,
        inode=req.inode,
        baseline_id=req.baseline_id,
    )


@app.get("/schema-baselines", tags=["schema-audit"])
def list_schema_baselines():
    return {"items": audit.list_baselines(DB_PATH)}


@app.get("/schema-baselines/{baseline_id}", tags=["schema-audit"])
def get_schema_baseline(baseline_id: str):
    return _audit_guard(audit.get_baseline, DB_PATH, baseline_id)


@app.post("/schema-baselines/{baseline_id}/check", tags=["schema-audit"])
def check_schema_drift(baseline_id: str, req: DriftCheckRequest):
    """在同一已提交快照内核查另一物理文件。

    返回缺失字段、新增字段、类型改变字段各自的受影响行数与最早字节位置，
    并注明坏 JSON 行数。采集持续提交时不会混入后来的行（单一只读事务持有
    WAL 快照）。原始 ``status`` 由数字变成数字字符串时，即使 ``/events``
    仍返回整数，这里也会作为类型改变检出。
    """
    return _audit_guard(
        audit.check_drift,
        DB_PATH,
        baseline_id=baseline_id,
        target_dev=req.target_dev,
        target_inode=req.target_inode,
    )
