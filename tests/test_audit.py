#!/usr/bin/env python3
"""Tests for the field-structure baseline / drift-check API.

Covers the requirements that motivated the feature:

* baselines are built only from committed, parseable, complete lines;
* at least one line is required and every line must share identical top-level
  fields and JSON types, otherwise creation fails and leaves no record;
* the baseline freezes the resume offset -- appends and service restarts do
  not change it;
* drift checks read the target's offset and raw lines in one committed
  snapshot, so concurrent ingestion can never mix later lines in;
* missing / added / type-changed fields carry affected-line counts and the
  earliest byte offset; bad JSON lines are counted separately;
* the canonical regression: numeric ``status`` becoming a numeric string is
  detected from the raw line even though ``/events`` still returns an int
  because ingestion coerces it with int().
"""

from __future__ import annotations

import json
import os
import shutil
import sys
import tempfile
import threading
import time
import urllib.error
import urllib.request

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "collector"))

from app import audit, storage  # noqa: E402
from app.ingest import poll_once  # noqa: E402

PASS, FAIL = 0, 0


def check(name: str, cond: bool, detail: str = "") -> None:
    global PASS, FAIL
    if cond:
        PASS += 1
        print(f"  PASS  {name}")
    else:
        FAIL += 1
        print(f"  FAIL  {name}  {detail}")


def line(obj: dict) -> str:
    return json.dumps(obj, ensure_ascii=False) + "\n"


def std_line(path: str, status=200, **extra) -> str:
    obj = {"time": "2026-11-14T10:00:00Z", "path": path,
           "status": status, "method": "GET"}
    obj.update(extra)
    return line(obj)


def fresh_env():
    base = tempfile.mkdtemp(prefix="audit-test-")
    log_dir = os.path.join(base, "logs")
    db_path = os.path.join(base, "data", "warehouse.db")
    os.makedirs(log_dir)
    os.makedirs(os.path.dirname(db_path))
    conn = storage.connect(db_path)
    storage.init_db(conn)
    return base, log_dir, db_path, conn


def write_and_ingest(conn, log_dir, name, lines, poll=True):
    path = os.path.join(log_dir, name)
    with open(path, "a" if os.path.exists(path) else "w") as fh:
        for ln in lines:
            fh.write(ln)
    if poll:
        poll_once(conn, log_dir)
    return path


def identity(path):
    st = os.stat(path)
    return st.st_dev, st.st_ino


# --------------------------------------------------------------------------- #

def test_baseline_happy_path_and_frozen_offset():
    print("\n[A1] 基线创建：已提交完整行、字段类型一致、冻结续采偏移")
    base, log_dir, db_path, conn = fresh_env()
    try:
        f = write_and_ingest(conn, log_dir, "access.log",
                             [std_line(f"/p/{i}", 200) for i in range(5)])
        dev, inode = identity(f)
        size = os.path.getsize(f)

        bl = audit.create_baseline(db_path, dev=dev, inode=inode,
                                   baseline_id="bl-fixed-1")
        check("baseline id honoured", bl["id"] == "bl-fixed-1")
        check("line count covers all committed good lines",
              bl["line_count"] == 5, str(bl["line_count"]))
        check("frozen offset equals committed resume offset",
              bl["frozen_offset"] == size, f"{bl['frozen_offset']} != {size}")
        check("fields recorded with native JSON types",
              bl["fields"] == [
                  {"name": "time", "type": "string"},
                  {"name": "path", "type": "string"},
                  {"name": "status", "type": "integer"},
                  {"name": "method", "type": "string"},
              ], str(bl["fields"]))
        check("field count", bl["field_count"] == 4)

        print("\n[A2] 文件追加后续采偏移前进，但基线冻结值不变")
        write_and_ingest(conn, log_dir, "access.log",
                         [std_line("/p/new", 500)])
        state = storage.get_state(conn, dev, inode)
        bl2 = audit.get_baseline(db_path, "bl-fixed-1")
        check("live offset advanced", state["offset"] > size)
        check("frozen offset unchanged after append",
              bl2["frozen_offset"] == size)
        check("line_count unchanged after append", bl2["line_count"] == 5)
        check("fields unchanged after append", bl2["fields"] == bl["fields"])
        check("list endpoint data contains the baseline",
              [b["id"] for b in audit.list_baselines(db_path)] == ["bl-fixed-1"])
    finally:
        conn.close()
        shutil.rmtree(base)


def test_baseline_survives_restart():
    print("\n[A3] 服务重启（重新连接库）后基线仍在且冻结值不变")
    base, log_dir, db_path, conn = fresh_env()
    try:
        f = write_and_ingest(conn, log_dir, "access.log",
                             [std_line("/a", 200), std_line("/b", 404)])
        dev, inode = identity(f)
        bl = audit.create_baseline(db_path, dev=dev, inode=inode)
        frozen = bl["frozen_offset"]
        bid = bl["id"]
        conn.close()

        conn2 = storage.connect(db_path)
        bl2 = audit.get_baseline(db_path, bid)
        check("baseline persisted across reconnect",
              bl2["frozen_offset"] == frozen and bl2["line_count"] == 2)
    finally:
        try:
            conn2.close()
        except Exception:
            pass
        shutil.rmtree(base)


def test_baseline_validation_failures_leave_no_trace():
    print("\n[A4] 失败场景：未追踪/无行/结构不一致/重复 ID，且失败不留记录")
    base, log_dir, db_path, conn = fresh_env()
    try:
        # not tracked
        try:
            audit.create_baseline(db_path, dev=999, inode=999)
        except audit.AuditError as e:
            check("unknown dev/inode -> 404", e.status_code == 404, str(e))
        else:
            check("unknown dev/inode -> 404", False)

        # Tracked file with zero committed parseable lines: a file whose only
        # complete line is bad creates an ingest_state row (the offset is
        # committed) but has no log_events row to build a baseline from.
        f = os.path.join(log_dir, "onlybad.log")
        with open(f, "w") as fh:
            fh.write("{this is not json}\n")
        poll_once(conn, log_dir)
        dev, inode = identity(f)
        check("bad-line-only file is tracked with zero events",
              storage.get_state(conn, dev, inode) is not None
              and conn.execute(
                  "SELECT COUNT(*) c FROM log_events WHERE inode=?",
                  (inode,)).fetchone()["c"] == 0)
        try:
            audit.create_baseline(db_path, dev=dev, inode=inode)
        except audit.AuditError as e:
            check("no parseable complete line -> 422",
                  e.status_code == 422, str(e))
        else:
            check("no parseable complete line -> 422", False)
        check("no baseline row after failure",
              conn.execute("SELECT COUNT(*) c FROM field_baselines")
              .fetchone()["c"] == 0)

        # heterogeneous structure inside one file
        f2 = write_and_ingest(conn, log_dir, "mixed.log", [
            std_line("/a", 200),
            std_line("/b", 200, extra_field=1),          # added field
        ])
        d2, i2 = identity(f2)
        try:
            audit.create_baseline(db_path, dev=d2, inode=i2,
                                  baseline_id="bl-mixed")
        except audit.AuditError as e:
            check("non-uniform structure -> 422", e.status_code == 422, str(e))
        else:
            check("non-uniform structure -> 422", False)
        check("rejected baseline left no row",
              conn.execute(
                  "SELECT COUNT(*) c FROM field_baselines WHERE id='bl-mixed'"
              ).fetchone()["c"] == 0)

        # type mismatch inside one file also rejected
        f3 = write_and_ingest(conn, log_dir, "typemix.log", [
            std_line("/a", 200),
            std_line("/b", "200"),                       # status as string
        ])
        d3, i3 = identity(f3)
        try:
            audit.create_baseline(db_path, dev=d3, inode=i3)
        except audit.AuditError as e:
            check("mixed JSON types -> 422", e.status_code == 422, str(e))
        else:
            check("mixed JSON types -> 422", False)

        # duplicate id
        f4 = write_and_ingest(conn, log_dir, "ok.log",
                              [std_line("/a", 200)])
        d4, i4 = identity(f4)
        audit.create_baseline(db_path, dev=d4, inode=i4, baseline_id="dup")
        try:
            audit.create_baseline(db_path, dev=d4, inode=i4, baseline_id="dup")
        except audit.AuditError as e:
            check("duplicate baseline id -> 409", e.status_code == 409, str(e))
        else:
            check("duplicate baseline id -> 409", False)
        check("exactly one row for the duplicated id",
              conn.execute(
                  "SELECT COUNT(*) c FROM field_baselines WHERE id='dup'"
              ).fetchone()["c"] == 1)
    finally:
        conn.close()
        shutil.rmtree(base)


def test_drift_missing_added_type_changed_and_bad_lines():
    print("\n[B1] 核查：缺失/新增/类型改变 + 受影响行数与最早字节偏移")
    base, log_dir, db_path, conn = fresh_env()
    try:
        base_file = write_and_ingest(conn, log_dir, "v1.log", [
            std_line("/a", 200, request_id="r1", bytes_sent=12),
            std_line("/b", 404, request_id="r2", bytes_sent=0),
        ])
        bd, bi = identity(base_file)
        bl = audit.create_baseline(db_path, dev=bd, inode=bi,
                                   baseline_id="bl-v1")

        # Target file: line 1 identical; line 2 loses `method`; line 3 gains
        # `trace_id`; line 4 has status as a numeric string; plus one bad line.
        target_lines = [
            std_line("/t/1", 200, request_id="t1", bytes_sent=9),
            line({"time": "2026-11-14T10:00:01Z", "path": "/t/2",
                  "status": 500, "request_id": "t2", "bytes_sent": 3}),
            std_line("/t/3", 200, request_id="t3", bytes_sent=4,
                     trace_id="abc"),
            std_line("/t/4", "503", request_id="t4", bytes_sent=5),
            "{broken json\n",
        ]
        target = write_and_ingest(conn, log_dir, "v2.log", target_lines)
        td, ti = identity(target)

        # compute expected earliest byte offsets straight from the file
        offsets = []
        cursor = 0
        for ln in target_lines:
            offsets.append(cursor)
            cursor += len(ln.encode("utf-8"))

        report = audit.check_drift(
            db_path, baseline_id=bl["id"], target_dev=td, target_inode=ti)

        check("target metadata returned",
              report["target"]["inode"] == ti
              and report["target"]["source_code"] == "v2.log")
        check("4 parseable committed lines examined",
              report["snapshot"]["committed_lines_examined"] == 4,
              str(report["snapshot"]))
        check("bad json line counted",
              report["bad_json_lines"]["count"] == 1
              and report["bad_json_lines"]["earliest_byte_offset"] == offsets[4],
              str(report["bad_json_lines"]))

        missing = {m["field"]: m for m in report["missing_fields"]}
        check("missing method: 1 affected line, earliest offset correct",
              missing.get("method", {}).get("affected_lines") == 1
              and missing.get("method", {}).get("earliest_byte_offset")
              == offsets[1], str(missing))

        added = {a["field"]: a for a in report["added_fields"]}
        check("added trace_id: 1 affected line with observed type",
              added.get("trace_id", {}).get("affected_lines") == 1
              and added.get("trace_id", {}).get("types") == ["string"]
              and added.get("trace_id", {}).get("earliest_byte_offset")
              == offsets[2], str(added))

        changed = {(c["field"], c["types"][0]["observed_type"]): c
                   for c in report["type_changed_fields"]}
        key = ("status", "string")
        check("status integer->string detected on raw line",
              key in changed
              and changed[key]["types"][0]["baseline_type"] == "integer"
              and changed[key]["affected_lines"] == 1
              and changed[key]["earliest_byte_offset"] == offsets[3],
              str(changed))
        check("drifted flag set", report["drifted"] is True)

        # The whole point: /events still coerces "503" to integer 503.
        row = conn.execute(
            "SELECT status FROM log_events WHERE inode=? AND byte_offset=?",
            (ti, offsets[3])).fetchone()
        check("/events-style stored status is still integer despite raw drift",
              row["status"] == 503 and isinstance(row["status"], int),
              str(row["status"]))

        # Comparing a file against itself reports no drift.
        same = audit.check_drift(
            db_path, baseline_id=bl["id"], target_dev=bd, target_inode=bi)
        check("same-file check shows no drift",
              same["drifted"] is False
              and not same["missing_fields"]
              and not same["added_fields"]
              and not same["type_changed_fields"]
              and same["bad_json_lines"]["count"] == 0, str(same))

        # Unknown baseline / target.
        try:
            audit.check_drift(db_path, baseline_id="nope",
                              target_dev=td, target_inode=ti)
        except audit.AuditError as e:
            check("unknown baseline -> 404", e.status_code == 404)
        else:
            check("unknown baseline -> 404", False)
        try:
            audit.check_drift(db_path, baseline_id=bl["id"],
                              target_dev=1, target_inode=2)
        except audit.AuditError as e:
            check("unknown target -> 404", e.status_code == 404)
        else:
            check("unknown target -> 404", False)
    finally:
        conn.close()
        shutil.rmtree(base)


def test_snapshot_excludes_later_concurrent_commits():
    print("\n[B2] 同一快照：核查期间采集继续提交也不会混入后来的行")
    base, log_dir, db_path, conn = fresh_env()
    try:
        base_file = write_and_ingest(conn, log_dir, "v1.log",
                                     [std_line("/a", 200)])
        bd, bi = identity(base_file)
        bl = audit.create_baseline(db_path, dev=bd, inode=bi)

        target = write_and_ingest(conn, log_dir, "v2.log",
                                  [std_line("/x", 200)])
        td, ti = identity(target)
        offset_at_start = storage.get_state(conn, td, ti)["offset"]

        # Hold an explicit read snapshot, exactly like check_drift does.
        # In WAL mode the snapshot is established by the first read and held
        # until the transaction closes.
        conn.execute("BEGIN")
        try:
            start_offset = conn.execute(
                "SELECT offset FROM ingest_state WHERE dev=? AND inode=?",
                (td, ti)).fetchone()["offset"]
            seen = conn.execute(
                "SELECT COUNT(*) c FROM log_events "
                "WHERE dev=? AND inode=? AND byte_offset < ?",
                (td, ti, start_offset)).fetchone()["c"]
            check("snapshot sees 1 line", seen == 1, str(seen))

            # Another connection (the collector) commits more lines meanwhile.
            conn_writer = storage.connect(db_path)
            try:
                write_and_ingest(conn_writer, log_dir, "v2.log",
                                 [std_line("/y", 200), std_line("/z", 200)])
            finally:
                conn_writer.close()

            # The held snapshot still resolves the state table at its old
            # view, so the offset and line set cannot drift under us.
            offset_in_snapshot = conn.execute(
                "SELECT offset FROM ingest_state WHERE dev=? AND inode=?",
                (td, ti)).fetchone()["offset"]
            seen_still = conn.execute(
                "SELECT COUNT(*) c FROM log_events "
                "WHERE dev=? AND inode=? AND byte_offset < ?",
                (td, ti, offset_in_snapshot)).fetchone()["c"]
            fresh_offset = storage.connect(db_path).execute(
                "SELECT offset FROM ingest_state WHERE dev=? AND inode=?",
                (td, ti)).fetchone()["offset"]
            check("writer committed new data", fresh_offset > offset_at_start)
            check("open snapshot keeps original offset and 1 line",
                  seen_still == 1 and offset_in_snapshot == offset_at_start,
                  f"seen={seen_still} offset={offset_in_snapshot}")
        finally:
            conn.execute("ROLLBACK")

        # A fresh check (new snapshot) now covers all three lines.
        report = audit.check_drift(
            db_path, baseline_id=bl["id"], target_dev=td, target_inode=ti)
        check("new snapshot after commits sees all 3 lines",
              report["snapshot"]["committed_lines_examined"] == 3,
              str(report["snapshot"]))
    finally:
        conn.close()
        shutil.rmtree(base)


def test_type_taxonomy_and_null():
    print("\n[B3] JSON 类型判定：bool 与 integer 区分、null/array/object")
    base, log_dir, db_path, conn = fresh_env()
    try:
        f1 = write_and_ingest(conn, log_dir, "a.log", [line({
            "n": 1, "f": 1.5, "s": "x", "b": True, "nul": None,
            "arr": [1], "obj": {"k": 1}})])
        d, i = identity(f1)
        bl = audit.create_baseline(db_path, dev=d, inode=i)
        types = {x["name"]: x["type"] for x in bl["fields"]}
        check("native JSON types classified",
              types == {"n": "integer", "f": "number", "s": "string",
                        "b": "boolean", "nul": "null",
                        "arr": "array", "obj": "object"}, str(types))

        f2 = write_and_ingest(conn, log_dir, "b.log", [line({
            "n": 1, "f": 1.5, "s": "x", "b": False, "nul": None,
            "arr": [1], "obj": {"k": 1}})])
        d2, i2 = identity(f2)
        rep = audit.check_drift(db_path, baseline_id=bl["id"],
                                target_dev=d2, target_inode=i2)
        check("boolean stays boolean regardless of true/false",
              not rep["type_changed_fields"], str(rep["type_changed_fields"]))

        f3 = write_and_ingest(conn, log_dir, "c.log", [line({
            "n": "1", "f": 1.5, "s": "x", "b": True, "nul": 1,
            "arr": [1], "obj": {"k": 1}})])
        d3, i3 = identity(f3)
        rep2 = audit.check_drift(db_path, baseline_id=bl["id"],
                                 target_dev=d3, target_inode=i3)
        changed = {c["field"]: c["types"][0] for c in
                   rep2["type_changed_fields"]}
        check("integer->string and null->integer both reported",
              changed.get("n") == {"baseline_type": "integer",
                                   "observed_type": "string"}
              and changed.get("nul") == {"baseline_type": "null",
                                         "observed_type": "integer"},
              str(changed))
    finally:
        conn.close()
        shutil.rmtree(base)


def test_http_api_and_docs():
    print("\n[C] HTTP 端到端：基线/核查 API、错误码、/docs 与 OpenAPI")
    base = tempfile.mkdtemp(prefix="audit-api-")
    log_dir = os.path.join(base, "logs")
    db_path = os.path.join(base, "data", "warehouse.db")
    os.makedirs(log_dir)
    os.makedirs(os.path.dirname(db_path))
    os.environ["DB_PATH"] = db_path
    os.environ["LOG_DIR"] = log_dir
    os.environ["POLL_INTERVAL"] = "3600"

    import uvicorn
    from app.main import app

    cfg = uvicorn.Config(app, host="127.0.0.1", port=8777,
                         log_level="warning")
    srv = uvicorn.Server(cfg)
    t = threading.Thread(target=srv.run, daemon=True)
    t.start()
    base_url = "http://127.0.0.1:8777"
    try:
        for _ in range(100):
            try:
                urllib.request.urlopen(base_url + "/health", timeout=1)
                break
            except Exception:
                time.sleep(0.1)

        def request(method, path, body=None):
            data = json.dumps(body).encode() if body is not None else None
            req = urllib.request.Request(
                base_url + path, data=data, method=method,
                headers={"Content-Type": "application/json"})
            try:
                with urllib.request.urlopen(req, timeout=5) as r:
                    return r.status, json.loads(r.read())
            except urllib.error.HTTPError as e:
                return e.code, json.loads(e.read())

        # seed two physical files via direct poll
        from app import storage as st
        from app.ingest import poll_once
        conn = st.connect(db_path)
        f1 = os.path.join(log_dir, "v1.log")
        with open(f1, "w") as fh:
            for i in range(3):
                fh.write(std_line(f"/v1/{i}", 200))
        poll_once(conn, log_dir)
        d1, i1 = identity(f1)
        f2 = os.path.join(log_dir, "v2.log")
        with open(f2, "w") as fh:
            fh.write(std_line("/v2/0", 200))
            fh.write(std_line("/v2/1", "500"))  # numeric-string status
        poll_once(conn, log_dir)
        d2, i2 = identity(f2)

        code, bl = request("POST", "/schema-baselines",
                           {"dev": d1, "inode": i1, "baseline_id": "http-1"})
        check("create baseline 201", code == 201 and bl["id"] == "http-1"
              and bl["line_count"] == 3, str((code, bl)))

        code, listing = request("GET", "/schema-baselines")
        check("list baselines", code == 200 and
              listing["items"][0]["id"] == "http-1")

        code, one = request("GET", "/schema-baselines/http-1")
        check("get baseline", code == 200 and one["frozen_offset"] >= 0)

        code, _ = request("GET", "/schema-baselines/missing")
        check("get missing baseline -> 404", code == 404)

        code, _ = request("POST", "/schema-baselines",
                          {"dev": d1, "inode": i1, "baseline_id": "http-1"})
        check("duplicate id -> 409", code == 409)

        code, _ = request("POST", "/schema-baselines",
                          {"dev": 999, "inode": 999})
        check("untracked file -> 404", code == 404)

        code, report = request("POST",
                               "/schema-baselines/http-1/check",
                               {"target_dev": d2, "target_inode": i2})
        check("drift check 200", code == 200, str(code))
        changed = {c["field"]: c for c in report["type_changed_fields"]}
        check("HTTP report detects numeric->string status",
              "status" in changed
              and changed["status"]["types"][0]["observed_type"] == "string"
              and changed["status"]["affected_lines"] == 1,
              str(report["type_changed_fields"]))

        code, spec = request("GET", "/openapi.json")
        paths = spec["paths"]
        check("all audit endpoints published in OpenAPI",
              all(p in paths for p in (
                  "/schema-baselines",
                  "/schema-baselines/{baseline_id}",
                  "/schema-baselines/{baseline_id}/check")), str(sorted(paths)))
        check("existing query endpoints preserved",
              all(p in paths for p in
                  ("/events", "/events/{event_id}", "/malformed",
                   "/ingest/progress", "/ingest/poll-now", "/health")))

        with urllib.request.urlopen(base_url + "/docs", timeout=5) as r:
            check("/docs Swagger UI reachable", r.status == 200)
    finally:
        srv.should_exit = True
        t.join(timeout=10)
        shutil.rmtree(base)


if __name__ == "__main__":
    test_baseline_happy_path_and_frozen_offset()
    test_baseline_survives_restart()
    test_baseline_validation_failures_leave_no_trace()
    test_drift_missing_added_type_changed_and_bad_lines()
    test_snapshot_excludes_later_concurrent_commits()
    test_type_taxonomy_and_null()
    test_http_api_and_docs()
    print(f"\n==== {PASS} passed, {FAIL} failed ====")
    sys.exit(1 if FAIL else 0)
