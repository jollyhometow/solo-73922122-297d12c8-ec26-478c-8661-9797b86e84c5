"""Field-structure baselines and schema drift checks.

Why this module exists
======================

The pipeline normalises selected fields when ingesting (for example
``int(status)`` in :func:`app.ingest._normalise`), so ``/events`` can return
an integer status even when the *raw* JSON line carried ``"status":"500"``.
After the nginx log format changes, that conversion would hide a structural
drift in the original fields. The audit API therefore never looks at the
normalised columns -- it re-parses the verbatim ``raw_line`` text and
compares the top-level field names and their native JSON types.

Snapshot semantics
==================

* **Baseline creation** reads only rows already committed in
  ``log_events`` (every stored row came from a complete, newline-terminated,
  parseable line) and freezes the file's then-current committed resume
  offset in ``ingest_state``. Everything happens inside one
  ``BEGIN IMMEDIATE`` transaction: any validation failure rolls back and
  leaves no record.
* **Drift check** opens one read transaction and keeps it open for the whole
  comparison. Under WAL the transaction holds a single snapshot -- the
  collector committing more rows concurrently cannot mix later lines into
  the result. The target file's resume offset and the raw lines are both
  read inside that same transaction, so they are one committed snapshot.
  Baselines' frozen offsets are immutable, so appends and service restarts
  can never change what an existing baseline covers.

Concurrency
===========

Audit operations open their **own short-lived SQLite connection** instead of
reusing the application-wide connection. A connection can carry only one
transaction at a time, and the shared connection also serves plain SELECT
endpoints whose autocommit deferred transactions could collide with an
explicit ``BEGIN``. Separate connections remove that coupling entirely; WAL
allows the collector to keep writing while an audit read transaction holds a
snapshot. :func:`storage.connect` applies the same WAL / ``synchronous``
/ busy-timeout pragmas.
"""

from __future__ import annotations

import json
import logging
import sqlite3
import uuid
from contextlib import closing
from dataclasses import dataclass
from typing import Any, Optional

from . import storage

log = logging.getLogger("collector.audit")


# Public error type: the API layer maps these onto HTTP status codes.
class AuditError(Exception):
    def __init__(self, status_code: int, detail: str):
        super().__init__(detail)
        self.status_code = status_code
        self.detail = detail


def json_type(value: Any) -> str:
    """Name the JSON type of a parsed value, using JSON's vocabulary.

    ``bool`` is checked before ``int`` because in Python ``True`` is an
    ``int`` -- JSON itself distinguishes ``true``/``false`` from numbers.
    """
    if value is None:
        return "null"
    if isinstance(value, bool):
        return "boolean"
    if isinstance(value, int):
        return "integer"
    if isinstance(value, float):
        return "number"
    if isinstance(value, str):
        return "string"
    if isinstance(value, list):
        return "array"
    if isinstance(value, dict):
        return "object"
    return "unknown"


@dataclass(frozen=True)
class FieldSpec:
    name: str
    json_type: str

    def as_dict(self) -> dict[str, str]:
        return {"name": self.name, "type": self.json_type}


def _structure_of(raw_line: str) -> list[FieldSpec]:
    """Top-level fields, in first-seen order, with their native JSON types."""
    obj = json.loads(raw_line)
    if not isinstance(obj, dict):
        raise ValueError("top-level JSON value is not an object")
    return [FieldSpec(name=k, json_type=json_type(v)) for k, v in obj.items()]


def _new_id() -> str:
    return "bl_" + uuid.uuid4().hex


def _baseline_dict(row: sqlite3.Row) -> dict[str, Any]:
    return {
        "id": row["id"],
        "dev": row["dev"],
        "inode": row["inode"],
        "source_code": row["source_code"],
        "frozen_offset": row["frozen_offset"],
        "field_count": row["field_count"],
        "line_count": row["line_count"],
        "fields": json.loads(row["fields_json"]),
        "created_at": row["created_at"],
    }


def _empty_change(name: str) -> dict[str, Any]:
    return {
        "field": name,
        "affected_lines": 0,
        "earliest_byte_offset": None,
        "types": [],
    }


# --------------------------------------------------------------------------- #
# Baseline creation
# --------------------------------------------------------------------------- #

def create_baseline(
    db_path: str,
    *,
    dev: int,
    inode: int,
    baseline_id: Optional[str] = None,
) -> dict[str, Any]:
    """Create a field-structure baseline for a tracked log file.

    The baseline is built only from committed, parseable, complete lines of
    the given physical file whose byte position precedes the committed
    resume offset. Validation rules:

    * the file must be tracked in ``ingest_state``;
    * at least one committed, parseable line must exist;
    * every line must expose exactly the same top-level field set, in any
      order, and every shared field must have the same JSON type.

    On any failure the surrounding transaction is rolled back, so no
    baseline row (or any other write) survives.
    """
    bid = baseline_id or _new_id()
    with closing(storage.connect(db_path)) as conn:
        with storage.transaction(conn):
            state = conn.execute(
                "SELECT * FROM ingest_state WHERE dev = ? AND inode = ?",
                (dev, inode),
            ).fetchone()
            if state is None:
                raise AuditError(
                    404,
                    f"no tracked log file for dev={dev} inode={inode}; "
                    "the file must have been ingested at least once",
                )
            frozen_offset = state["offset"]

            # Only committed rows ever exist in log_events; the offset
            # predicate restricts the view to the byte range covered now.
            rows = conn.execute(
                """SELECT byte_offset, raw_line
                     FROM log_events
                    WHERE dev = ? AND inode = ? AND byte_offset < ?
                    ORDER BY byte_offset""",
                (dev, inode, frozen_offset),
            ).fetchall()
            if not rows:
                raise AuditError(
                    422,
                    f"file {state['source_code']} (dev={dev} inode={inode}) "
                    "has no committed, parseable line; a baseline needs at "
                    "least one complete line",
                )

            reference: Optional[list[FieldSpec]] = None
            ref_map: Optional[dict[str, str]] = None
            for row in rows:
                try:
                    fields = _structure_of(row["raw_line"])
                except (json.JSONDecodeError, ValueError) as exc:
                    # Stored rows are parseable by construction; this keeps
                    # the contract explicit if data ever predates it.
                    raise AuditError(
                        422,
                        f"line at byte_offset={row['byte_offset']} cannot be "
                        f"parsed as a JSON object: {exc}",
                    )
                current = {f.name: f.json_type for f in fields}
                if reference is None:
                    reference, ref_map = fields, current
                    continue
                assert ref_map is not None
                if current != ref_map:
                    ref_names = set(ref_map)
                    cur_names = set(current)
                    missing = sorted(ref_names - cur_names)
                    added = sorted(cur_names - ref_names)
                    changed = sorted(
                        name for name in ref_names & cur_names
                        if ref_map[name] != current[name]
                    )
                    details = []
                    if missing:
                        details.append(f"missing fields {missing}")
                    if added:
                        details.append(f"extra fields {added}")
                    if changed:
                        details.append(
                            "type changes "
                            + str([
                                f"{name}: {ref_map[name]} -> {current[name]}"
                                for name in changed
                            ])
                        )
                    raise AuditError(
                        422,
                        "existing lines of this file already differ in field "
                        f"structure (first mismatch at byte_offset="
                        f"{row['byte_offset']}): " + "; ".join(details),
                    )

            assert reference is not None
            try:
                conn.execute(
                    """INSERT INTO field_baselines
                           (id, dev, inode, source_code, frozen_offset,
                            field_count, line_count, fields_json)
                       VALUES (?, ?, ?, ?, ?, ?, ?, ?)""",
                    (
                        bid, dev, inode, state["source_code"], frozen_offset,
                        len(reference), len(rows),
                        json.dumps([f.as_dict() for f in reference]),
                    ),
                )
            except sqlite3.IntegrityError:
                raise AuditError(
                    409, f"baseline id {bid!r} already exists"
                )

            return _baseline_dict(conn.execute(
                "SELECT * FROM field_baselines WHERE id = ?", (bid,)
            ).fetchone())


def get_baseline(db_path: str, baseline_id: str) -> dict[str, Any]:
    with closing(storage.connect(db_path)) as conn:
        row = conn.execute(
            "SELECT * FROM field_baselines WHERE id = ?", (baseline_id,)
        ).fetchone()
    if row is None:
        raise AuditError(404, f"baseline {baseline_id!r} not found")
    return _baseline_dict(row)


def list_baselines(db_path: str) -> list[dict[str, Any]]:
    with closing(storage.connect(db_path)) as conn:
        rows = conn.execute(
            "SELECT * FROM field_baselines ORDER BY created_at, id"
        ).fetchall()
    return [_baseline_dict(row) for row in rows]


# --------------------------------------------------------------------------- #
# Drift check against another physical file
# --------------------------------------------------------------------------- #

def check_drift(
    db_path: str,
    *,
    baseline_id: str,
    target_dev: int,
    target_inode: int,
) -> dict[str, Any]:
    """Compare another physical file against a baseline, in one snapshot.

    The target's resume offset and all raw lines are read inside a single
    read transaction on a dedicated connection, so concurrent ingestion
    cannot interleave later lines. Only complete committed lines
    (``byte_offset < target offset``) are examined; malformed lines are
    counted but never parsed as structure.
    """
    with closing(storage.connect(db_path)) as conn:
        # Snapshot isolation: a deferred read transaction under WAL sees a
        # stable view from its first read until it closes, even while the
        # collector commits on its own connection.
        conn.execute("BEGIN")
        try:
            brow = conn.execute(
                "SELECT * FROM field_baselines WHERE id = ?", (baseline_id,)
            ).fetchone()
            if brow is None:
                raise AuditError(404, f"baseline {baseline_id!r} not found")

            tstate = conn.execute(
                "SELECT * FROM ingest_state WHERE dev = ? AND inode = ?",
                (target_dev, target_inode),
            ).fetchone()
            if tstate is None:
                raise AuditError(
                    404,
                    f"no tracked target file for dev={target_dev} "
                    f"inode={target_inode}",
                )
            target_offset = tstate["offset"]

            baseline_fields = json.loads(brow["fields_json"])
            base_map: dict[str, str] = {
                f["name"]: f["type"] for f in baseline_fields
            }

            # Aggregations keyed so every reported item carries affected-line
            # counts and the earliest byte position where it occurs.
            missing: dict[str, dict[str, Any]] = {
                name: _empty_change(name) for name in base_map
            }
            added: dict[str, dict[str, Any]] = {}
            # type changes: key (field, observed type)
            changed: dict[tuple[str, str], dict[str, Any]] = {}

            good_lines = 0
            earliest_good_offset: Optional[int] = None

            line_rows = conn.execute(
                """SELECT byte_offset, raw_line
                     FROM log_events
                    WHERE dev = ? AND inode = ? AND byte_offset < ?
                    ORDER BY byte_offset""",
                (target_dev, target_inode, target_offset),
            ).fetchall()

            # Bad JSON lines quarantined for this exact committed range.
            # They are structure-less and therefore counted separately.
            bad_rows = conn.execute(
                """SELECT byte_offset
                     FROM malformed_lines
                    WHERE dev = ? AND inode = ? AND byte_offset < ?
                    ORDER BY byte_offset""",
                (target_dev, target_inode, target_offset),
            ).fetchall()
            bad_lines = len(bad_rows)
            earliest_bad_offset = (
                bad_rows[0]["byte_offset"] if bad_rows else None
            )

            for line_row in line_rows:
                offset = line_row["byte_offset"]
                try:
                    current_specs = _structure_of(line_row["raw_line"])
                except (json.JSONDecodeError, ValueError):
                    # Defensive: committed log_events rows are always parseable.
                    bad_lines += 1
                    if (earliest_bad_offset is None
                            or offset < earliest_bad_offset):
                        earliest_bad_offset = offset
                    continue

                good_lines += 1
                if earliest_good_offset is None:
                    earliest_good_offset = offset
                current = {f.name: f.json_type for f in current_specs}

                # Fields the baseline knows but this line does not provide.
                for name in base_map:
                    if name not in current:
                        item = missing[name]
                        item["affected_lines"] += 1
                        if (item["earliest_byte_offset"] is None
                                or offset < item["earliest_byte_offset"]):
                            item["earliest_byte_offset"] = offset

                # Fields the line has but the baseline never had.
                for name, observed in current.items():
                    if name not in base_map:
                        item = added.get(name)
                        if item is None:
                            item = _empty_change(name)
                            added[name] = item
                        item["affected_lines"] += 1
                        if (item["earliest_byte_offset"] is None
                                or offset < item["earliest_byte_offset"]):
                            item["earliest_byte_offset"] = offset
                        if observed not in item["types"]:
                            item["types"].append(observed)

                # Shared fields whose native JSON type differs. The canonical
                # example: numeric status became a numeric string -- /events
                # still returns an integer after int() coercion, but the raw
                # line's drift is detected here.
                for name, observed in current.items():
                    if name in base_map and base_map[name] != observed:
                        key = (name, observed)
                        item = changed.get(key)
                        if item is None:
                            item = _empty_change(name)
                            item["types"] = [
                                {"baseline_type": base_map[name],
                                 "observed_type": observed}
                            ]
                            changed[key] = item
                        item["affected_lines"] += 1
                        if (item["earliest_byte_offset"] is None
                                or offset < item["earliest_byte_offset"]):
                            item["earliest_byte_offset"] = offset

            missing_out = [
                missing[name] for name in sorted(missing)
                if missing[name]["affected_lines"]
            ]
            added_out = [added[name] for name in sorted(added)]
            changed_out = [
                changed[key]
                for key in sorted(changed, key=lambda k: (k[0], k[1]))
            ]
            drifted = bool(
                missing_out or added_out or changed_out or bad_lines
            )

            return {
                "baseline_id": brow["id"],
                "baseline": _baseline_dict(brow),
                "target": {
                    "dev": target_dev,
                    "inode": target_inode,
                    "source_code": tstate["source_code"],
                    "path": tstate["path"],
                    "committed_offset": target_offset,
                },
                "snapshot": {
                    "committed_lines_examined": good_lines,
                    "earliest_byte_offset": earliest_good_offset,
                },
                "missing_fields": missing_out,
                "added_fields": added_out,
                "type_changed_fields": changed_out,
                "bad_json_lines": {
                    "count": bad_lines,
                    "earliest_byte_offset": earliest_bad_offset,
                },
                "drifted": drifted,
            }
        finally:
            # No writes were made; rollback simply closes the read transaction.
            conn.execute("ROLLBACK")
