"""프로젝트 상태 저장소.

에이전트는 상태를 두지 않는다(stateless). 작업 상태와 메시지 기록은 모두 이 저장소에
둔다. 상태 전이는 비교 후 교환(compare-and-set)으로만 하여, 같은 에이전트의 복제본이
여럿이어도 하위 작업이 두 번 배정되지 않게 한다.

구현 두 가지:
- InMemoryStateStore: 시험·개발용
- PostgresStateStore: PostgreSQL (운영 환경에서는 PostGIS 확장 이미지 사용)
"""

from __future__ import annotations

import asyncio
import json
from datetime import UTC, datetime
from enum import StrEnum
from typing import Any, Protocol

from pydantic import BaseModel, Field

from .messages import DataAssetRef, Envelope


class TaskStatus(StrEnum):
    """gai:TaskStatus 개체의 지역명과 같다."""

    REQUESTED = "Requested"
    ACCEPTED = "Accepted"
    PLANNED = "Planned"
    RUNNING = "Running"
    SUCCEEDED = "Succeeded"
    FAILED = "Failed"
    REJECTED = "Rejected"


TERMINAL = {TaskStatus.SUCCEEDED, TaskStatus.FAILED, TaskStatus.REJECTED}


def _now() -> datetime:
    return datetime.now(UTC)


class TaskRecord(BaseModel):
    task_id: str
    parent_id: str | None = None
    project_id: str
    conversation_id: str
    task_type: str
    status: TaskStatus
    agent: str = Field(description="담당 에이전트 식별자")
    requester: str | None = Field(default=None, description="상위 작업만: 결과를 돌려받을 에이전트")
    request_message_id: str | None = Field(default=None, description="이 작업을 배정한 요청 메시지")
    depends_on: list[str] = Field(default_factory=list)
    input_assets: list[DataAssetRef] = Field(default_factory=list, description="요청에 직접 들어온 입력")
    input_from: list[str] = Field(default_factory=list, description="산출을 입력으로 받을 선행 작업")
    outputs: list[DataAssetRef] = Field(default_factory=list)
    metrics: dict[str, float] = Field(default_factory=dict)
    request: dict[str, Any] = Field(default_factory=dict, description="원 요청 본문 (domain, parameters 등)")
    detail: str = ""
    created_at: datetime = Field(default_factory=_now)
    updated_at: datetime = Field(default_factory=_now)


class StateStore(Protocol):
    async def create_tasks(self, records: list[TaskRecord]) -> None: ...

    async def get_task(self, task_id: str) -> TaskRecord | None: ...

    async def list_children(self, parent_id: str) -> list[TaskRecord]: ...

    async def transition(
        self, task_id: str, allowed_from: set[TaskStatus], to: TaskStatus, **changes: Any
    ) -> TaskRecord | None:
        """현재 상태가 allowed_from 에 있을 때만 바꾼다. 바뀌지 않으면 None."""
        ...

    async def record_message(self, envelope: Envelope) -> None: ...

    async def list_tasks(self, project_id: str | None = None, limit: int = 50) -> list[TaskRecord]:
        """최상위 작업(parent_id 없음), 최근 순."""
        ...

    async def list_messages(self, conversation_id: str) -> list[Envelope]:
        """대화 하나의 메시지 기록, 시간 순."""
        ...

    async def close(self) -> None: ...


class InMemoryStateStore:
    def __init__(self) -> None:
        self._tasks: dict[str, TaskRecord] = {}
        self.messages: list[Envelope] = []
        self._lock = asyncio.Lock()

    async def create_tasks(self, records: list[TaskRecord]) -> None:
        async with self._lock:
            for r in records:
                if r.task_id in self._tasks:
                    raise KeyError(f"이미 있는 작업: {r.task_id}")
            for r in records:
                self._tasks[r.task_id] = r.model_copy(deep=True)

    async def get_task(self, task_id: str) -> TaskRecord | None:
        r = self._tasks.get(task_id)
        return r.model_copy(deep=True) if r else None

    async def list_children(self, parent_id: str) -> list[TaskRecord]:
        children = [r for r in self._tasks.values() if r.parent_id == parent_id]
        return [r.model_copy(deep=True) for r in sorted(children, key=lambda r: r.created_at)]

    async def transition(
        self, task_id: str, allowed_from: set[TaskStatus], to: TaskStatus, **changes: Any
    ) -> TaskRecord | None:
        async with self._lock:
            r = self._tasks.get(task_id)
            if r is None or r.status not in allowed_from:
                return None
            updated = r.model_copy(update={**changes, "status": to, "updated_at": _now()}, deep=True)
            self._tasks[task_id] = TaskRecord.model_validate(updated.model_dump())
            return self._tasks[task_id].model_copy(deep=True)

    async def record_message(self, envelope: Envelope) -> None:
        if any(m.message_id == envelope.message_id for m in self.messages[-200:]):
            return
        self.messages.append(envelope)

    async def list_tasks(self, project_id: str | None = None, limit: int = 50) -> list[TaskRecord]:
        tops = [
            r
            for r in self._tasks.values()
            if r.parent_id is None and (project_id is None or r.project_id == project_id)
        ]
        tops.sort(key=lambda r: r.created_at, reverse=True)
        return [r.model_copy(deep=True) for r in tops[:limit]]

    async def list_messages(self, conversation_id: str) -> list[Envelope]:
        return [m for m in self.messages if m.conversation_id == conversation_id]

    async def close(self) -> None:
        return None


SCHEMA_SQL = """
CREATE TABLE IF NOT EXISTS geoai_task (
    task_id         TEXT PRIMARY KEY,
    parent_id       TEXT REFERENCES geoai_task(task_id),
    project_id      TEXT NOT NULL,
    conversation_id TEXT NOT NULL,
    task_type       TEXT NOT NULL,
    status          TEXT NOT NULL,
    agent           TEXT NOT NULL,
    record          JSONB NOT NULL,
    created_at      TIMESTAMPTZ NOT NULL,
    updated_at      TIMESTAMPTZ NOT NULL
);
CREATE INDEX IF NOT EXISTS geoai_task_parent_idx ON geoai_task(parent_id);
CREATE INDEX IF NOT EXISTS geoai_task_project_idx ON geoai_task(project_id);

CREATE TABLE IF NOT EXISTS geoai_message (
    message_id      TEXT PRIMARY KEY,
    conversation_id TEXT NOT NULL,
    project_id      TEXT NOT NULL,
    performative    TEXT NOT NULL,
    sender          TEXT NOT NULL,
    recipient       TEXT,
    task_id         TEXT,
    envelope        JSONB NOT NULL,
    created_at      TIMESTAMPTZ NOT NULL
);
CREATE INDEX IF NOT EXISTS geoai_message_conversation_idx ON geoai_message(conversation_id);
"""


class PostgresStateStore:
    def __init__(self, dsn: str) -> None:
        from psycopg_pool import AsyncConnectionPool

        self._pool = AsyncConnectionPool(dsn, min_size=1, max_size=5, open=False)

    async def open(self) -> PostgresStateStore:
        await self._pool.open(wait=True)
        async with self._pool.connection() as conn:
            await conn.execute(SCHEMA_SQL)
        return self

    @staticmethod
    def _row(r: TaskRecord) -> tuple[Any, ...]:
        return (
            r.task_id,
            r.parent_id,
            r.project_id,
            r.conversation_id,
            r.task_type,
            r.status.value,
            r.agent,
            r.model_dump_json(),
            r.created_at,
            r.updated_at,
        )

    async def create_tasks(self, records: list[TaskRecord]) -> None:
        async with self._pool.connection() as conn, conn.transaction(), conn.cursor() as cur:
            await cur.executemany(
                "INSERT INTO geoai_task (task_id, parent_id, project_id, conversation_id, task_type,"
                " status, agent, record, created_at, updated_at)"
                " VALUES (%s, %s, %s, %s, %s, %s, %s, %s::jsonb, %s, %s)",
                [self._row(r) for r in records],
            )

    async def get_task(self, task_id: str) -> TaskRecord | None:
        async with self._pool.connection() as conn:
            cur = await conn.execute("SELECT record FROM geoai_task WHERE task_id = %s", (task_id,))
            row = await cur.fetchone()
        return TaskRecord.model_validate(row[0]) if row else None

    async def list_children(self, parent_id: str) -> list[TaskRecord]:
        async with self._pool.connection() as conn:
            cur = await conn.execute(
                "SELECT record FROM geoai_task WHERE parent_id = %s ORDER BY created_at, task_id", (parent_id,)
            )
            rows = await cur.fetchall()
        return [TaskRecord.model_validate(r[0]) for r in rows]

    async def transition(
        self, task_id: str, allowed_from: set[TaskStatus], to: TaskStatus, **changes: Any
    ) -> TaskRecord | None:
        async with self._pool.connection() as conn, conn.transaction():
            cur = await conn.execute(
                "SELECT record FROM geoai_task WHERE task_id = %s AND status = ANY(%s) FOR UPDATE",
                (task_id, [s.value for s in allowed_from]),
            )
            row = await cur.fetchone()
            if row is None:
                return None
            current = TaskRecord.model_validate(row[0])
            updated = TaskRecord.model_validate({**current.model_dump(), **changes, "status": to, "updated_at": _now()})
            await conn.execute(
                "UPDATE geoai_task SET status = %s, record = %s::jsonb, updated_at = %s WHERE task_id = %s",
                (updated.status.value, updated.model_dump_json(), updated.updated_at, task_id),
            )
            return updated

    async def record_message(self, envelope: Envelope) -> None:
        async with self._pool.connection() as conn:
            await conn.execute(
                "INSERT INTO geoai_message (message_id, conversation_id, project_id, performative, sender,"
                " recipient, task_id, envelope, created_at) VALUES (%s, %s, %s, %s, %s, %s, %s, %s::jsonb, %s)"
                " ON CONFLICT (message_id) DO NOTHING",
                (
                    envelope.message_id,
                    envelope.conversation_id,
                    envelope.project_id,
                    envelope.performative.value,
                    envelope.sender,
                    envelope.recipient,
                    envelope.task_id,
                    json.dumps(envelope.model_dump(mode="json"), ensure_ascii=False),
                    envelope.created_at,
                ),
            )

    async def list_tasks(self, project_id: str | None = None, limit: int = 50) -> list[TaskRecord]:
        sql = "SELECT record FROM geoai_task WHERE parent_id IS NULL"
        args: list[Any] = []
        if project_id is not None:
            sql += " AND project_id = %s"
            args.append(project_id)
        sql += " ORDER BY created_at DESC LIMIT %s"
        args.append(limit)
        async with self._pool.connection() as conn:
            cur = await conn.execute(sql, args)
            rows = await cur.fetchall()
        return [TaskRecord.model_validate(r[0]) for r in rows]

    async def list_messages(self, conversation_id: str) -> list[Envelope]:
        async with self._pool.connection() as conn:
            cur = await conn.execute(
                "SELECT envelope FROM geoai_message WHERE conversation_id = %s ORDER BY created_at, message_id",
                (conversation_id,),
            )
            rows = await cur.fetchall()
        return [Envelope.model_validate(r[0]) for r in rows]

    async def close(self) -> None:
        await self._pool.close()
