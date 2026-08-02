"""Thin adapter that gives a `libsql` remote Turso connection the same shape as
`sqlite3.Connection`/`sqlite3.Row` so `store.py` can use either backend unmodified.
"""

from __future__ import annotations

from typing import Any, Iterator, Sequence


class Row:
    """sqlite3.Row-alike supporting access by column name or index."""

    __slots__ = ("_values", "_columns")

    def __init__(self, values: Sequence[Any], columns: tuple[str, ...]) -> None:
        self._values = tuple(values)
        self._columns = columns

    def __getitem__(self, key: Any) -> Any:
        if isinstance(key, str):
            return self._values[self._columns.index(key)]
        return self._values[key]

    def __iter__(self) -> Iterator[Any]:
        return iter(self._values)

    def __len__(self) -> int:
        return len(self._values)

    def __repr__(self) -> str:
        return f"Row{self._values!r}"

    def keys(self) -> tuple[str, ...]:
        return self._columns


class CursorAdapter:
    """Wraps a raw `libsql.Cursor`, exposing sqlite3-style row access and iteration."""

    def __init__(self, cursor: Any) -> None:
        self._cursor = cursor

    @property
    def _column_names(self) -> tuple[str, ...]:
        description = self._cursor.description or ()
        return tuple(col[0] for col in description)

    def fetchone(self) -> Row | None:
        row = self._cursor.fetchone()
        return Row(row, self._column_names) if row is not None else None

    def fetchall(self) -> list[Row]:
        cols = self._column_names
        return [Row(r, cols) for r in self._cursor.fetchall()]

    def __iter__(self) -> Iterator[Row]:
        return iter(self.fetchall())

    @property
    def rowcount(self) -> int:
        return self._cursor.rowcount

    @property
    def lastrowid(self) -> int | None:
        return self._cursor.lastrowid


class ConnectionAdapter:
    """Wraps a raw `libsql.Connection` to match the sqlite3.Connection surface `store.py` uses."""

    def __init__(self, connection: Any) -> None:
        self._connection = connection

    def execute(self, sql: str, params: Sequence[Any] = ()) -> CursorAdapter:
        return CursorAdapter(self._connection.execute(sql, params))

    def executescript(self, sql: str) -> None:
        self._connection.executescript(sql)

    def commit(self) -> None:
        self._connection.commit()

    def rollback(self) -> None:
        self._connection.rollback()

    def close(self) -> None:
        self._connection.close()


def connect(url: str, auth_token: str) -> ConnectionAdapter:
    """Open a remote Turso connection, wrapped to match sqlite3's Connection interface."""
    try:
        import libsql
    except ImportError as exc:  # pragma: no cover - exercised only when misconfigured
        raise ImportError(
            "TURSO_DATABASE_URL is set to a libsql:// URL but the 'libsql' package "
            "is not installed. Install it with: pip install 'horus-mcp[turso]'"
        ) from exc
    return ConnectionAdapter(libsql.connect(url, auth_token=auth_token))
