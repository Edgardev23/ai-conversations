"""SQLite: esquema y queries. Modo WAL, user_id en todas las tablas
relevantes desde el día uno aunque hoy solo exista un usuario.
"""

import json
import sqlite3

import config

SCHEMA = """
CREATE TABLE IF NOT EXISTS users (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    name TEXT NOT NULL UNIQUE,
    created_at TEXT NOT NULL DEFAULT (datetime('now'))
);

CREATE TABLE IF NOT EXISTS sessions (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    user_id INTEGER NOT NULL REFERENCES users(id),
    started_at TEXT NOT NULL DEFAULT (datetime('now')),
    ended_at TEXT,
    scenario TEXT,
    persona TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS session_reports (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    session_id INTEGER NOT NULL REFERENCES sessions(id),
    user_id INTEGER NOT NULL REFERENCES users(id),
    transcript_summary TEXT,
    grammar_errors_json TEXT,
    vocab_gaps_json TEXT,
    pronunciation_scores_json TEXT,
    fluency_metrics_json TEXT,
    recommendations TEXT,
    created_at TEXT NOT NULL DEFAULT (datetime('now'))
);
"""


def get_connection() -> sqlite3.Connection:
    conn = sqlite3.connect(config.DB_PATH)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA journal_mode=WAL")
    conn.execute("PRAGMA foreign_keys=ON")
    return conn


def init_db() -> None:
    conn = get_connection()
    try:
        conn.executescript(SCHEMA)
        conn.commit()
    finally:
        conn.close()


def get_or_create_default_user(name: str = "default") -> int:
    conn = get_connection()
    try:
        row = conn.execute("SELECT id FROM users WHERE name = ?", (name,)).fetchone()
        if row:
            return row["id"]
        cursor = conn.execute("INSERT INTO users (name) VALUES (?)", (name,))
        conn.commit()
        return cursor.lastrowid
    finally:
        conn.close()


def create_session(user_id: int, persona: str, scenario: str | None = None) -> int:
    conn = get_connection()
    try:
        cursor = conn.execute(
            "INSERT INTO sessions (user_id, persona, scenario) VALUES (?, ?, ?)",
            (user_id, persona, scenario),
        )
        conn.commit()
        return cursor.lastrowid
    finally:
        conn.close()


def end_session(session_id: int) -> None:
    conn = get_connection()
    try:
        conn.execute("UPDATE sessions SET ended_at = datetime('now') WHERE id = ?", (session_id,))
        conn.commit()
    finally:
        conn.close()


def save_session_report(session_id: int, user_id: int, report: dict) -> int:
    conn = get_connection()
    try:
        cursor = conn.execute(
            """
            INSERT INTO session_reports (
                session_id, user_id, transcript_summary, grammar_errors_json,
                vocab_gaps_json, pronunciation_scores_json, fluency_metrics_json,
                recommendations
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                session_id,
                user_id,
                report.get("transcript_summary"),
                json.dumps(report.get("grammar_errors", [])),
                json.dumps(report.get("vocab_gaps", [])),
                json.dumps(report.get("pronunciation_scores", {})),
                json.dumps(report.get("fluency_metrics", {})),
                report.get("recommendations"),
            ),
        )
        conn.commit()
        return cursor.lastrowid
    finally:
        conn.close()


def get_recent_session_reports(user_id: int, limit: int = 3) -> list[dict]:
    conn = get_connection()
    try:
        rows = conn.execute(
            """
            SELECT * FROM session_reports
            WHERE user_id = ?
            ORDER BY created_at DESC
            LIMIT ?
            """,
            (user_id, limit),
        ).fetchall()
        return [dict(row) for row in rows]
    finally:
        conn.close()


def get_memory_summary(user_id: int, limit: int = 5) -> str:
    """Resumen en texto de errores de gramática/fraseo y vocabulario pendiente
    de las últimas `limit` sesiones, para inyectar en el system prompt del
    profesor (ver teacher_llm.build_system_prompt). "" si no hay historial.
    """
    reports = get_recent_session_reports(user_id, limit=limit)
    if not reports:
        return ""

    grammar_errors: list[str] = []
    vocab_gaps: list[str] = []
    for report in reports:
        grammar_errors += json.loads(report["grammar_errors_json"] or "[]")
        vocab_gaps += json.loads(report["vocab_gaps_json"] or "[]")

    # de-duplicar preservando orden (más reciente primero, get_recent_session_reports ya ordena DESC)
    grammar_errors = list(dict.fromkeys(grammar_errors))
    vocab_gaps = list(dict.fromkeys(vocab_gaps))

    lines = []
    if grammar_errors:
        lines.append("Grammar/phrasing issues from recent sessions:")
        lines += [f"- {item}" for item in grammar_errors]
    if vocab_gaps:
        lines.append("Vocabulary the student has struggled with recently:")
        lines += [f"- {item}" for item in vocab_gaps]

    return "\n".join(lines)
