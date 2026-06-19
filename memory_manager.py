import sqlite3
import os
import time

BASE_DIR = os.path.dirname(os.path.abspath(__file__)) if '__file__' in locals() else os.getcwd()
DB_PATH = os.path.join(BASE_DIR, "session_memory.db")

class SessionMemoryManager:
    def __init__(self):
        self.conn = sqlite3.connect(DB_PATH, check_same_thread=False)
        self.create_schema()

    def create_schema(self):
        with self.conn:
            self.conn.execute("""
                CREATE TABLE IF NOT EXISTS chat_history (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    session_id TEXT,
                    timestamp REAL,
                    role TEXT,
                    content TEXT
                )
            """)

    def save_message(self, session_id: str, role: str, content: str):
        with self.conn:
            self.conn.execute(
                "INSERT INTO chat_history (session_id, timestamp, role, content) VALUES (?, ?, ?, ?)",
                (session_id, time.time(), role, content)
            )

    def fetch_history(self, session_id: str, limit: int = 6):
        cursor = self.conn.cursor()
        cursor.execute(
            "SELECT role, content FROM chat_history WHERE session_id = ? ORDER BY timestamp DESC LIMIT ?",
            (session_id, limit)
        )
        rows = cursor.fetchall()
        return [{"role": r, "content": c} for r, c in reversed(rows)]
