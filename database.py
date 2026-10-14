# Работа с базой данных SQLite через стандартный модуль sqlite3.
#
# Таблицы:
#   students — студенты и их текущий статус ("inside" — в Универе, "outside" — вне)
#   faces    — "отпечатки" лиц студентов (вектор из 512 чисел), у студента может быть несколько фото
#   events   — история проходов через камеру (кто, когда, вошёл или вышел)

import os
import sqlite3
from datetime import datetime

import numpy as np

import config


def get_connection():
    os.makedirs(os.path.dirname(config.DATABASE_FILE), exist_ok=True)
    connection = sqlite3.connect(config.DATABASE_FILE)
    connection.row_factory = sqlite3.Row  # строки можно читать как словарь: row["full_name"]
    return connection


def query(sql, params=()):
    """Выполнить SELECT и вернуть список словарей."""
    connection = get_connection()
    rows = connection.execute(sql, params).fetchall()
    connection.close()
    return [dict(row) for row in rows]


def execute(sql, params=()):
    """Выполнить INSERT / UPDATE / DELETE. Возвращает id новой строки (для INSERT)."""
    connection = get_connection()
    cursor = connection.execute(sql, params)
    connection.commit()
    connection.close()
    return cursor.lastrowid


def create_tables():
    connection = get_connection()
    connection.executescript("""
        CREATE TABLE IF NOT EXISTS students (
            id          INTEGER PRIMARY KEY AUTOINCREMENT,
            full_name   TEXT NOT NULL,
            group_name  TEXT,
            status      TEXT NOT NULL DEFAULT 'outside',
            last_seen   TEXT
        );

        CREATE TABLE IF NOT EXISTS faces (
            id          INTEGER PRIMARY KEY AUTOINCREMENT,
            student_id  INTEGER NOT NULL,
            embedding   BLOB NOT NULL,
            photo_file  TEXT
        );

        CREATE TABLE IF NOT EXISTS events (
            id          INTEGER PRIMARY KEY AUTOINCREMENT,
            student_id  INTEGER NOT NULL,
            action      TEXT NOT NULL,      -- 'in' или 'out'
            similarity  REAL,
            time        TEXT NOT NULL
        );
    """)
    connection.close()


def now():
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S")


# ---------- Студенты ----------

def get_students():
    return query("""
        SELECT students.*, COUNT(faces.id) AS photos_count
        FROM students
        LEFT JOIN faces ON faces.student_id = students.id
        GROUP BY students.id
        ORDER BY students.full_name
    """)


def get_student(student_id):
    """Один студент или None, если такого нет."""
    rows = query("SELECT * FROM students WHERE id = ?", (student_id,))
    return rows[0] if rows else None


def add_student(full_name, group_name):
    return execute("INSERT INTO students (full_name, group_name) VALUES (?, ?)", (full_name, group_name))


def update_student(student_id, full_name, group_name):
    execute("UPDATE students SET full_name = ?, group_name = ? WHERE id = ?", (full_name, group_name, student_id))


def delete_student(student_id):
    for face in query("SELECT photo_file FROM faces WHERE student_id = ?", (student_id,)):
        path = os.path.join(config.PHOTOS_FOLDER, face["photo_file"])
        if os.path.exists(path):
            os.remove(path)
    execute("DELETE FROM faces WHERE student_id = ?", (student_id,))
    execute("DELETE FROM events WHERE student_id = ?", (student_id,))
    execute("DELETE FROM students WHERE id = ?", (student_id,))


def toggle_status(student_id, similarity=None):
    """Студент прошёл мимо камеры: если был вне Универа — вошёл, если внутри — вышел."""
    student = query("SELECT status FROM students WHERE id = ?", (student_id,))[0]
    if student["status"] == "outside":
        new_status, action = "inside", "in"
    else:
        new_status, action = "outside", "out"
    execute("UPDATE students SET status = ?, last_seen = ? WHERE id = ?", (new_status, now(), student_id))
    execute("INSERT INTO events (student_id, action, similarity, time) VALUES (?, ?, ?, ?)",
            (student_id, action, similarity, now()))
    return new_status


def reset_all_statuses():
    """Отметить всех как "вне Универа" (например, в конце дня)."""
    execute("UPDATE students SET status = 'outside'")


# ---------- Лица ----------

def add_face(student_id, embedding, photo_file):
    # numpy-массив сохраняем в базу как байты
    execute("INSERT INTO faces (student_id, embedding, photo_file) VALUES (?, ?, ?)",
            (student_id, embedding.astype(np.float32).tobytes(), photo_file))


def get_all_faces():
    """Все сохранённые лица: список (student_id, full_name, embedding)."""
    rows = query("""
        SELECT faces.student_id, faces.embedding, students.full_name
        FROM faces JOIN students ON students.id = faces.student_id
    """)
    return [(row["student_id"], row["full_name"], np.frombuffer(row["embedding"], dtype=np.float32))
            for row in rows]


def get_photos(student_id):
    return query("SELECT id, photo_file FROM faces WHERE student_id = ?", (student_id,))


def delete_face(face_id):
    for face in query("SELECT photo_file FROM faces WHERE id = ?", (face_id,)):
        path = os.path.join(config.PHOTOS_FOLDER, face["photo_file"])
        if os.path.exists(path):
            os.remove(path)
    execute("DELETE FROM faces WHERE id = ?", (face_id,))


# ---------- Статистика для дашборда ----------

def get_stats():
    today = datetime.now().strftime("%Y-%m-%d")
    total = query("SELECT COUNT(*) AS n FROM students")[0]["n"]
    inside = query("SELECT COUNT(*) AS n FROM students WHERE status = 'inside'")[0]["n"]
    came_today = query("SELECT COUNT(DISTINCT student_id) AS n FROM events WHERE action = 'in' AND time LIKE ?",
                       (today + "%",))[0]["n"]
    return {"total": total, "inside": inside, "outside": total - inside, "came_today": came_today}


def get_student_events(student_id, limit=50):
    return query("SELECT * FROM events WHERE student_id = ? ORDER BY id DESC LIMIT ?", (student_id, limit))


def get_last_events(limit=15):
    return query("""
        SELECT events.*, students.full_name, students.group_name
        FROM events JOIN students ON students.id = events.student_id
        ORDER BY events.id DESC
        LIMIT ?
    """, (limit,))
