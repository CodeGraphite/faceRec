# Веб-сервер (FastAPI): отдаёт сайт и API для него.
# Запуск:  venv\Scripts\python.exe main.py   ->   http://localhost:8000

import asyncio
import os
import uuid

import cv2
import numpy as np
import uvicorn
from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.responses import StreamingResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

import camera
import config
import database
import face_engine

# Чтобы пути "data/..." и "static" работали, откуда бы ни запустили скрипт
os.chdir(os.path.dirname(os.path.abspath(__file__)))

app = FastAPI(title="Учёт студентов по лицу")


# ---------- Дашборд ----------

@app.get("/api/stats")
def get_stats():
    return database.get_stats()


@app.get("/api/events")
def get_events():
    return database.get_last_events()


@app.get("/api/camera")
def get_camera_state():
    return camera.camera_state


@app.get("/api/camera/stream")
async def camera_stream():
    """Live-видео для сайта. Формат MJPEG: сервер отправляет картинки JPEG одну за другой,
    а браузер показывает их в обычном теге <img> как видео."""
    async def send_frames():
        while True:
            if camera.annotated_frame is not None:
                success, jpeg = cv2.imencode(".jpg", camera.annotated_frame)
                yield b"--frame\r\nContent-Type: image/jpeg\r\n\r\n" + jpeg.tobytes() + b"\r\n"
            await asyncio.sleep(0.1)  # ~10 кадров в секунду

    return StreamingResponse(send_frames(), media_type="multipart/x-mixed-replace; boundary=frame")


@app.post("/api/reset")
def reset_statuses():
    database.reset_all_statuses()
    return {"ok": True}


# ---------- Студенты ----------

class StudentData(BaseModel):
    full_name: str
    group_name: str = ""


@app.get("/api/students")
def get_students():
    return database.get_students()


@app.post("/api/students")
def add_student(student: StudentData):
    if not student.full_name.strip():
        raise HTTPException(400, "Введите ФИО")
    student_id = database.add_student(student.full_name.strip(), student.group_name.strip())
    return {"id": student_id}


@app.get("/api/students/{student_id}")
def get_student_card(student_id: int):
    """Всё для карточки студента: данные, фото и история проходов."""
    student = database.get_student(student_id)
    if student is None:
        raise HTTPException(404, "Студент не найден")
    student["photos"] = database.get_photos(student_id)
    student["events"] = database.get_student_events(student_id)
    return student


@app.put("/api/students/{student_id}")
def update_student(student_id: int, student: StudentData):
    if not student.full_name.strip():
        raise HTTPException(400, "Введите ФИО")
    database.update_student(student_id, student.full_name.strip(), student.group_name.strip())
    return {"ok": True}


@app.delete("/api/students/{student_id}")
def delete_student(student_id: int):
    database.delete_student(student_id)
    return {"ok": True}


@app.post("/api/students/{student_id}/toggle")
def toggle_student(student_id: int):
    """Отметить вход/выход вручную (если камера не узнала)."""
    return {"status": database.toggle_status(student_id)}


@app.delete("/api/photos/{face_id}")
def delete_photo(face_id: int):
    database.delete_face(face_id)
    return {"ok": True}


def save_face_photo(student_id, image):
    """Найти лицо на фото, сохранить его отпечаток в базу и само фото в папку."""
    if face_engine.model is None:
        raise HTTPException(503, "Модель ещё загружается, подождите пару секунд")
    faces = face_engine.find_faces(image)
    if len(faces) == 0:
        raise HTTPException(400, "На фото не найдено лицо")
    if len(faces) > 1:
        raise HTTPException(400, "На фото несколько лиц — нужно фото одного человека")

    photo_file = f"{student_id}_{uuid.uuid4().hex[:8]}.jpg"
    cv2.imwrite(os.path.join(config.PHOTOS_FOLDER, photo_file), image)
    database.add_face(student_id, faces[0].normed_embedding, photo_file)
    return {"ok": True, "photo_file": photo_file}


@app.post("/api/students/{student_id}/photo")
async def upload_photo(student_id: int, file: UploadFile = File(...)):
    """Загрузить фото студента с компьютера."""
    data = await file.read()
    image = cv2.imdecode(np.frombuffer(data, np.uint8), cv2.IMREAD_COLOR)
    if image is None:
        raise HTTPException(400, "Это не изображение")
    return save_face_photo(student_id, image)


@app.post("/api/students/{student_id}/photo-from-camera")
def photo_from_camera(student_id: int):
    """Сделать фото студента с камеры (студент стоит перед камерой)."""
    if camera.last_frame is None:
        raise HTTPException(503, "Камера ещё не готова")
    return save_face_photo(student_id, camera.last_frame.copy())


# ---------- Файлы сайта ----------
# Порядок важен: сначала API, потом статические файлы.
os.makedirs(config.PHOTOS_FOLDER, exist_ok=True)
app.mount("/photos", StaticFiles(directory=config.PHOTOS_FOLDER), name="photos")
app.mount("/", StaticFiles(directory="static", html=True), name="site")


if __name__ == "__main__":
    database.create_tables()
    camera.start_camera()  # камера работает в фоне, параллельно с сайтом
    uvicorn.run(app, host="0.0.0.0", port=8000)
