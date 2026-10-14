# Работа камеры у входа. Запускается в отдельном потоке и работает всё время, пока работает сервер.
#
# Цикл:
#   1. Взять кадр с камеры.
#   2. Найти все лица и для каждого найти студента в базе.
#   3. Нарисовать рамки с именами (это видео показывается на сайте).
#   4. Узнанному студенту поменять статус (вошёл/вышел),
#      но не чаще, чем раз в COOLDOWN_SECONDS.

import threading
import time

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont

import config
import database
import face_engine

# Текущее состояние камеры — его показывает сайт
camera_state = {
    "running": False,
    "error": None,
    "last_recognized": None,  # имя последнего распознанного студента
}

last_frame = None        # последний кадр с камеры (из него делаем фото студента при регистрации)
annotated_frame = None   # тот же кадр, но с рамками и именами (для live-видео на сайте)
last_pass_time = {}      # student_id -> когда последний раз отметили

try:
    font = ImageFont.truetype("arial.ttf", 22)  # шрифт Windows с русскими буквами
except OSError:
    font = ImageFont.load_default()


def open_camera():
    source = config.CAMERA_SOURCE
    if isinstance(source, int):
        # CAP_DSHOW — быстрее открывает веб-камеру на Windows
        return cv2.VideoCapture(source, cv2.CAP_DSHOW)
    return cv2.VideoCapture(source)


def draw_faces(frame, results):
    """Нарисовать рамки: зелёная — студент узнан, красная — неизвестный.
    results — список (лицо, имя_или_None, похожесть)."""
    # OpenCV не умеет писать русские буквы, поэтому рисуем через библиотеку Pillow
    image = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    draw = ImageDraw.Draw(image)
    for face, name, similarity in results:
        x1, y1, x2, y2 = [int(v) for v in face.bbox]
        if name:
            color, text = (40, 200, 90), f"{name} ({similarity:.2f})"
        else:
            color, text = (230, 60, 60), "Неизвестный"
        draw.rectangle([x1, y1, x2, y2], outline=color, width=3)
        draw.text((x1, max(0, y1 - 28)), text, fill=color, font=font)
    return cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)


def camera_loop():
    global last_frame, annotated_frame
    face_engine.load_model()
    known_faces = database.get_all_faces()
    last_reload = time.time()

    camera = open_camera()
    camera_state["running"] = True

    while True:
        success, frame = camera.read()
        if not success:
            camera_state["error"] = "Нет изображения с камеры. Переподключение..."
            time.sleep(2)
            camera.release()
            camera = open_camera()
            continue
        camera_state["error"] = None
        last_frame = frame

        # Раз в 5 секунд перечитываем лица из базы (вдруг добавили нового студента)
        if time.time() - last_reload > 5:
            known_faces = database.get_all_faces()
            last_reload = time.time()

        results = []
        for face in face_engine.find_faces(frame):
            student_id, full_name, similarity = face_engine.find_best_match(face.normed_embedding, known_faces)
            results.append((face, full_name, similarity))
            if student_id is not None:
                mark_pass(student_id, full_name, similarity)

        annotated_frame = draw_faces(frame, results)


def mark_pass(student_id, full_name, similarity):
    # Студент только что проходил — пропускаем
    if time.time() - last_pass_time.get(student_id, 0) < config.COOLDOWN_SECONDS:
        return
    last_pass_time[student_id] = time.time()
    new_status = database.toggle_status(student_id, similarity)
    camera_state["last_recognized"] = full_name
    print(f"{full_name}: {'вошёл' if new_status == 'inside' else 'вышел'} (похожесть {similarity:.2f})")


def start_camera():
    thread = threading.Thread(target=camera_loop, daemon=True)  # daemon — поток завершится вместе с сервером
    thread.start()
