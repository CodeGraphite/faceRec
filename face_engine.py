# Распознавание лиц с помощью библиотеки InsightFace.
#
# Как это работает:
#   1. Нейросеть-детектор находит лица на кадре (прямоугольники).
#   2. Вторая нейросеть превращает каждое лицо в "отпечаток" — вектор из 512 чисел (embedding).
#      У одного и того же человека отпечатки похожи, у разных людей — нет.
#   3. Сравниваем отпечаток с камеры с отпечатками из базы. Похожесть считаем как
#      скалярное произведение векторов (cosine similarity): 1.0 — одно и то же лицо, ~0 — разные люди.

import warnings

import numpy as np
from insightface.app import FaceAnalysis

import config

warnings.filterwarnings("ignore", category=FutureWarning)

model = None  # нейросети загружаются один раз при старте


def load_model():
    global model
    if config.USE_GPU:
        make_gpu_wait_without_cpu()
        providers = ["CUDAExecutionProvider", "CPUExecutionProvider"]
    else:
        providers = ["CPUExecutionProvider"]

    # buffalo_l — готовый набор моделей InsightFace. Берём из него только две:
    # detection (поиск лиц) и recognition (отпечаток лица).
    model = FaceAnalysis(name="buffalo_l", allowed_modules=["detection", "recognition"], providers=providers)
    model.prepare(ctx_id=0 if config.USE_GPU else -1, det_size=(640, 640))

    used = model.det_model.session.get_providers()
    print("Модель загружена. Работает на:", "видеокарте (GPU)" if "CUDAExecutionProvider" in used else "процессоре (CPU)")


def find_faces(image):
    """Найти все лица на изображении. У каждого лица есть .bbox (рамка) и .normed_embedding (отпечаток)."""
    return model.get(image)


def find_best_match(embedding, known_faces):
    """Сравнить отпечаток со всеми отпечатками из базы.
    known_faces — список (student_id, full_name, embedding).
    Возвращает (student_id, full_name, similarity) самого похожего студента
    или (None, None, лучшее_значение), если похожих нет."""
    best_id, best_name, best_similarity = None, None, 0.0
    for student_id, full_name, known_embedding in known_faces:
        similarity = float(np.dot(embedding, known_embedding))
        if similarity > best_similarity:
            best_id, best_name, best_similarity = student_id, full_name, similarity

    if best_similarity < config.SIMILARITY_THRESHOLD:
        return None, None, best_similarity
    return best_id, best_name, best_similarity


def make_gpu_wait_without_cpu():
    """Без этой настройки, пока видеокарта считает, процессор "крутится" в цикле ожидания
    и одно ядро загружено на 100%. Здесь мы говорим CUDA: "пока ждёшь видеокарту — спи".
    Скорость та же, а нагрузка на процессор падает почти до нуля."""
    import ctypes
    import glob
    import os

    import nvidia
    import onnxruntime

    onnxruntime.preload_dlls()  # загрузить библиотеки CUDA из pip-пакетов nvidia-*
    for folder in nvidia.__path__:
        for dll in glob.glob(os.path.join(folder, "**", "cudart64_*.dll"), recursive=True):
            ctypes.CDLL(dll).cudaSetDeviceFlags(4)  # 4 = cudaDeviceScheduleBlockingSync
            return
