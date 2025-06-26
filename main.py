# detector_api/main.py
from fastapi import FastAPI, UploadFile, Form, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from PIL import Image
import io
import os
import logging
from transformers import pipeline

# Настройка логирования
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Параметры удобно отдавать через переменные окружения
MODEL_NAME = os.getenv("MODEL_NAME", "google/owlv2-base-patch16-ensemble")
THRESHOLD = float(os.getenv("THRESHOLD", "0.30"))
DEVICE_ID = int(os.getenv("CUDA_DEVICE", "0"))

# Инициализация модели
try:
    detector = pipeline(
        task="zero-shot-object-detection",
        model=MODEL_NAME,
        device=DEVICE_ID,
        threshold=THRESHOLD,
        box_format="xyxy"
    )
    logger.info(f"Model {MODEL_NAME} loaded successfully on device {DEVICE_ID}")
except Exception as e:
    logger.error(f"Failed to load model: {e}")
    raise

app = FastAPI(title="Open-Set Detector API", version="1.0")

# CORS middleware - в продакшене лучше ограничить origins
app.add_middleware(
    CORSMiddleware, 
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.get("/health")
async def health_check():
    """Проверка работоспособности сервиса"""
    return {"status": "ok", "model": MODEL_NAME}

@app.post("/detect/")
async def detect_object(image: UploadFile, label: str = Form(...)):
    """
    Детекция объектов на изображении
    - image: JPEG/PNG изображение
    - label: текстовое описание искомого объекта
    Возвращает: {bbox, score, label} или {error}
    """
    try:
        # Проверка типа файла
        if not image.content_type or not image.content_type.startswith('image/'):
            raise HTTPException(status_code=400, detail="File must be an image")
        
        # Чтение и обработка изображения
        data = await image.read()
        if len(data) == 0:
            raise HTTPException(status_code=400, detail="Empty file uploaded")
            
        pil_img = Image.open(io.BytesIO(data)).convert("RGB")
        
        # Детекция
        results = detector(pil_img, candidate_labels=[label])
        
        if not results:
            return {
                "error": "nothing found", 
                "label": label,
                "threshold": THRESHOLD
            }
        
        # Возвращаем самый уверенный результат
        best = results[0]
        return {
            "bbox": best["box"],
            "score": float(best["score"]),
            "label": best["label"],
            "total_detections": len(results)
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error during detection: {e}")
        raise HTTPException(status_code=500, detail=f"Internal server error: {str(e)}")

@app.post("/detect_all/")
async def detect_all_objects(image: UploadFile, label: str = Form(...)):
    """
    Детекция всех найденных объектов (не только лучшего)
    """
    try:
        if not image.content_type or not image.content_type.startswith('image/'):
            raise HTTPException(status_code=400, detail="File must be an image")
        
        data = await image.read()
        if len(data) == 0:
            raise HTTPException(status_code=400, detail="Empty file uploaded")
            
        pil_img = Image.open(io.BytesIO(data)).convert("RGB")
        results = detector(pil_img, candidate_labels=[label])
        
        if not results:
            return {
                "error": "nothing found", 
                "label": label,
                "threshold": THRESHOLD
            }
        
        # Возвращаем все результаты
        detections = []
        for result in results:
            detections.append({
                "bbox": result["box"],
                "score": float(result["score"]),
                "label": result["label"]
            })
        
        return {
            "detections": detections,
            "total_count": len(detections),
            "label": label
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error during detection: {e}")
        raise HTTPException(status_code=500, detail=f"Internal server error: {str(e)}")
