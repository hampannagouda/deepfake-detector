from fastapi import FastAPI, File, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
import shutil
import os
import cv2
from utils.face_detection import FaceDetector
from utils.video_processor import extract_frames
from inference import DeepfakeDetector
import tempfile

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

detector = FaceDetector()
model_path = "app/models/xception_deepfake.pth"
classifier = DeepfakeDetector(model_path)

@app.post("/analyze")
async def analyze(file: UploadFile = File(...)):
    with tempfile.NamedTemporaryFile(delete=False, suffix=os.path.splitext(file.filename)[1]) as tmp:
        shutil.copyfileobj(file.file, tmp)
        tmp_path = tmp.name

    is_video = file.filename.lower().endswith(('.mp4', '.avi', '.mov'))
    all_scores = []

    try:
        if is_video:
            frames = extract_frames(tmp_path, max_frames=16)
            for frame in frames:
                faces = detector.detect(frame)
                for face in faces:
                    pred = classifier.predict(face)
                    all_scores.append(pred["fake"])
        else:
            img = cv2.imdecode(np.frombuffer(await file.read(), np.uint8), cv2.IMREAD_COLOR)
            faces = detector.detect(img)
            for face in faces:
                pred = classifier.predict(face)
                all_scores.append(pred["fake"])

        if not all_scores:
            return JSONResponse({"result": "Unknown", "confidence": 0})

        avg_score = sum(all_scores) / len(all_scores)
        result = "Fake" if avg_score > 0.5 else "Real"
        confidence = round(avg_score * 100, 2) if result == "Fake" else round((1 - avg_score) * 100, 2)

        return {"result": result, "confidence": confidence}

    finally:
        os.unlink(tmp_path)