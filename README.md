# Face Detection API

A production-ready Face Detection application built with **OpenCV**, **FastAPI**, and **Haar Cascade**. The project supports face detection from images, videos, and live webcam streams through a REST API.

---

## Features

- Face Detection using Haar Cascade
- Image Face Detection
- Video Face Detection
- Webcam Face Detection
- FastAPI REST API
- OpenCV Image Processing
- Automatic Input & Output Directory Management
- Modular Project Structure
- Production-ready Code Organization

---

## Tech Stack

### Backend

- FastAPI
- Uvicorn

### Computer Vision

- OpenCV
- Haar Cascade

### Language

- Python 3.13+

---

## Project Structure

```text
Face-Detection/
│
├── app/
│   ├── main.py
│   └── schemas.py
│
├── checkpoints/
│   └── haarcascade_frontalface_default.xml
│
├── inputs/
│   ├── image/
│   └── video/
│
├── outputs/
│   ├── image/
│   ├── video/
│   └── webcam/
│
├── src/
│   ├── config.py
│   ├── detector.py
│   ├── draw.py
│   ├── inference.py
│   ├── preprocess.py
│   └── utils.py
│
├── tests/
│
├── requirements.txt
└── README.md
```

---

## Installation

Clone the repository

```bash
git clone https://github.com/maroofiums/Face-Detection.git

cd Face-Detection
```

Create a virtual environment

```bash
python -m venv .venv
```

Activate the environment

Windows

```bash
.venv\Scripts\activate
```

Linux / macOS

```bash
source .venv/bin/activate
```

Install dependencies

```bash
pip install -r requirements.txt
```

---

## Run the API

```bash
uvicorn app.main:app --reload
```

Server

```
http://127.0.0.1:8000
```

Swagger UI

```
http://127.0.0.1:8000/docs
```

ReDoc

```
http://127.0.0.1:8000/redoc
```

---

## API Endpoints

### Health Check

```
GET /
```

Response

```json
{
  "status": "success",
  "message": "Face Detection API is running."
}
```

---

### Detect Faces in Image

```
POST /detect/image
```

Request

- Multipart Form Data
- Upload an image

Response

```json
{
  "filename": "person.jpg",
  "faces_detected": 2,
  "output_path": "outputs/image/person.jpg"
}
```

---

### Detect Faces in Video

```
POST /detect/video
```

Request

- Multipart Form Data
- Upload a video

Response

```json
{
  "filename": "video.mp4",
  "faces_detected": 0,
  "output_path": "outputs/video/video.mp4"
}
```

---

### Webcam Detection

```
POST /detect/webcam
```

Response

```json
{
  "status": "success",
  "message": "Webcam detection completed.",
  "output_directory": "outputs/webcam"
}
```

---

## Input Directories

Images

```
inputs/image/
```

Videos

```
inputs/video/
```

---

## Output Directories

Processed Images

```
outputs/image/
```

Processed Videos

```
outputs/video/
```

Webcam Results

```
outputs/webcam/
```

---

## Detection Pipeline

```text
          Image / Video / Webcam
                    │
                    ▼
              Preprocessing
                    │
                    ▼
           Grayscale Conversion
                    │
                    ▼
              Gaussian Blur
                    │
                    ▼
           Histogram Equalization
                    │
                    ▼
          Haar Cascade Face Detection
                    │
                    ▼
            Bounding Box Drawing
                    │
                    ▼
                Save Output
```

---

## Future Improvements

- YOLOv8 Face Detection
- Face Recognition
- Face Tracking
- Face Mask Detection
- Age & Gender Prediction
- Docker Support
- CI/CD Pipeline
- Unit Tests
- Logging Middleware
- Model Selection via API
- Confidence Threshold Parameter

---

## Learning Objectives

This project demonstrates:

- OpenCV Fundamentals
- Image Processing
- Face Detection
- Object Localization
- FastAPI Development
- REST API Design
- Modular Python Architecture
- Clean Code Principles

---

## License

This project is licensed under the MIT License.
