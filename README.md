# Deepfake Detector

A full-stack web application designed to detect deepfakes using advanced computer vision techniques. 

## 🚀 Tech Stack

### Frontend
- **React 18**
- **Vite**
- **Axios**

### Backend
- **FastAPI**
- **OpenCV**
- **MediaPipe**
- **NumPy**
- **Uvicorn**

## 📦 Prerequisites
- **Python 3.8+**
- **Node.js 18+**
- **npm** or **yarn**

## 🛠️ Installation & Setup

### 1. Clone the repository
```bash
git clone <your-repository-url>
cd deepfake-detector
```

### 2. Backend Setup (FastAPI)
Navigate to the backend directory and set up a virtual environment:

```bash
cd backend
python -m venv venv

# Activate the virtual environment
# On Windows:
venv\Scripts\activate
# On macOS/Linux:
source venv/bin/activate

# Install the dependencies
pip install -r requirements.txt
```

### 3. Frontend Setup (React/Vite)
Open a new terminal, navigate to the frontend directory, and install the dependencies:

```bash
cd frontend
npm install
```

## 🚦 Running the Application

### Start the Backend Server
Make sure your Python virtual environment is activated, then run:

```bash
cd backend
# Replace app.main:app with the actual entry point of your FastAPI app if it's different
uvicorn app.main:app --reload
```
The FastAPI backend will typically be accessible at `http://localhost:8000`.

### Start the Frontend Server
In a separate terminal:

```bash
cd frontend
npm run dev
```
Your frontend should now be running locally at `http://localhost:5173`.

## 📁 Project Structure

```text
deepfake-detector/
├── backend/          # Python backend (FastAPI, OpenCV, MediaPipe)
│   ├── app/          # Source code for the backend API
│   ├── venv/         # Python virtual environment (ignored in git)
│   └── requirements.txt
├── frontend/         # React + Vite frontend
│   ├── src/          # Source code for the UI
│   ├── public/       # Static assets
│   └── package.json
├── .gitignore        # Ignored files definition
└── README.md         # Project documentation
```
