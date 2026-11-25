# Radiology AI Assistant — Web Application

A standalone React + FastAPI website built around the supplied `radiology_ai_assistant.py`.

## What is included
- React/Vite frontend with login, registration, dashboard, X-ray upload and results
- FastAPI backend with JWT authentication
- MongoDB-backed user accounts and scan history
- Integration with the supplied CNN/SVC inference pipeline
- CNN Grad-CAM result display
- Research-use disclaimer

## 1. Requirements
- Python 3.10 or 3.11 recommended
- Node.js 20+
- MongoDB running locally or a MongoDB Atlas connection
- The datasets listed in the original Python script
- Trained model weights before predictions can work

## 2. Put your AI engine in place
Copy your original `radiology_ai_assistant.py` into:

`backend/engine/radiology_ai_assistant.py`

Do not rename the file.

## 3. Backend setup (Windows PowerShell)
```powershell
cd backend
python -m venv venv
.\venv\Scripts\Activate.ps1
pip install -r requirements.txt
Copy-Item .env.example .env
```
Edit `.env` and set a strong `JWT_SECRET` and your `MONGO_URI`.

Train the model using the original script from the `backend` folder, after placing datasets in the expected structure:
```powershell
python engine/radiology_ai_assistant.py prepare --data-root .\data --workdir .\workdir
python engine/radiology_ai_assistant.py train-cnn --preset clahe_aug --workdir .\workdir
```
Optional SVC branch:
```powershell
python engine/radiology_ai_assistant.py train-svc --preset clahe_aug --workdir .\workdir
```

The app expects CNN weights at:
`backend/workdir/runs/clahe_aug/cnn/model.pt`

For SVC, it additionally expects:
`backend/workdir/runs/clahe_aug/svc/svc.joblib`
and `backbone.pt`.

Start the API:
```powershell
uvicorn main:app --reload --host 127.0.0.1 --port 8000
```
API docs: http://127.0.0.1:8000/docs

## 4. Frontend setup
Open a second terminal:
```powershell
cd frontend
npm install
npm run dev
```
Open the URL printed by Vite (normally http://localhost:5173).

## 5. MongoDB
For local MongoDB, use:
`MONGO_URI=mongodb://127.0.0.1:27017`

Or paste your Atlas connection string into `.env`.

## Notes
- The backend can start without trained weights; `/api/health` will report that models are unavailable. Prediction requests return HTTP 503 until weights exist.
- The supplied engine uses its own preprocessing configuration stored in the checkpoint. Do not independently preprocess images in the browser.
- CNN scores are uncalibrated softmax outputs, not clinical probabilities. SVC margins are not probabilities unless the SVC was trained with probability calibration.
- Grad-CAM is available only for the CNN branch.
- This is a research/teaching prototype, not a medical device. Never use it for diagnosis, triage or treatment.
