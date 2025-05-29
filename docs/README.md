# AlphaPulse Backend (FastAPI)

Modular, containerized backend for AI-based financial predictions.

---

## 🚀 Quickstart

### 1. Clone & Install Dependencies
```bash
git clone https://github.com/<your-org>/alphapulse-backend.git
cd alphapulse-backend
```

### 2. Docker Build & Run
```bash
docker-compose up --build
```

App should be available at: http://localhost:8000

---

## ✅ Health Check

Check if the backend is up and responsive:

```bash
curl http://localhost:8000/health
```

Expected:
```json
{ "status": "ok" }
```

---

## 🗂 Folder Structure

```bash
app/
├── main.py
├── core/
├── routers/
├── services/
├── models/
└── utils/
```

---

## 🧪 Test Scenarios

| Scenario | Command | Expected |
|----------|---------|----------|
| Server boot | `docker-compose up` | App on :8000 |
| Health check | `curl localhost:8000/health` | `{"status":"ok"}` |

---

## ⚙️ Tech Stack

- FastAPI
- Python 3.11
- Docker
- Uvicorn
- Pydantic v2 (via `pydantic-settings`)
