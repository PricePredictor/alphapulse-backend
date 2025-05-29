# Project Folder Structure (AlphaPulse Backend)

This document outlines the modular folder structure used in the AlphaPulse backend to enable scalability, clarity, and clean separation of responsibilities.

## 📦 Base Layout

```
alphapulse-backend/
│
├── app/
│   ├── main.py             # Entry point for FastAPI
│   ├── core/               # Settings, config, logging
│   │   ├── config.py
│   │   ├── logger.py
│   ├── routers/            # All FastAPI route modules
│   │   └── health.py
│   ├── services/           # Business logic and processing
│   ├── models/             # ML models, schemas, or Pydantic models
│   ├── utils/              # Helper utilities
│   └── __init__.py
│
├── Dockerfile              # Container definition
├── docker-compose.yml      # Dev container orchestration
├── requirements.txt        # Python dependencies
├── .env.example            # Sample environment file
├── README.md               # Project overview
├── docs/                   # Internal developer docs
└── logs/                   # Runtime logs (volume-mapped)
```

## 🔍 Notes
- Each folder has an `__init__.py` to ensure importability.
- The `main.py` initializes the app and connects all routers.
- The `.env` file manages config outside of code for security and environment separation.
