# Migration to Pydantic v2.x – Import Changes (May 2025)

## 🔄 Context

FastAPI 0.110.0 requires `pydantic>=2.x`, which introduces breaking changes in how configuration classes (`BaseSettings`) are handled.

## 🔥 Issue Encountered

```
ModuleNotFoundError: No module named 'pydantic_settings'
```

### ⚠️ Why:
In Pydantic 2.x, `BaseSettings` is now located in a **new package**: `pydantic-settings`.

## ✅ Fix Applied

### 1. Update requirements.txt
```txt
pydantic-settings==2.2.1
```

### 2. Modify `config.py`
**Old:**
```python
from pydantic import BaseSettings
```

**New:**
```python
from pydantic_settings import BaseSettings
```

## 🔗 Resources
- [Pydantic Migration Docs](https://docs.pydantic.dev/latest/migration/)
