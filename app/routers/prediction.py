from fastapi import APIRouter

router = APIRouter()

@router.get("/predict")
def predict(...):
    # call model prediction service
    pass
