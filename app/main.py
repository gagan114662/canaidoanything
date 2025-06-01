from fastapi import FastAPI, File, UploadFile, HTTPException, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
import uvicorn
import os
from pathlib import Path

from app.api.endpoints import image_processing, health
from app.core.config import settings

app = FastAPI(
    title="Garment Creative AI",
    description="Transform ugly garment photos into professional creative imagery",
    version="1.0.0"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.ALLOWED_HOSTS,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Mount static files
app.mount("/static", StaticFiles(directory="app/static"), name="static")

# Include routers
app.include_router(health.router, prefix="/api/v1", tags=["health"])
app.include_router(image_processing.router, prefix="/api/v1", tags=["image-processing"])

# Import and include model transformation router
from app.api.endpoints import model_transformation
app.include_router(model_transformation.router, prefix="/api/v1", tags=["model-transformation"])

@app.get("/")
async def root():
    return {"message": "Garment Creative AI API", "version": "1.0.0"}

if __name__ == "__main__":
    uvicorn.run(
        "app.main:app",
        host=settings.HOST,
        port=settings.PORT,
        reload=settings.DEBUG,
        workers=1 if settings.DEBUG else 4
    )