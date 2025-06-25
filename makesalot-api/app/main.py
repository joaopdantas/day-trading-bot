from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from app.routers import analysis, ml  # se usares routers

app = FastAPI()

# Permitir pedidos da extensão Chrome
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # ou ["chrome-extension://<id>"] se quiseres restringir
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(analysis.router, prefix="/api/analysis")
app.include_router(ml.router, prefix="/api/predict")

@app.get("/")
def root():
    return {"message": "API FastAPI MakesALot online!"}
