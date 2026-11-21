"""Application ASGI minimale pour valider le conteneur applicatif."""

from fastapi import FastAPI

app = FastAPI(title="SLR-Nexus", version="0.1.0")


@app.get("/health", tags=["health"])
def health() -> dict[str, str]:
    """Expose l'état du processus sans dépendre des services externes."""
    return {"status": "ok"}
