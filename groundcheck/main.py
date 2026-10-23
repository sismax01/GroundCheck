"""
ASGI entrypoint for running GroundCheck via `uvicorn groundcheck.main:app`.
"""

from groundcheck.api import create_app

app = create_app()

if __name__ == "__main__":
    import uvicorn
    from groundcheck.config import Settings

    cfg = Settings.from_env()
    uvicorn.run("groundcheck.main:app", host=cfg.host, port=cfg.port, reload=False)
