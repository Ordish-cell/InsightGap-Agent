from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.orm import Session

from src.web_app.agent.llm.schemas import ConnectionCreate, ConnectionTest, ConnectionUpdate, ModelCreate, ModelUpdate, PreferenceUpdate
from src.web_app.db.session import get_db
from src.web_app.schemas.common import ok
from src.web_app.services.auth_service import get_current_user_id
from src.web_app.services.llm_registry_service import (
    add_model, create_connection, delete_connection, delete_model,
    discover_models, get_catalog, get_preferences, list_connections, test_connection,
    update_connection, update_model, update_preferences,
)

router = APIRouter()


def _call(action):
    try:
        return ok(action())
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc


@router.get("/catalog")
def catalog():
    return ok(get_catalog())


@router.get("/connections")
def connections(user_id: int = Depends(get_current_user_id), db: Session = Depends(get_db)):
    return ok(list_connections(db, user_id))


@router.post("/connections")
def create(payload: ConnectionCreate, user_id: int = Depends(get_current_user_id), db: Session = Depends(get_db)):
    return _call(lambda: create_connection(db, user_id, payload.model_dump()))


@router.patch("/connections/{connection_id}")
def update(connection_id: int, payload: ConnectionUpdate, user_id: int = Depends(get_current_user_id), db: Session = Depends(get_db)):
    return _call(lambda: update_connection(db, user_id, connection_id, payload.model_dump(exclude_unset=True)))


@router.delete("/connections/{connection_id}")
def remove(connection_id: int, user_id: int = Depends(get_current_user_id), db: Session = Depends(get_db)):
    return _call(lambda: delete_connection(db, user_id, connection_id))


@router.post("/connections/test")
def test(payload: ConnectionTest, user_id: int = Depends(get_current_user_id), db: Session = Depends(get_db)):
    return _call(lambda: test_connection(db, user_id, payload.model_dump()))


@router.post("/connections/{connection_id}/discover-models")
def discover(connection_id: int, user_id: int = Depends(get_current_user_id), db: Session = Depends(get_db)):
    return _call(lambda: discover_models(db, user_id, connection_id))


@router.post("/connections/{connection_id}/models")
def create_model(connection_id: int, payload: ModelCreate, user_id: int = Depends(get_current_user_id), db: Session = Depends(get_db)):
    return _call(lambda: add_model(db, user_id, connection_id, payload.model_dump()))


@router.patch("/connections/{connection_id}/models/{model_id}")
def patch_model(connection_id: int, model_id: int, payload: ModelUpdate, user_id: int = Depends(get_current_user_id), db: Session = Depends(get_db)):
    return _call(lambda: update_model(db, user_id, connection_id, model_id, payload.model_dump(exclude_unset=True)))


@router.delete("/connections/{connection_id}/models/{model_id}")
def remove_model(connection_id: int, model_id: int, user_id: int = Depends(get_current_user_id), db: Session = Depends(get_db)):
    return _call(lambda: delete_model(db, user_id, connection_id, model_id))


@router.get("/preferences")
def preferences(user_id: int = Depends(get_current_user_id), db: Session = Depends(get_db)):
    return ok(get_preferences(db, user_id))


@router.patch("/preferences")
def patch_preferences(payload: PreferenceUpdate, user_id: int = Depends(get_current_user_id), db: Session = Depends(get_db)):
    return _call(lambda: update_preferences(db, user_id, payload.default_model_config_id))
