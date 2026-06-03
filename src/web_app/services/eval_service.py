def create_eval_record(target_type: str, target_id: str, metrics: dict) -> dict:
    return {"target_type": target_type, "target_id": target_id, "metrics": metrics}
