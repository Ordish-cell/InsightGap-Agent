from sqlalchemy.orm import Session

from src.web_app.db.repositories.profile_repository import ProfileRepository
from src.web_app.services.deletion_guard import guarded_transition

ALLOWED_SEGMENTS = {"ai_developer", "entrepreneur", "general_user", "researcher"}


def profile_to_dict(profile) -> dict:
    return {
        "id": profile.id,
        "user_id": profile.user_id,
        "segment": profile.segment,
        "goals": profile.goals or [],
        "explicit_interests": profile.explicit_interests or [],
        "adjacent_domains": profile.adjacent_domains or [],
        "far_domains": profile.far_domains or [],
        "disliked_topics": profile.disliked_topics or [],
        "preferred_outputs": profile.preferred_outputs or [],
        "risk_preference": profile.risk_preference,
        "feed_ratio_config": profile.feed_ratio_config or {"explicit_related": 0.30, "adjacent_domain": 0.40, "far_domain": 0.30},
        "use_memory": profile.use_memory,
        "generate_memory": profile.generate_memory,
    }


def get_profile(db: Session, user_id: int) -> dict:
    return profile_to_dict(ProfileRepository(db).get_or_create_default(user_id))


@guarded_transition
def update_profile(db: Session, user_id: int, payload: dict) -> dict:
    if payload.get("segment") and payload["segment"] not in ALLOWED_SEGMENTS:
        raise ValueError("Invalid segment")
    repo = ProfileRepository(db)
    profile = repo.get_or_create_default(user_id)
    allowed = {
        "use_memory",
        "generate_memory",
        "segment",
        "goals",
        "explicit_interests",
        "adjacent_domains",
        "far_domains",
        "disliked_topics",
        "preferred_outputs",
        "risk_preference",
        "feed_ratio_config",
    }
    for key in ("use_memory", "generate_memory"):
        if key in payload and not isinstance(payload[key], bool):
            raise ValueError(f"{key} must be boolean")
    values = {k: v for k, v in payload.items() if k in allowed}
    if "generate_memory" in values and values["generate_memory"] != profile.generate_memory:
        values["memory_settings_version"] = profile.memory_settings_version + 1
    repo.update(profile, **values)
    return profile_to_dict(profile)
