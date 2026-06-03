from src.web_app.models.orm import UserProfile


def test_profile_default_feed_ratio():
    profile = UserProfile(user_id=1)
    assert profile.feed_ratio_config == {"explicit_related": 0.30, "adjacent_domain": 0.40, "far_domain": 0.30}
