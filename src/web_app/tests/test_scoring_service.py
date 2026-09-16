"""Feed relevance filtering uses the production scorer and configured threshold."""
from src.web_app.core.config import settings
from src.web_app.tests.test_scoring import score_candidate


def test_scoring_formula_and_hard_filter(monkeypatch):
    relevance = score_candidate()["personal_relevance"]
    monkeypatch.setattr(settings, "feed_min_personal_relevance", relevance + 0.01)
    assert score_candidate()["filtered"] is True
    monkeypatch.setattr(settings, "feed_min_personal_relevance", relevance)
    assert score_candidate()["filtered"] is False
