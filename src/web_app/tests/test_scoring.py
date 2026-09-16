"""Scoring contracts exercised through the scorer used by the Feed service."""
from types import SimpleNamespace

from src.web_app.feed.scorer import FeedScorer


def score_candidate():
    item = SimpleNamespace(title="Agent workflow", summary="A workflow opportunity",
        topics=["agent"], raw_metadata={"source_credibility": 0.8},
        published_at=None, source_url="https://example.test/article")
    profile = SimpleNamespace(explicit_interests=["agent"], goals=[], adjacent_domains=[],
        far_domains=[], disliked_topics=[])
    return FeedScorer().score(item, profile)


def test_calculate_final_score():
    result = score_candidate()
    expected = round(0.30 * result["personal_relevance"] + 0.20 * result["novelty"]
        + 0.15 * result["cross_domain_distance"] + 0.15 * result["opportunity_value"]
        + 0.10 * result["source_credibility"] + 0.10 * result["actionability"], 4)
    assert result["final"] == expected


def test_classify_exposure_bucket():
    assert score_candidate()["relation_type"] == "explicit_related"
