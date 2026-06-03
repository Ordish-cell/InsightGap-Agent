from src.web_app.services.scoring_service import calculate_final_score, classify_exposure_bucket


def test_calculate_final_score():
    score = calculate_final_score(
        {
            "personal_relevance": 1,
            "novelty": 1,
            "cross_domain_distance": 1,
            "opportunity_value": 1,
            "source_credibility": 1,
            "actionability": 1,
        }
    )
    assert score == 1.0


def test_classify_exposure_bucket():
    assert classify_exposure_bucket({"topics": ["ai"]}, {"explicit_interests": ["ai"], "adjacent_domains": []}) == "explicit_related"
