from src.web_app.services.scoring_service import calculate_final_score, should_show_card


def test_scoring_formula_and_hard_filter():
    score = calculate_final_score(
        {
            "personal_relevance": 0.31,
            "novelty": 0.74,
            "cross_domain_distance": 0.65,
            "opportunity_value": 0.69,
            "source_credibility": 0.82,
            "actionability": 0.68,
        }
    )
    assert score == 0.592
    assert should_show_card({"personal_relevance": 0.14}) is False
