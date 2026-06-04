from src.web_app.db.repositories.info_repository import InfoItemRepository
from src.web_app.models.orm import InfoItem
from src.web_app.tests.db_test_utils import make_test_session


def test_get_by_content_hash_tolerates_historical_duplicates():
    db = make_test_session()
    db.add_all(
        [
            InfoItem(title="A", summary="", content="", source_url="https://example.com/a", content_hash="same"),
            InfoItem(title="B", summary="", content="", source_url="https://example.com/b", content_hash="same"),
        ]
    )
    db.commit()

    item = InfoItemRepository(db).get_by_content_hash("same")

    assert item
    assert item.title == "A"
