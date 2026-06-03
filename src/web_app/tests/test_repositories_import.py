def test_repositories_import():
    from src.web_app.db.repositories.agent_repository import AgentRunRepository, AgentStepRepository
    from src.web_app.db.repositories.approval_repository import ApprovalRepository
    from src.web_app.db.repositories.artifact_repository import ArtifactRepository
    from src.web_app.db.repositories.feed_repository import FeedRepository
    from src.web_app.db.repositories.memory_repository import MemoryRepository
    from src.web_app.db.repositories.profile_repository import ProfileRepository
    from src.web_app.db.repositories.skill_repository import SkillRepository
    from src.web_app.db.repositories.user_repository import UserRepository

    assert all([AgentRunRepository, AgentStepRepository, ApprovalRepository, ArtifactRepository, FeedRepository, MemoryRepository, ProfileRepository, SkillRepository, UserRepository])
