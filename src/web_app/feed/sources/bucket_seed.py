import hashlib

from src.web_app.feed.sources.base import FeedSource, RawFeedItem

BUCKET_SEEDS: list[dict] = [
    # Explicit domain seeds — used by full_seed_fallback when real search fails
    {
        "title": "LangGraph memory and tool routing patterns for Agent OS",
        "summary": "LangGraph provides graph-based orchestration primitives for building stateful agent memory systems and tool routing patterns.",
        "url": "https://github.com/langchain-ai/langgraph",
        "source_type": "web",
        "tags": ["agent", "langgraph", "memory", "tool routing"],
        "domain_hints": ["agent", "langgraph", "devtools"],
        "search_bucket": "explicit_related",
    },
    {
        "title": "Dify workflow and agent application builder as Agent OS reference",
        "summary": "Dify provides visual workflow orchestration, agent application templates, and RAG pipeline components useful for building an Agent OS.",
        "url": "https://github.com/langgenius/dify",
        "source_type": "web",
        "tags": ["agent", "workflow", "llm app", "tool orchestration"],
        "domain_hints": ["agent", "workflow", "devtools"],
        "search_bucket": "explicit_related",
    },
    {
        "title": "RAGAS evaluation patterns for RAG system quality",
        "summary": "RAGAS provides standardized evaluation metrics for RAG systems, covering retrieval quality, answer faithfulness and relevance.",
        "url": "https://github.com/explodinggradients/ragas",
        "source_type": "web",
        "tags": ["rag", "evaluation", "quality"],
        "domain_hints": ["rag", "research", "devtools"],
        "search_bucket": "explicit_related",
    },
    {
        "title": "MCP server ecosystem as tool protocol reference",
        "summary": "The Model Context Protocol standardizes how AI agents connect to tools and data, creating a reusable integration ecosystem.",
        "url": "https://modelcontextprotocol.io/",
        "source_type": "web",
        "tags": ["mcp", "tool protocol", "agent tools"],
        "domain_hints": ["mcp", "agent", "devtools"],
        "search_bucket": "explicit_related",
    },
    # Adjacent domain seeds
    {
        "title": "LLM observability and tracing are becoming standard for AI products",
        "summary": "LLM tracing, evaluation dashboards and observability pipelines help teams debug multi-step AI workflows and catch quality regressions.",
        "url": "https://opentelemetry.io/",
        "source_type": "web",
        "tags": ["llm observability", "tracing", "workflow", "quality"],
        "domain_hints": ["observability", "workflow"],
        "search_bucket": "adjacent_domain",
    },
    {
        "title": "Human-in-the-loop approval workflows are becoming a key pattern for AI automation",
        "summary": "Approval gates, review queues and audit trails are being used to make AI workflow automation safer and more controllable.",
        "url": "https://temporal.io/",
        "source_type": "web",
        "tags": ["workflow automation", "human in the loop", "approval", "audit"],
        "domain_hints": ["workflow automation", "approval"],
        "search_bucket": "adjacent_domain",
    },
    {
        "title": "Browser automation with Playwright is becoming a practical layer for AI workflow execution",
        "summary": "Browser automation frameworks can give AI systems a controlled way to operate web workflows without full desktop control.",
        "url": "https://playwright.dev/",
        "source_type": "web",
        "tags": ["browser automation", "playwright", "workflow execution"],
        "domain_hints": ["browser automation"],
        "search_bucket": "adjacent_domain",
    },
    {
        "title": "Developer productivity teams are measuring AI coding assistant impact with workflow metrics",
        "summary": "Engineering teams are moving from anecdotal AI coding gains to measurable developer workflow metrics and review quality signals.",
        "url": "https://github.blog/",
        "source_type": "web",
        "tags": ["developer productivity", "AI coding assistant", "workflow metrics"],
        "domain_hints": ["developer productivity"],
        "search_bucket": "adjacent_domain",
    },
    {
        "title": "Personal knowledge management is shifting toward AI-assisted knowledge graphs",
        "summary": "PKM tools increasingly use embeddings, graph structure and AI summarization to organize personal and team knowledge.",
        "url": "https://www.mem.ai/",
        "source_type": "web",
        "tags": ["personal knowledge management", "knowledge graph", "AI knowledge base"],
        "domain_hints": ["knowledge management"],
        "search_bucket": "adjacent_domain",
    },
    {
        "title": "Prompt and context management is becoming its own operations layer",
        "summary": "Teams are separating prompt versions, context assembly, evaluation and rollout workflows as a dedicated AI operations layer.",
        "url": "https://langfuse.com/",
        "source_type": "web",
        "tags": ["prompt management", "context engineering", "evaluation"],
        "domain_hints": ["context engineering"],
        "search_bucket": "adjacent_domain",
    },
    {
        "title": "产品分析团队用 AI 反馈环发现流失与机会信号",
        "summary": "产品分析工具将用户行为转化为弱信号，用于优先级判断、留存分析和机会发现。",
        "url": "https://posthog.com/",
        "source_type": "web",
        "tags": ["product analytics", "user feedback", "opportunity signal"],
        "domain_hints": ["product analytics", "feedback loop"],
        "search_bucket": "far_domain",
    },
    {
        "title": "竞争情报正在从报告转向连续市场信号监控",
        "summary": "市场情报工作流正在转向对竞争对手、客户和公开数据的持续信号检测。",
        "url": "https://www.crayon.co/",
        "source_type": "web",
        "tags": ["competitive intelligence", "market signal", "industry intelligence"],
        "domain_hints": ["competitive intelligence"],
        "search_bucket": "far_domain",
    },
    {
        "title": "教育 AI 产品正在走向自适应学习反馈回路",
        "summary": "自适应学习系统使用学生互动信号来个性化学习路径、发现知识缺口并提升留存率。",
        "url": "https://www.khanacademy.org/khan-labs",
        "source_type": "web",
        "tags": ["education AI", "adaptive learning", "feedback loop"],
        "domain_hints": ["education product"],
        "search_bucket": "far_domain",
    },
    {
        "title": "投资研究团队用另类数据发现早期市场信号",
        "summary": "另类数据工作流帮助分析师在传统报告之前检测到弱信号。",
        "url": "https://www.alpha-sense.com/",
        "source_type": "web",
        "tags": ["investment research", "alternative data", "market signal"],
        "domain_hints": ["investment research"],
        "search_bucket": "far_domain",
    },
    {
        "title": "企业知识运营正在变成连续工作流系统",
        "summary": "知识运营团队从静态文档转向持续采集、路由和复用机构知识。",
        "url": "https://www.atlassian.com/software/confluence",
        "source_type": "web",
        "tags": ["enterprise knowledge management", "knowledge operations", "workflow"],
        "domain_hints": ["knowledge ops"],
        "search_bucket": "far_domain",
    },
    {
        "title": "PLG 团队用行为信号发现扩张机会",
        "summary": "PLG 系统分析使用模式和激活信号来识别留存、增购和入驻机会。",
        "url": "https://www.pendo.io/",
        "source_type": "web",
        "tags": ["product-led growth", "behavioral analytics", "opportunity discovery"],
        "domain_hints": ["growth analytics"],
        "search_bucket": "far_domain",
    },
]


class BucketSeedSource(FeedSource):
    name = "bucket_seed_source"
    source_type = "web"

    def __init__(self, buckets: list[str] | None = None):
        self.enabled = True
        self._buckets = buckets or []

    async def fetch(self) -> list[RawFeedItem]:
        items: list[RawFeedItem] = []
        for seed in BUCKET_SEEDS:
            bucket = seed.get("search_bucket", "")
            if self._buckets and bucket not in self._buckets:
                continue
            source_id = "bucket_seed:" + hashlib.sha256(seed["url"].encode()).hexdigest()[:16]
            items.append(RawFeedItem(
                source_id=source_id,
                source_type=seed.get("source_type", "web"),
                title=seed["title"],
                summary=seed.get("summary", ""),
                url=seed.get("url"),
                tags=seed.get("tags", []),
                domain_hints=seed.get("domain_hints", []),
                search_bucket=bucket,
                source_kind="bucket_seed",
                provider="bucket_seed",
            ))
        return items
