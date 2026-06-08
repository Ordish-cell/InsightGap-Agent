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
        "title": "Product analytics teams use AI feedback loops to detect churn and opportunity signals",
        "summary": "AI product analytics turns user behavior into weak signals for prioritization, retention and opportunity discovery.",
        "url": "https://posthog.com/",
        "source_type": "web",
        "tags": ["product analytics", "user feedback", "opportunity signal"],
        "domain_hints": ["product analytics", "feedback loop"],
        "search_bucket": "far_domain",
    },
    {
        "title": "Competitive intelligence is shifting from reports to continuous market signal monitoring",
        "summary": "Market intelligence workflows are moving toward continuous signal detection across competitors, customers and public data.",
        "url": "https://www.crayon.co/",
        "source_type": "web",
        "tags": ["competitive intelligence", "market signal", "industry intelligence"],
        "domain_hints": ["competitive intelligence"],
        "search_bucket": "far_domain",
    },
    {
        "title": "Education AI products are moving toward adaptive learning feedback loops",
        "summary": "Adaptive learning systems use student interaction signals to personalize next steps, surface gaps and improve retention.",
        "url": "https://www.khanacademy.org/khan-labs",
        "source_type": "web",
        "tags": ["education AI", "adaptive learning", "feedback loop"],
        "domain_hints": ["education product"],
        "search_bucket": "far_domain",
    },
    {
        "title": "Investment research teams are using alternative data to discover early market signals",
        "summary": "Alternative data workflows help analysts detect weak signals before they appear in traditional reports.",
        "url": "https://www.alpha-sense.com/",
        "source_type": "web",
        "tags": ["investment research", "alternative data", "market signal"],
        "domain_hints": ["investment research"],
        "search_bucket": "far_domain",
    },
    {
        "title": "Enterprise knowledge operations are becoming continuous workflow systems",
        "summary": "Knowledge operations teams are moving from static documentation to continuous capture, routing and reuse of institutional knowledge.",
        "url": "https://www.atlassian.com/software/confluence",
        "source_type": "web",
        "tags": ["enterprise knowledge management", "knowledge operations", "workflow"],
        "domain_hints": ["knowledge ops"],
        "search_bucket": "far_domain",
    },
    {
        "title": "Product-led growth teams use behavioral signals to find expansion opportunities",
        "summary": "PLG systems analyze usage patterns and activation signals to identify retention, upsell and onboarding opportunities.",
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
