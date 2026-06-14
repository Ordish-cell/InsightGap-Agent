from __future__ import annotations


LOCAL_TOOL_PATTERNS: dict[str, tuple[str, ...]] = {
    "system.time": (
        "今天几号",
        "今天是几月几号",
        "今天星期几",
        "现在几点",
        "当前时间",
        "当前日期",
        "现在时间",
        "当前时区",
        "what time is it",
        "what date is it",
        "today's date",
        "todays date",
    ),
    "system.calc": (
        "计算",
        "算一下",
        "加减乘除",
        "百分比",
        "利率",
    ),
    "system.unit_convert": (
        "单位换算",
        "公里转英里",
        "摄氏度转华氏度",
        "摄氏",
        "华氏",
        "mb 等于多少 gb",
        "mb等于多少gb",
        "kg转lb",
    ),
    "system.uuid": (
        "生成uuid",
        "生成 uuid",
        "生成唯一id",
        "生成唯一 id",
    ),
    "system.hash": (
        "md5",
        "sha256",
        "hash一下",
        "hash 一个",
        "哈希",
    ),
}

_EXPLICIT_WEB_SEARCH_TERMS = (
    "联网",
    "联网搜索",
    "联网查",
    "上网查",
    "上网搜",
    "搜索",
    "搜一个",
    "搜索一个",
    "查最新",
    "查一下最新",
    "web search",
    "search the web",
    "internet search",
)

_REALTIME_DOMAIN_TERMS = (
    "新闻",
    "消息",
    "价格",
    "股价",
    "行情",
    "天气",
    "赛程",
    "比赛",
    "政策",
    "法规",
    "官网",
    "版本",
    "模型",
    "model",
    "models",
    "发布",
    "发布日期",
    "可用性",
    "排行榜",
    "实时数据",
    "release",
    "latest version",
    "recent news",
    "price",
    "stock price",
    "weather",
    "schedule",
    "policy",
)

_TEMPORAL_TERMS = (
    "今天",
    "今日",
    "现在",
    "当前",
    "实时",
    "最新",
    "today",
    "current",
    "latest",
    "recent",
    "now",
)


def _normalize(text: str) -> str:
    return (text or "").lower().strip()


def _compact(text: str) -> str:
    return "".join(_normalize(text).split())


def detect_local_tool(text: str) -> str | None:
    normalized = _normalize(text)
    compact = _compact(text)
    if not normalized:
        return None
    for tool_name, patterns in LOCAL_TOOL_PATTERNS.items():
        for pattern in patterns:
            pattern_norm = _normalize(pattern)
            if pattern_norm and (pattern_norm in normalized or _compact(pattern_norm) in compact):
                return tool_name
    return None


def is_explicit_or_realtime_web_query(text: str) -> bool:
    normalized = _normalize(text)
    if not normalized:
        return False
    explicit_search = any(term in normalized for term in _EXPLICIT_WEB_SEARCH_TERMS)
    realtime_domain = any(term in normalized for term in _REALTIME_DOMAIN_TERMS)
    temporal = any(term in normalized for term in _TEMPORAL_TERMS)
    return explicit_search or (temporal and realtime_domain)
