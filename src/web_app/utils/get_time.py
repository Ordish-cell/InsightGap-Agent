from datetime import datetime

from langchain_core.tools import tool


@tool
def get_time() -> str:
    """获取当前的本地时间"""
    local_time = datetime.datetime.now().strftime("%I:%M %p")
    return local_time