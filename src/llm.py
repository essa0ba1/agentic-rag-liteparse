import logging

from langchain_openai import ChatOpenAI

import config

logger = logging.getLogger(__name__)


def get_llm() -> ChatOpenAI:
    logger.info("Initializing LLM at %s", config.LLM_BASE_URL)
    return ChatOpenAI(
        openai_api_base=config.LLM_BASE_URL,
        openai_api_key=config.LLM_API_KEY,
        temperature=0.2,
    )
