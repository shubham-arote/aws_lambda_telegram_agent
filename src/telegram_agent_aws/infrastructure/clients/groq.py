from functools import lru_cache

from telegram_agent_aws.config import settings

from groq import Groq
@lru_cache(maxsize=1)
def get_groq_client() -> Groq:
    """
    Get or create the Groq client singleton.
    The client is created once and cached for subsequent calls.
    """
    return Groq(
        # This is the default and can be omitted
        api_key=settings.GROQ_API_KEY,
    )