import random

from langchain_core.messages import HumanMessage, RemoveMessage, SystemMessage
from langchain_groq import ChatGroq
from langchain_openai import ChatOpenAI  # noqa: F401 (kept intentionally)
from pydantic import BaseModel, Field  # noqa: F401 (kept intentionally)

from telegram_agent_aws.application.conversation_service.workflow.state import TelegramAgentState
from telegram_agent_aws.application.conversation_service.workflow.tools import get_retriever_tool
from telegram_agent_aws.config import settings
from telegram_agent_aws.domain.prompts import ROUTER_SYSTEM_PROMPT, SYSTEM_PROMPT
from telegram_agent_aws.infrastructure.clients.elevenlabs import get_elevenlabs_client
from telegram_agent_aws.infrastructure.clients.openai import get_openai_client  # noqa: F401 (kept intentionally)

elevenlabs_client = get_elevenlabs_client()

_ROUTER_SUFFIX = "\nReply with only one word: 'text' or 'audio'."


def router_node(state: TelegramAgentState):
    llm = ChatGroq(model=settings.GROQ_MODEL, api_key=settings.GROQ_API_KEY)

    sys_msg = SystemMessage(content=ROUTER_SYSTEM_PROMPT.prompt + _ROUTER_SUFFIX)
    response = llm.invoke([sys_msg, state["messages"][-1]])

    response_type = "audio" if "audio" in response.content.lower() else "text"

    if response_type == "text" and random.random() > 0.5:
        return {"response_type": "audio"}

    return {"response_type": response_type}


def generate_text_response_node(state: TelegramAgentState):
    llm = ChatGroq(model=settings.GROQ_MODEL, api_key=settings.GROQ_API_KEY)
    llm_with_tools = llm.bind_tools([get_retriever_tool()])

    summary = state.get("summary", "")

    if summary:
        system_message = f"{SYSTEM_PROMPT.prompt} \n Summary of conversation earlier: {summary}"
        messages = [SystemMessage(content=system_message)] + state["messages"]

    else:
        messages = [SystemMessage(content=SYSTEM_PROMPT.prompt)] + state["messages"]

    response = llm_with_tools.invoke(messages)

    return {"messages": response}


def summarize_conversation_node(state: TelegramAgentState):
    llm = ChatGroq(model=settings.GROQ_MODEL, api_key=settings.GROQ_API_KEY)

    summary = state.get("summary", "")

    if summary:
        summary_message = f"This is summary of the conversation to date: {summary}\n\nExtend the summary by taking into account the new messages above:"

    else:
        summary_message = "Create a summary of the conversation above:"

    messages = state["messages"] + [HumanMessage(content=summary_message)]
    response = llm.invoke(messages)

    delete_messages = [RemoveMessage(id=m.id) for m in state["messages"][:-2]]

    return {"summary": response.content, "messages": delete_messages}


def generate_final_response_node(state: TelegramAgentState):
    if state["response_type"] == "audio":
        audio = elevenlabs_client.text_to_speech.convert(
            text=state["messages"][-1].content,
            voice_id=settings.ELEVENLABS_VOICE_ID,
            model_id=settings.ELEVENLABS_MODEL_ID,
        )

        audio_bytes = b"".join(audio)

        return {"audio_buffer": audio_bytes}

    else:
        return state
