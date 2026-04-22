from functools import lru_cache

from langgraph.graph import END, START, StateGraph
from langgraph.prebuilt import ToolNode, tools_condition

from telegram_agent_aws.application.conversation_service.workflow.edges import should_summarize_conversation
from telegram_agent_aws.application.conversation_service.workflow.nodes import (
    generate_final_response_node,
    generate_text_response_node,
    router_node,
    summarize_conversation_node,
)
from telegram_agent_aws.application.conversation_service.workflow.state import TelegramAgentState
from telegram_agent_aws.application.conversation_service.workflow.tools import get_retriever_tool

@lru_cache(maxsize=1)
def create_workflow_graph():
    graph_builder = StateGraph(TelegramAgentState)

    graph_builder.add_node("router_node", router_node)
    graph_builder.add_node("generate_text_response_node", generate_text_response_node)
    graph_builder.add_node("summarize_conversation_node", summarize_conversation_node)
    graph_builder.add_node("tools", ToolNode([get_retriever_tool()]))
    graph_builder.add_node("generate_final_response_node", generate_final_response_node)

    graph_builder.add_edge(START, "router_node")
    graph_builder.add_edge("router_node", "generate_text_response_node")
    graph_builder.add_conditional_edges("generate_text_response_node", tools_condition, {"tools": "tools", END: "generate_final_response_node"})
    graph_builder.add_edge("tools", "generate_text_response_node")
    graph_builder.add_conditional_edges("generate_final_response_node", should_summarize_conversation)

    graph_builder.add_edge("summarize_conversation_node", END)

    return graph_builder
