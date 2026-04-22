from langgraph.checkpoint.mongodb import MongoDBSaver

from telegram_agent_aws.application.conversation_service.workflow.graph import create_workflow_graph
from telegram_agent_aws.infrastructure.clients.mongodb import get_mongodb_client


def get_agent_response(payload: dict, user_id: int):
    checkpointer = MongoDBSaver(get_mongodb_client())

    graph = create_workflow_graph().compile(checkpointer=checkpointer)

    config = {"configurable": {"thread_id": str(user_id)}}

    return graph.invoke(payload, config)
