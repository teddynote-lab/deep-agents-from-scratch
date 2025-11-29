"""TODO management tools for task planning and progress tracking.

This module provides tools for creating and managing structured task lists
that enable agents to plan complex workflows and track progress through
multi-step operations.
"""

# TODO 리스트 기반 작업 플래닝 및 진행 상황 추적 툴 구현 목적

from typing import Annotated

from langchain_core.messages import ToolMessage
from langchain_core.tools import InjectedToolCallId, tool
from langgraph.prebuilt import InjectedState
from langgraph.types import Command

from deep_agents_from_scratch.prompts import WRITE_TODOS_DESCRIPTION
from deep_agents_from_scratch.state import DeepAgentState, Todo


# write_todos 툴 정의, LLM이 전달한 TODO 리스트를 state에 저장 및 메시지 기록
@tool(description=WRITE_TODOS_DESCRIPTION,parse_docstring=True)
def write_todos(
    todos: list[Todo], tool_call_id: Annotated[str, InjectedToolCallId]
) -> Command:
    """Create or update the agent's TODO list for task planning and tracking.

    Args:
        todos: List of Todo items with content and status
        tool_call_id: Tool call identifier for message response

    Returns:
        Command to update agent state with new TODO list
    """
    # TODO 리스트와 메시지 업데이트를 위한 Command 객체 반환
    return Command(
        update={
            "todos": todos,
            "messages": [
                ToolMessage(f"Updated todo list to {todos}", tool_call_id=tool_call_id)
            ],
        }
    )


# read_todos 툴 정의, 현재 state의 TODO 리스트를 읽어 포맷된 문자열로 반환
@tool(parse_docstring=True)
def read_todos(
    state: Annotated[DeepAgentState, InjectedState],
    tool_call_id: Annotated[str, InjectedToolCallId],
) -> Command:
    """Read the current TODO list from the agent state.

    This tool allows the agent to retrieve and review the current TODO list
    to stay focused on remaining tasks and track progress through complex workflows.

    Args:
        state: Injected agent state containing the current TODO list
        tool_call_id: Injected tool call identifier for message tracking

    Returns:
        Command to update agent state with ToolMessage containing formatted TODO list
    """
    # state에서 todos 리스트 추출, 없으면 빈 리스트 반환
    todos = state.get("todos", [])
    if not todos:
        # TODO 리스트가 비어 있을 때 안내 메시지 반환
        message_content = "No todos currently in the list."
    else:
        # 현재 TODO 리스트를 번호, 이모지, 상태와 함께 포맷팅하여 문자열로 생성
        result = "Current TODO List:\n"
        for i, todo in enumerate(todos, 1):
            status_emoji = {"pending": "⏳", "in_progress": "🔄", "completed": "✅"}
            emoji = status_emoji.get(todo["status"], "❓")
            result += f"{i}. {emoji} {todo['content']} ({todo['status']})\n"
        message_content = result.strip()

    # Command 객체로 래핑하여 ToolMessage와 함께 반환
    return Command(
        update={
            "messages": [
                ToolMessage(message_content, tool_call_id=tool_call_id)
            ],
        }
    )
