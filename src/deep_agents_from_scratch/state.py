"""State management for deep agents with TODO tracking and virtual file systems.

This module defines the extended agent state structure that supports:
- Task planning and progress tracking through TODO lists
- Context offloading through a virtual file system stored in state
- Efficient state merging with reducer functions
"""

from typing import Annotated, Literal, NotRequired
from typing_extensions import TypedDict

#from langgraph.prebuilt.chat_agent_executor import AgentState
from langchain.agents import AgentState  # updated in 1.0

# 복잡한 작업 플로우의 진행 상황 추적을 위한 TODO 항목 구조 정의
class Todo(TypedDict):
    """A structured task item for tracking progress through complex workflows.

    Attributes:
        content: Short, specific description of the task
        status: Current state - pending, in_progress, or completed
    """

    content: str
    status: Literal["pending", "in_progress", "completed"]

# 두 파일 딕셔너리 병합, 오른쪽 값이 우선 적용되는 가상 파일 시스템 업데이트용 reducer 함수
def file_reducer(left, right):
    """Merge two file dictionaries, with right side taking precedence.

    Used as a reducer function for the files field in agent state,
    allowing incremental updates to the virtual file system.

    Args:
        left: Left side dictionary (existing files)
        right: Right side dictionary (new/updated files)

    Returns:
        Merged dictionary with right values overriding left values
    """
    if left is None:
        return right
    elif right is None:
        return left
    else:
        return {**left, **right}

# LangGraph AgentState 상속, TODO 리스트와 가상 파일 시스템 포함한 확장 state 구조 정의
class DeepAgentState(AgentState):
    """Extended agent state that includes task tracking and virtual file system.

    Inherits from LangGraph's AgentState and adds:
    - todos: List of Todo items for task planning and progress tracking
    - files: Virtual file system stored as dict mapping filenames to content
    """

    # 작업 플래닝 및 진행 상황 추적을 위한 Todo 리스트 필드
    todos: NotRequired[list[Todo]]
    # 파일명과 내용 매핑, file_reducer로 병합되는 가상 파일 시스템 필드
    files: Annotated[NotRequired[dict[str, str]], file_reducer]
