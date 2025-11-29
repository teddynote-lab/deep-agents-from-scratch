"""Task delegation tools for context isolation through sub-agents.

This module provides the core infrastructure for creating and managing sub-agents
with isolated contexts. Sub-agents prevent context clash by operating with clean
context windows containing only their specific task description.
"""

from typing import Annotated, NotRequired

from langchain.agents import create_agent
from langchain_core.messages import HumanMessage, ToolMessage
from langchain_core.tools import BaseTool, InjectedToolCallId, tool
from langgraph.prebuilt import InjectedState
from langgraph.types import Command
from typing_extensions import TypedDict

from deep_agents_from_scratch.prompts import TASK_DESCRIPTION_PREFIX
from deep_agents_from_scratch.state import DeepAgentState


class SubAgent(TypedDict):
    """Configuration for a specialized sub-agent."""

    name: str
    description: str
    prompt: str
    tools: NotRequired[list[str]]


def _create_task_tool(tools, subagents: list[SubAgent], model, state_schema):
    """Create a task delegation tool that enables context isolation through sub-agents.

    This function implements the core pattern for spawning specialized sub-agents with
    isolated contexts, preventing context clash and confusion in complex multi-step tasks.

    Args:
        tools: List of available tools that can be assigned to sub-agents
        subagents: List of specialized sub-agent configurations
        model: The language model to use for all agents
        state_schema: The state schema (typically DeepAgentState)

    Returns:
        A 'task' tool that can delegate work to specialized sub-agents
    """
    # Sub-agent 레지스트리 딕셔너리 생성, 이름을 키로 하여 에이전트 인스턴스 저장
    agents = {}

    # 도구 이름별 매핑 딕셔너리 생성, Sub-agent별 도구 할당에 활용
    tools_by_name = {}
    for tool_ in tools:
        if not isinstance(tool_, BaseTool):
            tool_ = tool(tool_)
        tools_by_name[tool_.name] = tool_

    # Sub-agent 구성 정보 기반으로 특화 에이전트 생성 및 레지스트리에 등록
    for _agent in subagents:
        if "tools" in _agent:
            # Sub-agent에 지정된 도구만 할당
            _tools = [tools_by_name[t] for t in _agent["tools"]]
        else:
            # 도구 미지정 시 전체 도구 할당
            _tools = tools
        agents[_agent["name"]] = create_agent(  # updated 1.0
            model,
            system_prompt=_agent["prompt"],
            tools=_tools,
            state_schema=state_schema,
        )

    # 사용 가능한 Sub-agent 목록을 도구 설명에 활용하기 위한 문자열 리스트 생성
    other_agents_string = [
        f"- {_agent['name']}: {_agent['description']}" for _agent in subagents
    ]

    @tool(description=TASK_DESCRIPTION_PREFIX.format(other_agents=other_agents_string))
    def task(
        description: str,
        subagent_type: str,
        state: Annotated[DeepAgentState, InjectedState],
        tool_call_id: Annotated[str, InjectedToolCallId],
    ):
        """Delegate a task to a specialized sub-agent with isolated context.

        This creates a fresh context for the sub-agent containing only the task description,
        preventing context pollution from the parent agent's conversation history.
        """
        # 요청된 Sub-agent 타입이 레지스트리에 존재하는지 검증, 미존재 시 에러 반환
        if subagent_type not in agents:
            return f"Error: invoked agent of type {subagent_type}, the only allowed types are {[f'`{k}`' for k in agents]}"

        # 요청된 Sub-agent 인스턴스 가져오기
        sub_agent = agents[subagent_type]

        # 작업 설명만 포함된 격리된 컨텍스트 생성, 부모 에이전트의 히스토리 미포함
        # HumanMessage 객체 사용 (일반 dict는 LangChain 메시지 검증 실패)
        new_state = {
            "messages": [HumanMessage(content=description)],
            "files": state.get("files", {}),
            "todos": state.get("todos", []),
        }

        # 격리된 환경에서 Sub-agent 실행 및 결과 획득
        result = sub_agent.invoke(new_state)

        # 작업 결과를 Command 객체로 래핑하여 부모 에이전트에 ToolMessage 형태로 반환
        return Command(
            update={
                "files": result.get("files", {}),  # 파일 변경 사항 병합
                "messages": [
                    # Sub-agent의 마지막 메시지를 ToolMessage로 변환하여 반환
                    ToolMessage(
                        result["messages"][-1].content, tool_call_id=tool_call_id
                    )
                ],
            }
        )

    return task
