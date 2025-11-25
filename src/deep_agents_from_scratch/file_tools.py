"""Virtual file system tools for agent state management.

This module provides tools for managing a virtual filesystem stored in agent state,
enabling context offloading and information persistence across agent interactions.
"""

from typing import Annotated

from langchain_core.messages import ToolMessage
from langchain_core.tools import InjectedToolCallId, tool
from langgraph.prebuilt import InjectedState
from langgraph.types import Command

from deep_agents_from_scratch.prompts import (
    LS_DESCRIPTION,
    READ_FILE_DESCRIPTION,
    WRITE_FILE_DESCRIPTION,
)
from deep_agents_from_scratch.state import DeepAgentState


# 가상 파일 시스템 내 파일 목록을 반환합니다.
@tool(description=LS_DESCRIPTION)
def ls(state: Annotated[DeepAgentState, InjectedState]) -> list[str]:
    """List all files in the virtual filesystem."""
    return list(state.get("files", {}).keys())


# 파일을 읽어 반환합니다. 오프셋과 최대 라인 수를 지정할 수 있습니다.
@tool(description=READ_FILE_DESCRIPTION, parse_docstring=True)
def read_file(
    file_path: str,
    state: Annotated[DeepAgentState, InjectedState],
    offset: int = 0,
    limit: int = 2000,
) -> str:
    """Read file content from virtual filesystem with optional offset and limit.

    Args:
        file_path: Path to the file to read
        state: Agent state containing virtual filesystem (injected in tool node)
        offset: Line number to start reading from (default: 0)
        limit: Maximum number of lines to read (default: 2000)

    Returns:
        Formatted file content with line numbers, or error message if file not found
    """
    # 파일 시스템에서 파일을 조회합니다.
    files = state.get("files", {})
    if file_path not in files:
        return f"Error: File '{file_path}' not found"

    # 파일이 비어 있으면 안내 메시지를 반환합니다.
    content = files[file_path]
    if not content:
        return "System reminder: File exists but has empty contents"

    # 파일 내용을 줄 단위로 분할하고 오프셋 및 제한을 적용합니다.
    lines = content.splitlines()
    start_idx = offset
    end_idx = min(start_idx + limit, len(lines))

    # 오프셋이 파일 길이를 초과하면 에러 메시지를 반환합니다.
    if start_idx >= len(lines):
        return f"Error: Line offset {offset} exceeds file length ({len(lines)} lines)"

    # 각 줄에 번호를 붙이고 2000자까지 잘라서 결과를 만듭니다.
    result_lines = []
    for i in range(start_idx, end_idx):
        line_content = lines[i][:2000]
        result_lines.append(f"{i + 1:6d}\t{line_content}")

    # 결과를 문자열로 합쳐 반환합니다.
    return "\n".join(result_lines)


# 파일에 내용을 기록하고 상태를 갱신하는 커맨드를 반환합니다.
@tool(description=WRITE_FILE_DESCRIPTION, parse_docstring=True)
def write_file(
    file_path: str,
    content: str,
    state: Annotated[DeepAgentState, InjectedState],
    tool_call_id: Annotated[str, InjectedToolCallId],
) -> Command:
    """Write content to a file in the virtual filesystem.

    Args:
        file_path: Path where the file should be created/updated
        content: Content to write to the file
        state: Agent state containing virtual filesystem (injected in tool node)
        tool_call_id: Tool call identifier for message response (injected in tool node)

    Returns:
        Command to update agent state with new file content
    """
    # 파일 시스템에 파일을 저장합니다.
    files = state.get("files", {})
    files[file_path] = content
    # 파일 업데이트 메시지와 함께 커맨드를 반환합니다.
    return Command(
        update={
            "files": files,
            "messages": [
                ToolMessage(f"Updated file {file_path}", tool_call_id=tool_call_id)
            ],
        }
    )
