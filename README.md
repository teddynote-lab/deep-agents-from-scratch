# 🧱 밑바닥부터 만드는 Deep Agent

![Python](https://img.shields.io/badge/python-3.11%2B-blue)
![License](https://img.shields.io/badge/license-MIT-green)

![](./notebooks/assets/agent_header.png)

[Deep Research](https://academy.langchain.com/courses/deep-research-with-langgraph)는 코딩과 함께 최초의 주요 에이전트 사용 사례 중 하나로 부상했습니다. 이제 우리는 광범위한 작업에 사용할 수 있는 범용 에이전트의 출현을 목격하고 있습니다. 예를 들어, [Manus](https://manus.im/blog/Context-Engineering-for-AI-Agents-Lessons-from-Building-Manus)는 장기(long-horizon) 작업을 수행하는 능력으로 큰 주목을 받았으며, 평균적인 Manus 작업은 약 50회의 툴 호출을 사용합니다.

두 번째 예로, Claude Code는 코딩을 넘어선 일반적인 작업에도 사용되고 있습니다.

이러한 인기 있는 "Deep" 에이전트들의 [컨텍스트 엔지니어링 패턴](https://docs.google.com/presentation/d/16aaXLu40GugY-kOpqDU4e-S0hD1FmHcNyF0rRRnb1OU/edit?slide=id.p#slide=id.p)을 주의 깊게 살펴보면 몇 가지 공통적인 접근 방식을 발견할 수 있습니다.

**주요 특징**
* **작업 계획 (예: TODO), 종종 암송(recitation)과 함께 사용됨**
* **파일 시스템으로의 컨텍스트 오프로딩 (Context offloading)**
* **서브 에이전트 위임을 통한 컨텍스트 격리 (Context isolation)**

핸즈온 튜토리얼에서는 LangGraph를 사용하여 이러한 패턴들을 밑바닥부터 구현하는 방법을 보여줍니다.

## 🚀 퀵스타트 (Quickstart)

### 사전 요구 사항 (Prerequisites)

- Python 3.11 이상을 사용해야 합니다.
- 이 버전은 LangGraph와의 최적의 호환성을 위해 필요합니다.
```bash
python3 --version
```
- [uv](https://docs.astral.sh/uv/) 패키지 매니저

**Mac/Linux:**
```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
# 새로운 uv 버전을 사용하기 위해 PATH 업데이트
export PATH="/Users/$USER/.local/bin:$PATH"
```

**Windows:**

Windows 의 경우 아래 링크를 참고해 주세요:
https://teddynote-lab.notion.site/Windows-WSL-2baadc4b25538096a02fdf3e9ec18e4d?source=copy_link

### 설치 (Installation)

1. 저장소 클론:
```bash
git clone https://github.com/teddynote-lab/deep-agents-from-scratch.git
cd deep_agents_from_scratch
```

2. 패키지 및 의존성 설치 (가상 환경을 자동으로 생성하고 관리합니다):
```bash
uv sync
```

3. 프로젝트 루트에 API 키를 포함한 `.env` 파일을 생성합니다:
```bash
# .env 파일 생성
touch .env
```

`.env` 파일에 API 키를 추가하세요:
```env
# 외부 검색을 수행하는 연구 에이전트에 필요
TAVILY_API_KEY=your_tavily_api_key_here

# 모델 사용에 필요
ANTHROPIC_API_KEY=your_anthropic_api_key_here

# 선택 사항: 평가 및 추적용
LANGSMITH_API_KEY=your_langsmith_api_key_here
LANGSMITH_TRACING=true
LANGSMITH_PROJECT=deep-agents-from-scratch
```

4. uv를 사용하여 노트북 또는 코드 실행:
```bash
# Jupyter 노트북 직접 실행
uv run jupyter notebook

# 또는 가상 환경을 활성화하여 실행
source .venv/bin/activate  # Windows의 경우: .venv\Scripts\activate
jupyter notebook
```

## 📚 튜토리얼 개요 (Tutorial Overview)

이 저장소는 고급 AI 에이전트를 구축하는 방법을 가르쳐주는 5개의 단계별 노트북을 포함하고 있습니다:

### `notebooks/01-DeepAgents-Basic.ipynb` - LangGraph를 활용한 ReAct 에이전트
**LangChain**의 `create_agent`를 활용하여 **ReAct** 기반의 AI 에이전트를 구축하는 방법을 다룹니다. 계산기 툴 예제를 통해 에이전트의 핵심 구성 요소와 상태 관리 방법을 학습합니다.
- **ReAct Agent**: Reasoning and Acting 프레임워크의 개념 및 동작 원리
- **Prebuilt Agent**: 메모리, Human-in-the-loop, 스트리밍, 배포 등 주요 기능
- **Tool Integration**: 계산기 툴 정의 및 에이전트 연동 실습

### `notebooks/02-DeepAgents-TODO.ipynb` - 작업 계획 기초 (Task Planning Foundations)
**LangChain**과 **LangGraph**를 활용하여 Deep Agent의 상태 관리와 TODO 리스트 기반의 작업 플로우 설계 방법을 다룹니다. 장기 작업 관리 전략과 커스텀 State 및 툴(`write_todos`, `read_todos`) 구현 과정을 단계별로 설명합니다.
- **TODO 리스트 기반 플래닝**: 장기 작업 수행 시 에이전트의 목표 집중 유도
- **DeepAgentState 설계**: `messages`, `todos`, `files` 등 커스텀 State 구조 및 Reducer 정의
- **TODO 관리 툴 구현**: `write_todos`, `read_todos` 툴의 구현 및 활용

### `notebooks/03-DeepAgents-Context-Offloading.ipynb` - 가상 파일 시스템 (Virtual File Systems)
**Context Offloading** 기법을 활용하여 에이전트의 컨텍스트 윈도우를 효율적으로 관리하고, 가상 파일 시스템을 구축하는 방법을 다룹니다.
- **Context Offloading**: 파일 시스템을 활용한 효율적인 컨텍스트 관리 전략
- **Virtual File System**: LangGraph State 내 가상 파일 시스템 설계 및 구현
- **File Tools**: `ls`, `read_file`, `write_file` 도구 개발 및 프롬프트 설계

### `notebooks/04-DeepAgents-Sub-Agent-Delegation.ipynb` - 컨텍스트 격리 (Context Isolation)
**LangChain**을 활용하여 복잡한 에이전트 시스템에서 **Context Isolation(컨텍스트 격리)** 및 **Task Delegation(작업 위임)** 을 구현하는 방법을 다룹니다.
- **Context Isolation**: Sub-agent를 활용한 독립적 작업 처리 및 컨텍스트 충돌 방지
- **Sub-agent 구성**: `SubAgent` 타입 정의, 프롬프트 및 도구 할당
- **Task Delegation**: Supervisor Agent의 작업 위임 도구(`task`) 개발

### `notebooks/05-DeepAgents-Full-Version.ipynb` - 완전한 연구 에이전트 (Complete Research Agent)
이전까지 알아본 모든 내용을 종합하여 Deep Agent를 완성합니다. **Context Offloading**, **Sub-agent Delegation**, **Strategic Thinking** 을 결합하여 복잡한 리서치 워크플로우를 최적화하는 방법을 다룹니다.
- **Deep Agent 아키텍처**: 파일 기반 컨텍스트 관리 및 토큰 효율화
- **Search Tool**: Tavily API 연동, 웹페이지 요약, 파일 저장
- **Think Tool**: 리서치 진행 상황 분석 및 전략적 의사결정 도구
- **Sub-agent**: Research Sub-agent 구성 및 작업 위임

각 노트북은 이전 개념을 바탕으로 구축되며, 실제 연구 및 분석 작업을 처리할 수 있는 정교한 에이전트 아키텍처로 완성됩니다.

## 참고 자료 (References)

- [Deep Agents UI](https://github.com/langchain-ai/deep-agents-ui)
- [Deep Agents Repository](https://github.com/langchain-ai/deepagents)
- [Deep Agents Documentation](https://docs.langchain.com/oss/python/deepagents/overview)

## 라이선스 (License)

이 프로젝트는 MIT 라이선스 하에 배포됩니다. 자세한 내용은 [LICENSE](LICENSE) 파일을 참조하세요.
