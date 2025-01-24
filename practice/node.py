from langgraph.types import StreamWriter
from model import llm
from state import GraphState


def filter_node(state:GraphState, writer: StreamWriter) -> GraphState:
    print("[Filter Node] AI가 질문을 식별중입니다!!!!")
    system_prompt = "당신은 사용자를 도와주는 한국어 챗봇입니다."
    response = llm.invoke([
        SystemMessage(content=system_prompt),
        HumanMessage(state["messages"][-1].content)
    ])

    async for chunk in llm.astream(
        {"context": documents, "question": question}
    ):
        writer(chunk)
        chunks.append(chunk)

    messages = response.content.strip()

    return {"messages": messages}