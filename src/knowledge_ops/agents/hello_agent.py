"""
Phase 1: Hello World agent.

The simplest possible "agent": one question in, one LLM call, one answer
out. No tools, no memory, no documents. Its only job is to prove that the
environment, the API key, and the LangChain wiring all work — and to do it
transparently, by printing exactly what's sent to the model and exactly
what comes back.

Later phases replace the "answer directly" step with retrieval-augmented
answers (Phase 4) and eventually a multi-node LangGraph pipeline
(Phase 5), but the shape of "ask -> show your work -> answer" stays.
"""

from langchain_core.messages import HumanMessage, SystemMessage

from knowledge_ops.config import get_llm

SYSTEM_PROMPT = (
    "You are a helpful, concise assistant for a knowledge-operations "
    "project that is being built step by step. Answer plainly. If you "
    "don't know something, say so instead of guessing."
)


def ask(question: str, llm=None) -> str:
    """Send `question` to the LLM and return its text answer.

    Prints the outgoing prompt and the incoming response so the exchange
    is fully visible, not just the final answer.
    """
    llm = llm or get_llm()
    messages = [SystemMessage(content=SYSTEM_PROMPT), HumanMessage(content=question)]

    print("\n--- prompt sent to model ---")
    for message in messages:
        print(f"[{message.type}] {message.content}")

    response = llm.invoke(messages)

    print("--- raw model response ---")
    print(response.content)
    print("--- end ---\n")

    return response.content


def main() -> None:
    print("Hello World agent (Phase 1). Type 'exit' or 'quit' to stop.\n")
    llm = get_llm()
    while True:
        try:
            question = input("You: ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nGoodbye.")
            break

        if not question:
            continue
        if question.lower() in {"exit", "quit"}:
            print("Goodbye.")
            break

        answer = ask(question, llm=llm)
        print(f"Agent: {answer}\n")


if __name__ == "__main__":
    main()
