
"""
Phase 6 — Conversation Memory Utilities

This module handles:
- Unique conversation IDs
- LangGraph checkpoint configuration
- Recent conversation history
- Follow-up question rewriting
"""

import uuid

from langgraph.checkpoint.memory import MemorySaver

from config.settings import (
    MAX_HISTORY_MESSAGES,
    NVIDIA_MODEL,
)


def create_memory():
    """
    Create a LangGraph in-memory checkpointer.

    Checkpoints are associated with thread IDs.

    Important:
        MemorySaver stores state in server memory.
        It does not survive a server restart.
    """

    return MemorySaver()


def create_thread_id():
    """
    Generate a unique ID for one conversation.

    Different IDs keep separate conversations
    from sharing the same checkpoint state.
    """

    return str(uuid.uuid4())


def create_thread_config(thread_id):
    """
    Build the configuration LangGraph expects.

    The same thread_id must be used on subsequent
    questions within one conversation.
    """

    return {
        "configurable": {
            "thread_id": thread_id
        }
    }


def trim_history(history):
    """
    Keep only the most recent conversation messages.

    This controls how much history is sent to
    the follow-up rewriting model.
    """

    return list(history)[-MAX_HISTORY_MESSAGES:]


def append_exchange(history, question, answer):
    """
    Add a completed user-assistant exchange.

    We store copies rather than modifying the
    existing history list in place.
    """

    updated = list(history)

    updated.append({
        "role": "user",
        "content": question,
    })

    updated.append({
        "role": "assistant",
        "content": answer,
    })

    return trim_history(updated)


def rewrite_followup_question(
    question,
    history,
    llm_client,
):
    """
    Rewrite the latest question using conversation
    context.

    Example:
        History: "How do I renew my IRP?"
        Latest: "How much does it cost?"

    Desired standalone question:
        "How much does renewing an Irish Residence
         Permit cost?"

    The rewritten question is used for retrieval,
    not as factual evidence.
    """

    # No history means there is nothing to resolve.
    if not history:
        return question

    # The system message defines a narrow task:
    # rewrite the question, but do not answer it.

    messages = [
        {
            "role": "system",
            "content": (
                "You rewrite follow-up questions for "
                "an Irish public-services assistant. "
                "Use previous conversation turns to "
                "resolve references such as 'it', "
                "'that' or 'one'. "
                "If the user changes topics, preserve "
                "the new topic. "
                "Do not answer the question. "
                "Do not invent facts. "
                "Return only a standalone question."
            ),
        }
    ]

    # Supply only the recent history.
    for item in trim_history(history):

        role = item.get("role")
        content = item.get("content", "")

        if role in ("user", "assistant") and content:

            messages.append({
                "role": role,
                "content": content,
            })

    # The final message is the latest question.
    messages.append({
        "role": "user",
        "content": question,
    })

    try:
        response = llm_client.chat.completions.create(
            model=NVIDIA_MODEL,
            messages=messages,
            temperature=0.0,
            max_tokens=150,
            stream=False,
        )

        rewritten = (
            response.choices[0].message.content or ""
        ).strip()

        # Avoid replacing the original question
        # with an empty model response.
        return rewritten or question

    except Exception as error:

        # A rewriting failure should not crash
        # the entire chatbot.
        print(f"[Memory] Rewrite failed: {error}")

        return question
