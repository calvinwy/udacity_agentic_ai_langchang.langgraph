Here's the extracted markdown text from your image:


# Solution — Library Assistant State Management Exercise

This document provides answers to the conceptual questions and outlines one possible implementation of the library assistant. It highlights how to use `MessagesState` for conversational history, how to treat state as immutable by returning new values instead of mutating, and how to persist state with a checkpointer.

## Answers to Conceptual Questions

1. **Difference between a plain `TypedDict` and `MessagesState`**

   A plain `TypedDict` defines a fixed set of keys and value types but has no special behaviour when multiple updates are merged. `MessagesState` inherits from `TypedDict` and automatically includes a `messages` field annotated with the `add_messages` reducer. This reducer appends new messages to the existing list rather than overwriting it, preserving the conversation history. Inheriting from `MessagesState` is therefore ideal when building conversational state machines.

2. **Why treat the state as immutable and return new values?**

   In LangGraph the state is conceptually immutable: rather than mutating the state in place, nodes should return dictionaries containing new values. When you want to add a book title or append a message, you return a new list containing the existing values plus the new item. When you want to remove a book title, you return a new list that omits it. LangGraph then merges these returned values into the state for the next step.

3. **Difference between `state` and `config`**

   The `state` represents the conversation context and flows through the graph; nodes can read and modify it. The `config` contains per-invocation parameters that control execution but are not stored in the state. For example, the `thread_id` is used by the checkpointer comes from the `config`. Changing the `config` does not mutate the state; instead it determines which state snapshot is loaded or saved.

## Solution Overview

The completed notebook defines a `LibraryState` that inherits from `MessagesState` and adds four fields: `section` (determines which handler to run next), `books_borrowed` (a plain list of book titles), `pending_tasks` (a plain list of tasks), and `resolved` (a simple boolean). The router function inspects the latest human message to determine the intent, returns a dictionary that sets the `section`, appends a new task to the `pending_tasks` list, and resets `resolved` to `False`.

Each handler performs its specific job by returning a dictionary of updated values rather than mutating the state directly:

Because we treat the state as immutable, the returned dictionaries replace the corresponding fields in the next state. The `messages` field still comes from `MessagesState` and is automatically appended to, but `books_borrowed` and `pending_tasks` are replaced wholesale. The `next_step` function checks whether `resolved` has been set to `True`. If so, it returns `END` to stop the workflow; otherwise it returns the current value of `state.section` so the router can send control to the appropriate handler. Each handler node has an edge to `END` so the workflow terminates after it runs.

The solution compiles the workflow with an `InMemorySaver` and demonstrates state persistence by invoking the workflow multiple times with the same `thread_id`. The demonstration shows that after borrowing Moby Dick, the book remains in `books_borrowed` during subsequent interactions. Running the workflow with a new `thread_id` yields a fresh state and an empty borrowed books list, illustrating thread isolation.

## Key Takeaways

• `MessagesState` can be used for conversational state so messages are automatically appended via the built-in reducer.
• Treat other fields as immutable: return new lists when you need to add or remove items instead of mutating the existing state.
• Implement a router node that inspects user messages and determines which handler to run next.
• Persist conversation state across invocations by compiling the workflow with a checkpointer and specifying a `thread_id`.
```