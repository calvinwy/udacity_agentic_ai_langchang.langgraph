# Memory Exercise: Dining Assistant (Starter)

In this exercise, you'll build a dining assistant that remembers each user's dietary preferences and personalizes its responses across multiple conversation sessions.

## Part 1: Conceptual Questions

1. **The role of checkpointers in persisting state**

2. **The difference between state and config**

3. **How thread isolation works in LangGraph**

## Part 2: Implement the Dining Assistant

Follow the TODOs in the code cell below to complete the dining assistant. Be sure to:

- Import `InMemorySaver` from `langgraph.checkpoint.memory` and compile the workflow with it.
- Expand `MemoryState` to include a `user_memory` dictionary to store dietary preferences and visit counts.
- Write a node function `remember_preferences(state: MemoryState) -> MemoryState` that:
  - Detects dietary preferences (vegan, vegetarian, gluten-free) from the latest user message
  - Stores them in `state["user_memory"]["diet"]` (a list)
  - Increments a visit counter in `state["user_memory"]["visits"]`
  - Appends a personalized dish suggestion and a welcome-back message on return visits
- Add this new node to the workflow after the greet node and update the edges accordingly.
- Compile the workflow with an `InMemorySaver` to enable persistence.
- Test your implementation by invoking the workflow multiple times with the same `thread_id`.