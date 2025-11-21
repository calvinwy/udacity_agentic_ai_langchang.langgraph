# State Management Exercise: Library Assistant (Updated Starter)

In this updated exercise, you'll build a library assistant that responds to borrowing, returning, and overdue book queries. The state is treated as immutable—nodes should return dictionaries of updates rather than modifying the state directly. A proper START edge has been added to the workflow.

## Part 1: Conceptual Questions

1. **Difference between a plain `TypedDict` and `MessagesState`**

2. **Why treat the state as immutable and return new values?**

3. **Difference between `state` and `config`**

4. **Why specify a `thread_id` with a checkpointer**

## Part 2: Implement the Library Assistant

Follow the TODOs in the code cell below to complete the library assistant. Be sure to:

- Add `messages` field to `LibraryState` that stores a `List[AnyMessage]` with an `add_messages` reducer.
- Add a `books_borrowed` field to `LibraryState` that stores a `List[str]` without a reducer to allow the ability to remove or add values.
- Add a `last_user_message` field to `LibraryState` to store the last human message received.
- Implement the handlers to return dictionaries containing `messages`, `books_borrowed` when appropriate, and `resolved`.
- Add an edge from `START` to the `router` node in the workflow.
- Connect each handler node to the `END` node.
- Compile the graph with an `InMemorySaver` to persist state across multiple invocations.
- Run the application to test your solution.

> NOTE:
>
> In your handler code, **DO NOT** mutate or modify the passed in `state`. Instead update the global state by returning a new dictionary with only the fields you want to update. LangGraph will merge these updates into the existing state.
```