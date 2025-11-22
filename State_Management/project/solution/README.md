# Udacity Agentic AI Engineer with LangChain and LangGraph
## Project 1: Report-Building Agent

### Schema
`AnswerResponse` and `UserIntent` were implemented using Pydantic model according to
the instruction.  The `AnswerResponse` ensure the react agent respond with a consistant
results, the `UserIntent` ensure the selection is limited to agents available in this
project.

### Agent State
The state memory is collected in the `AgentState` TypedDict class instead of the
Pydantic class, this allow faster runtime. Although, the project was working at the
end but the mixing of using TypedDict for state and Pydantic for schema requires 
additional attention due to the syntax difference, and a few mistakes were made
along the way.  (eg. Accessing a TypedDict can use with the .get() dictionary
function, but for Pydantic, UserIntent, it requires a .dict() to convert it to a
dictionary object).

For the implementation of `classify_intent`, the above mentioned issue resulting
a bit of extra effort for debugging due to the fact that the intent was returned
as a string initially instead of the `UserIntent` object.  It was working after
replacing it with the llm structured output.

The agents, config, and `update_memory` are properly implemented without any issues.
However, one thing to note is that the `should_continue` and `create_workflow` was
mistakenly indented in the starter template, and it was fixed.

### Prompt Template and Calculation Tool
The calculation system prompt was designed using the other provided system prompts
as template to ensure readability and consistency.
For the `calculator` tool, I used AI tool to come up with the regular expression
for the safety criteria of the mathematic expression by limiting it to do basic
arithmetic with the following operators: +,-,*,/,(,) and allowing only integer and
float as input.
Since the `calculator` tool is the output of `create_calculator_tool` and initially
a return statement was missing, and it took some time to find out the problem.

### Run-time Results
Please find it in the `output.txt` document.