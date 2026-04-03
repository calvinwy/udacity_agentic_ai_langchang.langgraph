# Architecture-first scaffolder (structure only)
# - No business logic
# - Produces a compiled graph + mermaid PNG for visualization
#
# Architectures supported:
#   - "pipeline"
#   - "network"
#   - "supervisor"
#   - "hierarchical"

from typing import Callable, Dict, List, Literal, get_args
from IPython.display import Image, display

from langgraph.graph import START, END, StateGraph
from langgraph.graph.message import MessagesState
from langgraph.types import Command

# ---- helpers to create stub nodes -------------------------------------------------

def _noop(*args, **kwargs):
    # does nothing; useful for fixed-edge nodes
    pass

def _dynamic_stub(name: str, choices: List[str]) -> Callable[[MessagesState], Command]:
    """
    Create a node function that returns a Command[Literal[...]] dynamically,
    so LangGraph draws dotted edges to the possible 'goto' targets.

    NOTE: We set the function's __annotations__ at runtime so that LangGraph
    can introspect possible transitions and render them.
    """
    # Build a function that *returns* a Command to one of the choices (we won't execute it here)
    def _fn(state: MessagesState) -> Command:
        # structure-only: return is irrelevant for rendering; we set the type below
        return Command(goto=choices[0] if choices else END)

    # Dynamically set the return annotation to Command[Literal[choices...]]
    # This is what makes LangGraph draw the dotted edges.
    from typing import Literal as _Literal, Annotated as _Annotated  # localize for clarity
    literal = _Literal[tuple(choices + ([END] if END not in choices else []))]  # type: ignore
    _fn.__name__ = name
    _fn.__annotations__ = {"state": MessagesState, "return": Command[literal]}  # type: ignore
    return _fn

# ---- the class --------------------------------------------------------------------

class AgentArchitectureDesigner:
    def __init__(self):
        # You can customize default labels, etc., here
        pass

    def build(self, kind: str, n_agents: int):
        kind = kind.lower()
        if n_agents < 1:
            raise ValueError("n_agents must be >= 1")

        if kind == "pipeline":
            return self._build_pipeline(n_agents)
        elif kind == "network":
            return self._build_network(n_agents)
        elif kind == "supervisor":
            return self._build_supervisor(n_agents)
        elif kind == "hierarchical":
            # simple balanced binary-style hierarchy just for the diagram
            return self._build_hierarchical(n_agents)
        else:
            raise ValueError(f"Unknown architecture kind: {kind}")

    # ---- architectures ----

    def _build_pipeline(self, n: int):
        """
        Start → l1 → l2 → ... → lN → End
        Uses fixed edges; nodes are simple no-ops.
        """
        workflow = StateGraph(MessagesState)

        names = [f"l{i}_agent" for i in range(1, n + 1)]
        for name in names:
            def make_noop(nn):
                def _f(state: MessagesState):
                    # structure-only
                    pass
                _f.__name__ = nn
                return _f
            workflow.add_node(make_noop(name))

        workflow.add_edge(START, names[0])
        for a, b in zip(names, names[1:]):
            workflow.add_edge(a, b)
        workflow.add_edge(names[-1], END)

        return workflow.compile()

    def _build_network(self, n: int):
        """
        Peer-to-peer network where any node may hand off to any other (or END).
        Uses Command[Literal[...]] to show dotted possible routes.
        """
        workflow = StateGraph(MessagesState)
        names = [f"l{i}_agent" for i in range(1, n + 1)]

        # Each node can goto any other node or END (self loops optional; here we include them)
        for name in names:
            choices = [nm for nm in names] + [END]
            workflow.add_node(_dynamic_stub(name, choices))

        # Use a single explicit entry edge
        workflow.add_edge(START, names[0])

        return workflow.compile()

    def _build_supervisor(self, n: int):
        """
        One central supervisor that can route to any worker; workers return to supervisor or END.
        Dotted edges are inferred from Command[Literal[...]].
        """
        if n < 2:
            raise ValueError("Supervisor needs at least 2 nodes (1 supervisor + >=1 worker).")

        workflow = StateGraph(MessagesState)
        supervisor = "supervisor"
        workers = [f"l{i}_agent" for i in range(1, n)]  # n-1 workers + 1 supervisor = n total

        # Supervisor can send to any worker or END
        workflow.add_node(_dynamic_stub(supervisor, workers + [END]))

        # Each worker can go back to supervisor or END
        for w in workers:
            workflow.add_node(_dynamic_stub(w, [supervisor, END]))

        workflow.add_edge(START, supervisor)
        return workflow.compile()

    def _build_hierarchical(self, n: int):
        """
        Simple “tree-ish” hierarchy:
            root → level-1 children → level-2 children ...
        Each non-leaf can go to its children or END; leaves can go up to parent or END.
        """
        workflow = StateGraph(MessagesState)

        # Create names and a parent map (a simple linear parent assignment for clarity)
        names = [f"l{i}_agent" for i in range(1, n + 1)]
        root = names[0]
        parents: Dict[str, str] = {}
        for i in range(1, len(names)):
            parents[names[i]] = names[(i - 1) // 2]  # crude parent: binary-ish tree

        # Children lookup
        children: Dict[str, List[str]] = {nm: [] for nm in names}
        for child, parent in parents.items():
            children[parent].append(child)

        # Build nodes: non-leaf → children or END; leaf → parent or END
        for nm in names:
            if children[nm]:  # non-leaf
                workflow.add_node(_dynamic_stub(nm, children[nm] + [END]))
            else:  # leaf
                parent = parents.get(nm)
                workflow.add_node(_dynamic_stub(nm, ([parent] if parent else []) + [END]))

        workflow.add_edge(START, root)
        return workflow.compile()

    # ---- render helper ----

    def render(self, graph):
        display(Image(graph.get_graph().draw_mermaid_png()))
        return graph
