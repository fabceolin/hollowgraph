# Hollowgraph

Hollowgraph is a lightweight, single-app state graph library inspired by LangGraph. It focuses on simplicity for use with local standalone AI agents, providing an easy-to-use framework for building state-driven LLM workflows without unnecessary features for running single apps.

## Features

- Simple state management
- Easy-to-use graph construction
- Single app focus
- Streamlined workflow creation
- Ease integration with any language models (like GPT)
- LLM library agnostic
- Parallel fan out fan in support
- Visualization of state graphs

## Installation

You can install Hollowgraph using pip:

```
pip install git+https://github.com/fabceolin/hollowgraph.git
```

# Quick Start
Here's a simple example to get you started:

```
from hollowgraph import StateGraph, START, END

# Initialize the StateGraph
graph = StateGraph({"value": int, "result": str})

# Add nodes with print statements
def start_node(state):
    new_state = {"value": state["value"] + 5}
    print(f"Start node: {state} -> {new_state}")
    return new_state

def process_node(state):
    new_state = {"value": state["value"] * 2}
    print(f"Process node: {state} -> {new_state}")
    return new_state

def end_node(state):
    new_state = {"result": f"Final value: {state['value']}"}
    print(f"End node: {state} -> {new_state}")
    return new_state

graph.add_node("start", start_node)
graph.add_node("process", process_node)
graph.add_node("end", end_node)

# Add edges
graph.set_entry_point("start")
graph.add_conditional_edges(
    "start",
    lambda state: state["value"] > 10,
    {True: "end", False: "process"}
)
graph.add_edge("process", "start")
graph.set_finish_point("end")

# Compile the graph
compiled_graph = graph.compile()

# Run the graph and print results
print("Starting graph execution:")
results = list(compiled_graph.invoke({"value": 1}))

print("\nFinal result:")
for result in results:
    print(result)
```

Graph navigation
```

  ┌──────────────────────────┐
  │                          ▼
┌─────────┐  value <= 10   ┌─────────────┐
│ process │ ◀───────────── │    start    │
└─────────┘                └─────────────┘
                             │
                             │ value > 10
                             ▼
                           ╔═════════════╗
                           ║     end     ║
                           ╚═════════════╝

```
output:
```
Starting graph execution:
Start node: {'value': 1} -> {'value': 6}
Process node: {'value': 6} -> {'value': 12}
Start node: {'value': 12} -> {'value': 17}
End node: {'value': 17} -> {'result': 'Final value: 17'}

Final result:
{'type': 'final', 'state': {'value': 17, 'result': 'Final value: 17'}}
```

A full example with LLM capabilities and fan out fan in examples can be found in the examples directory.

# Contributing
We welcome contributions! Please see our contributing guidelines for more details.

# License
Hollowgraph is released under the MIT License. See the LICENSE file for more details.

# Acknowledgements
Hollowgraph is inspired by LangGraph. We thank the LangGraph team for their innovative work in the field of language model workflows.


