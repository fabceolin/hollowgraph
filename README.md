# Hollowgraph

Hollowgraph is a lightweight, single-app state graph library inspired by LangGraph. It focuses on simplicity for use with local standalone AI agents, providing an easy-to-use framework for building state-driven LLM workflows without unnecessary features for running single apps.

## Features

- Simple state management
- Easy-to-use graph construction
- Single app focus
- Streamlined workflow creation
- Ease integration with any language models (like GPT)
- LLM library agnostic
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

# Add nodes
graph.add_node("start", lambda state: {"value": state["value"] + 5})
graph.add_node("process", lambda state: {"value": state["value"] * 2})
graph.add_node("end", lambda state: {"result": f"Final value: {state['value']}"})

# Add edges
graph.set_entry_point("start")
graph.add_conditional_edges(
    "start",
    lambda state: state["value"] > 10,
    {True: "end", False: "process"}
)
graph.add_edge("process", "start")
graph.set_finish_point("end")

# Compile and run the graph
compiled_graph = graph.compile()
result = compiled_graph.invoke({"value": 1})

print(result)
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

A full example with LLM capabilities can be found in the examples directory.

# Contributing
We welcome contributions! Please see our contributing guidelines for more details.

# License
Hollowgraph is released under the MIT License. See the LICENSE file for more details.

# Acknowledgements
Hollowgraph is inspired by LangGraph. We thank the LangGraph team for their innovative work in the field of language model workflows.


