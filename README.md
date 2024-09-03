# SoloLGraph

SololGraph is a lightweight, single-thread-agent state graph library inspired by LangGraph. It focuses on simplicity and ease of use for scenarios where complex multi-thread systems are not required.

## Features

- Simple state management
- Easy-to-use graph construction
- Single thread focus
- Streamlined workflow creation
- Ease integration with language models (like GPT)
- Visualization of state graphs

## Installation

You can install SololGraph using pip:

```bash
pip install sololgraph
```

# Quick Start
Here's a simple example to get you started:

```python
from sololgraph import StateGraph, START, END
from langchain_community.chat_models import ChatPerplexity

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

# Contributing
We welcome contributions! Please see our contributing guidelines for more details.

# License
SololGraph is released under the MIT License. See the LICENSE file for more details.

# Acknowledgements
SololGraph is inspired by LangGraph. We thank the LangGraph team for their innovative work in the field of language model workflows.


