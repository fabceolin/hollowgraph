from hollowgraph import StateGraph, START, END

# Define the graph
graph = StateGraph({"value": int})

# Add nodes
graph.add_node("start", run=lambda state: {"value": state.get("value", 0) + 1})
graph.add_node("end", run=lambda state: {"result": f"Final value: {state['value']}"})

# Set entry and finish points
graph.set_entry_point("start")
graph.set_finish_point("end")

# Add edges
graph.add_edge("start", "end")

# Compile the graph (optional interrupts)
graph.compile(interrupt_before=["start"], interrupt_after=["end"])

# Execute the graph using stream (yields intermediate states and interrupts)
for output in graph.stream({"value": 1}):
    if output["type"] == "state":
        print(f"Intermediate state at node {output['node']}: {output['state']}")
    elif output["type"].startswith("interrupt"):
        print(f"Interrupt at node {output['node']}: {output['state']}")
    elif output["type"] == "final":
        print(f"Final state: {output['state']}")

