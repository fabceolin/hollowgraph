from sololgraph import StateGraph, START, END

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

