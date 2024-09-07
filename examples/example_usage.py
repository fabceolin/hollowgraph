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

# Optionally, you can also visualize the graph
compiled_graph.save_graph_image("example_graph.png")
print("\nGraph visualization saved as 'example_graph.png'")
