from hollowgraph import StateGraph

def run_fan_out_fan_in_example():
    """
    This example demonstrates the fan-out and fan-in execution pattern using the StateGraph class.

    Fan-out and fan-in patterns are widely used in parallel processing and workflow orchestration.
    In the fan-out phase, the workflow branches out into multiple parallel paths, where each path
    performs independent processing. After the parallel tasks are completed, the workflow fans
    back in at a designated point, where the results of the parallel tasks are aggregated or synchronized.

    In this example, we will simulate a workflow that branches into three parallel execution flows,
    each performing simple arithmetic operations on an initial 'value'. The parallel flows increment
    the value by different amounts (1, 2, and 3), and the fan-in point collects these results and sums
    them into a final result.

    ## Workflow structure:
       - The graph starts at the 'start' node.
       - From 'start', the graph fans out into three parallel flows:
         - 'flow1_start': Increments the 'value' by 1.
         - 'flow2_start': Increments the 'value' by 2.
         - 'flow3_start': Increments the 'value' by 3.
       - After the parallel flows finish, they fan back in at the 'fan_in' node.
       - The 'fan_in' node aggregates the results from all the parallel flows.
       - Finally, the workflow proceeds to the 'end' node, where the final result is printed.

    ## Node Definitions:
    1. **start_run**: Initializes the 'value' if not already set in the state.
    2. **flow1_start_run**: Increments the 'value' by 1 and returns the result in 'flow1_value'.
    3. **flow2_start_run**: Increments the 'value' by 2 and returns the result in 'flow2_value'.
    4. **flow3_start_run**: Increments the 'value' by 3 and returns the result in 'flow3_value'.
    5. **fan_in_run**: This node collects results from all the parallel flows ('flow1_value', 'flow2_value',
       and 'flow3_value'), sums them up, and stores the aggregated value in 'result'.
    6. **end_run**: The final node that prints the 'result' as the final output.

    ## StateGraph Configuration:
    1. **Nodes**: We define nodes for 'start', three parallel flows ('flow1_start', 'flow2_start', 'flow3_start'),
       a fan-in node ('fan_in'), and an 'end' node.
    2. **Edges**: The graph defines the relationships between the nodes. From 'start', parallel edges lead to
       the three flows, which eventually connect to the 'fan_in' node. After the fan-in node, a normal edge
       connects to the 'end' node.

    ## Fan-Out and Fan-In Logic:
    - The fan-out occurs when the graph transitions from the 'start' node to the three parallel nodes
      ('flow1_start', 'flow2_start', 'flow3_start').
    - Each of these parallel nodes executes in its own thread, processing the 'value' independently and
      returning their results.
    - The fan-in node ('fan_in') is responsible for waiting for all the parallel threads to finish. Once
      they complete, it collects their results from the shared state and computes a final aggregate value.
    - The result is passed to the 'end' node, which prints the final value.

    ## Example:
    - The graph starts with an initial state of {'value': 5}.
    - After fan-out, the three parallel nodes increment the 'value' as follows:
      - 'flow1_start' increments by 1 -> flow1_value = 6
      - 'flow2_start' increments by 2 -> flow2_value = 7
      - 'flow3_start' increments by 3 -> flow3_value = 8
    - The fan-in node collects these values and sums them: 6 + 7 + 8 = 21.
    - Finally, the 'end' node prints the total result: 21.

    ## Expected Output:
    Running this example should produce output like:

    ┌───────────────────┐
    │                   │
    │                   │
    │ parallel     ┌────┼───────────────────────────────────────────────┐
    ▼              │    │                                               ▼
  ┌─────────────┐  │  ┌─────────────┐  parallel   ┌─────────────┐     ┌────────┐     ┌─────┐
  │ flow3_start │ ─┘  │    START    │ ──────────▶ │ flow1_start │ ──▶ │ fan_in │ ──▶ │ END │
  └─────────────┘     └─────────────┘             └─────────────┘     └────────┘     └─────┘
                        │                                               ▲
                        │ parallel                                      │
                        ▼                                               │
                      ┌─────────────┐                                   │
                      │ flow2_start │ ──────────────────────────────────┘
                      └─────────────┘

    """
    # Define node functions
    def start_run(state, config, node, graph):
        # Initialize 'value' if not present
        state.setdefault('value', 0)
        print(f"start_run: Initialized 'value' = {state['value']}")
        return {}

    def flow1_start_run(state, config, node, graph):
        # Increment 'value' by 1
        new_value = state.get('value', 0) + 1
        print(f"flow1_start_run: Incremented 'value' by 1 -> flow1_value = {new_value}")
        return {'flow1_value': new_value}

    def flow2_start_run(state, config, node, graph):
        # Increment 'value' by 2
        new_value = state.get('value', 0) + 2
        print(f"flow2_start_run: Incremented 'value' by 2 -> flow2_value = {new_value}")
        return {'flow2_value': new_value}

    def flow3_start_run(state, config, node, graph):
        # Increment 'value' by 3
        new_value = state.get('value', 0) + 3
        print(f"flow3_start_run: Incremented 'value' by 3 -> flow3_value = {new_value}")
        return {'flow3_value': new_value}

    def fan_in_run(state, config, node, graph):
        # Collect results from all parallel flows
        parallel_results = state.get('parallel_results', [])
        total = sum(result.get('flow1_value', 0) for result in parallel_results) + \
                sum(result.get('flow2_value', 0) for result in parallel_results) + \
                sum(result.get('flow3_value', 0) for result in parallel_results)
        print(f"fan_in_run: Summed parallel results -> total = {total}")
        return {'result': total}

    def end_run(state, config, node, graph):
        # Finalize the result
        final_result = state.get('result', 0)
        print(f"end_run: Final result = {final_result}")
        return {'final_result': final_result}

    # Create the StateGraph instance
    state_graph = StateGraph(state_schema={"value": int})

    # Add nodes
    state_graph.add_node("start", run=start_run)
    state_graph.add_node("flow1_start", run=run_fan_out_fan_in_example)
    state_graph.add_node("flow2_start", run=run_fan_out_fan_in_example)
    state_graph.add_fanin_node("fan_in", run=fan_in_run)
    state_graph.add_node("end", run=end_run)

    # Set entry and finish points
    state_graph.set_entry_point("start")
    state_graph.set_finish_point("end")

    # Add edges
    # From start to flow1, flow2, flow3 (parallel edges)
    state_graph.add_parallel_edge("start", "flow1_start", "fan_in")
    state_graph.add_parallel_edge("start", "flow2_start", "fan_in")

    # From fan_in to end
    state_graph.add_edge("fan_in", "end")

    # Invoke the graph with an initial state
    initial_state = {'value': 5}
    print("Starting the fan-out and fan-in example with initial state:", initial_state)

    # Execute the graph
    execution = state_graph.invoke(initial_state)

    # Iterate through the generator to completion
    final_output = None
    for output in execution:
        if output['type'] == 'final':
            final_output = output

    return final_output

def build_fan_out_fan_in_example():
    # Define node functions
    def start_run(state, config, node, graph):
        # Initialize 'value' if not present
        state.setdefault('value', 0)
        return {}

    def flow1_start_run(state, config, node, graph):
        # Increment 'value' by 1
        new_value = state.get('value', 0) + 1
        return {'flow1_value': new_value}

    def flow2_start_run(state, config, node, graph):
        # Increment 'value' by 2
        new_value = state.get('value', 0) + 2
        return {'flow2_value': new_value}

    def flow3_start_run(state, config, node, graph):
        # Increment 'value' by 3
        new_value = state.get('value', 0) + 3
        return {'flow3_value': new_value}

    def fan_in_run(state, config, node, graph):
        # Collect results from all parallel flows
        parallel_results = state.get('parallel_results', [])
        total = sum(result.get('flow1_value', 0) for result in parallel_results) + \
                sum(result.get('flow2_value', 0) for result in parallel_results) + \
                sum(result.get('flow3_value', 0) for result in parallel_results)
        return {'result': total}

    def end_run(state, config, node, graph):
        # Finalize the result
        final_result = state.get('result', 0)
        return {'final_result': final_result}

    # Create the StateGraph instance
    state_graph = StateGraph(state_schema={"value": int})

    # Add nodes
    state_graph.add_node("start", run=start_run)
    state_graph.add_node("flow1_start", run=flow1_start_run)
    state_graph.add_node("flow2_start", run=flow2_start_run)
    state_graph.add_node("flow3_start", run=flow3_start_run)
    state_graph.add_fanin_node("fan_in", run=fan_in_run)
    state_graph.add_node("end", run=end_run)

    # Set entry and finish points
    state_graph.set_entry_point("start")
    state_graph.set_finish_point("end")

    # Add edges
    # From start to flow1, flow2, flow3 (parallel edges)
    state_graph.add_parallel_edge("start", "flow1_start", "fan_in")
    state_graph.add_parallel_edge("start", "flow2_start", "fan_in")
    state_graph.add_parallel_edge("start", "flow3_start", "fan_in")

    # From fan_in to end
    state_graph.add_edge("fan_in", "end")

    # Return the state graph
    return state_graph


def run_two_flows_nested_fan_out_fan_in_example():
    """
    Demonstrates a nested fan-out and fan-in pattern using the StateGraph class with parallel execution.

    The concept of "fan-out" and "fan-in" is commonly used in distributed systems and parallel processing.
    In a fan-out scenario, a single process branches out into multiple parallel tasks, where each task
    operates independently on a portion of the input data. After the parallel tasks complete, the results
    are merged back into a single process at the "fan-in" stage.

    This example builds upon this idea by nesting two fan-out and fan-in graphs. The main workflow contains
    two parallel flows (`flow1_main` and `flow2_main`), and each of these flows is itself a fan-out and fan-in
    graph. The results of these two subgraphs are then aggregated in the main fan-in node before completing the
    workflow.

    ## Workflow structure:
       - The main graph starts at the 'start' node.
       - From 'start', the main graph fans out into two parallel flows:
         - 'flow1_main': Executes a subgraph that performs arithmetic operations on an initial 'value' (based on 'x').
         - 'flow2_main': Executes another subgraph that also performs arithmetic operations on a separate 'value' (based on 'y').
       - Both 'flow1_main' and 'flow2_main' are independent parallel flows, each running a fan-out/fan-in graph.
       - After both flows finish, they fan back in at the 'fan_in_main' node.
       - The 'fan_in_main' node aggregates the results of the two subgraphs, summing the results from both flows.
       - Finally, the workflow proceeds to the 'end' node, where the aggregated result is printed.

    ## Node Definitions:
    1. **start_run**: Initializes 'x' and 'y' in the state (representing two separate initial values for the parallel flows).
    2. **flow1_main_run**: This node executes the first subgraph (similar to the `run_fan_out_fan_in_example` function)
       with 'x' as the input. The subgraph itself fans out into three parallel flows, increments the value in different ways,
       and then aggregates the results.
    3. **flow2_main_run**: This node executes the second subgraph, identical to `flow1_main_run` but with 'y' as the input.
    4. **fan_in_main_run**: The fan-in node that collects the results from both 'flow1_main' and 'flow2_main' and sums them.
    5. **end_run**: The final node that prints the aggregated result from both parallel flows.

    ## Subgraph Structure (for both `flow1_main` and `flow2_main`):
    - Each subgraph is a fan-out and fan-in pattern that works as follows:
       - Starts with the 'start' node.
       - Fans out into three parallel flows:
         - 'flow1_start': Increments the 'value' by 1.
         - 'flow2_start': Increments the 'value' by 2.
         - 'flow3_start': Increments the 'value' by 3.
       - All three flows run in parallel, performing independent calculations.
       - After the parallel flows finish, they fan back in at the 'fan_in' node, which aggregates the results by summing
         the values from each flow.
       - The final result is passed to the 'end' node of the subgraph.

    ## Main Fan-Out and Fan-In Logic:
    - The main fan-out occurs when the graph transitions from the 'start' node to the two parallel nodes
      ('flow1_main' and 'flow2_main').
    - Each of these parallel nodes executes its own subgraph, running three parallel flows inside.
    - The main fan-in occurs at the 'fan_in_main' node, where the results from both subgraphs are aggregated into a single sum.

    ## Example:
    - The main graph starts with an initial state of {'x': 3, 'y': 4}.
    - **Subgraph for `flow1_main`**:
      - The input state has 'value' = 3 (from 'x').
      - After fan-out, the three parallel nodes increment the 'value' as follows:
        - 'flow1_start' increments by 1 -> flow1_value = 4
        - 'flow2_start' increments by 2 -> flow2_value = 5
        - 'flow3_start' increments by 3 -> flow3_value = 6
      - The fan-in node collects these values and sums them: 4 + 5 + 6 = 15.
      - The subgraph result for `flow1_main` is 15.
    - **Subgraph for `flow2_main`**:
      - The input state has 'value' = 4 (from 'y').
      - After fan-out, the three parallel nodes increment the 'value' as follows:
        - 'flow1_start' increments by 1 -> flow1_value = 5
        - 'flow2_start' increments by 2 -> flow2_value = 6
        - 'flow3_start' increments by 3 -> flow3_value = 7
      - The fan-in node collects these values and sums them: 5 + 6 + 7 = 18.
      - The subgraph result for `flow2_main` is 18.
    - **Main Fan-In**:
      - The results from the two subgraphs are collected at the 'fan_in_main' node.
      - Aggregation: 15 (from `flow1_main`) + 18 (from `flow2_main`) = 33.
    - **Final Output**:
      - The 'end' node prints the aggregated result: 33.

    ## Expected Output:
    Running this example should produce output like:

    ```
    start_run: Initialized 'x' = 3, 'y' = 4
    fan_in_main_run: Aggregated subgraph results -> total = 33
    end_run: Final aggregated result = 33

                                          ┌−−−−−−−−−−−−−−−−−−−−−−−−−−−−−┐
                                          ╎     Subgraph flow1_main     ╎
                                          ╎                             ╎
                                          ╎ ┌─────────────────────────┐ ╎
                                          ╎ │       flow1_start       │ ╎
                                          ╎ │ interrupt_before: False │ ╎
                                          ╎ │ interrupt_after: False  │ ╎
                                          ╎ └─────────────────────────┘ ╎
                                          ╎   │                         ╎
                                          ╎   │ parallel                ╎
                                          ╎   │                         ╎
┌−−−−−−−−−−−−−−−−−−−−−−−−−−−−−−−−−−−−−−−−−    │                          −−−−−−−−−−−−−−−−−−−−−−−−−−−−−−−−−−−−−−−−−┐
╎                                             ▼                                                                   ╎
╎ ┌─────────────────────────┐               ┌─────────────────────────┐               ┌─────────────────────────┐ ╎
╎ │       flow3_start       │               │         fan_in          │               │       flow2_start       │ ╎
╎ │ interrupt_before: False │  parallel     │ interrupt_before: False │   parallel    │ interrupt_before: False │ ╎
╎ │ interrupt_after: False  │ ────────────▶ │ interrupt_after: False  │ ◀──────────── │ interrupt_after: False  │ ╎
╎ └─────────────────────────┘               └─────────────────────────┘               └─────────────────────────┘ ╎
╎                                             │                                                                   ╎
└−−−−−−−−−−−−−−−−−−−−−−−−−−−−−−−−−−−−−−−−−    │                          −−−−−−−−−−−−−−−−−−−−−−−−−−−−−−−−−−−−−−−−−┘
                                          ╎   │                         ╎
                                          ╎   │                         ╎
                                          ╎   ▼                         ╎
                                          ╎ ┌─────────────────────────┐ ╎
                                          ╎ │           end           │ ╎
                                          ╎ │ interrupt_before: False │ ╎
                                          ╎ │ interrupt_after: False  │ ╎
                                          ╎ └─────────────────────────┘ ╎
                                          ╎                             ╎
                                          └−−−−−−−−−−−−−−−−−−−−−−−−−−−−−┘
  ┌─────────────────────────┐               ┌─────────────────────────┐
  │       flow2_main        │               │          start          │
  │ interrupt_before: False │  parallel     │ interrupt_before: False │
  │ interrupt_after: False  │ ◀──────────   │ interrupt_after: False  │
  └─────────────────────────┘               └─────────────────────────┘
    │                                         │
    │                                         │ parallel
    │                                         ▼
    │                                       ┌─────────────────────────┐
    │                                       │       flow1_main        │
    │                                       │ interrupt_before: False │
    │                                       │ interrupt_after: False  │
    │                                       └─────────────────────────┘
    │                                         │
    │                                         │
    │                                         ▼
    │                                       ┌─────────────────────────┐
    │                                       │       fan_in_main       │
    │                                       │ interrupt_before: False │
    └───────────────────────────────────▶   │ interrupt_after: False  │
                                            └─────────────────────────┘
                                              │
                                              │
                                              ▼
                                            ┌─────────────────────────┐
                                            │           end           │
                                            │ interrupt_before: False │
                                            │ interrupt_after: False  │
                                            └─────────────────────────┘
                                          ┌−−−−−−−−−−−−−−−−−−−−−−−−−−−−−┐
                                          ╎     Subgraph flow2_main     ╎
                                          ╎                             ╎
                                          ╎ ┌─────────────────────────┐ ╎
                                          ╎ │       flow1_start       │ ╎
                                          ╎ │ interrupt_before: False │ ╎
                                          ╎ │ interrupt_after: False  │ ╎
                                          ╎ └─────────────────────────┘ ╎
                                          ╎   │                         ╎
                                          ╎   │ parallel                ╎
                                          ╎   │                         ╎
┌−−−−−−−−−−−−−−−−−−−−−−−−−−−−−−−−−−−−−−−−−    │                          −−−−−−−−−−−−−−−−−−−−−−−−−−−−−−−−−−−−−−−−−┐
╎                                             ▼                                                                   ╎
╎ ┌─────────────────────────┐               ┌─────────────────────────┐               ┌─────────────────────────┐ ╎
╎ │       flow3_start       │               │         fan_in          │               │       flow2_start       │ ╎
╎ │ interrupt_before: False │  parallel     │ interrupt_before: False │   parallel    │ interrupt_before: False │ ╎
╎ │ interrupt_after: False  │ ────────────▶ │ interrupt_after: False  │ ◀──────────── │ interrupt_after: False  │ ╎
╎ └─────────────────────────┘               └─────────────────────────┘               └─────────────────────────┘ ╎
╎                                             │                                                                   ╎
└−−−−−−−−−−−−−−−−−−−−−−−−−−−−−−−−−−−−−−−−−    │                          −−−−−−−−−−−−−−−−−−−−−−−−−−−−−−−−−−−−−−−−−┘
                                          ╎   │                         ╎
                                          ╎   │                         ╎
                                          ╎   ▼                         ╎
                                          ╎ ┌─────────────────────────┐ ╎
                                          ╎ │           end           │ ╎
                                          ╎ │ interrupt_before: False │ ╎
                                          ╎ │ interrupt_after: False  │ ╎
                                          ╎ └─────────────────────────┘ ╎
                                          ╎                             ╎
                                          └−−−−−−−−−−−−−−−−−−−−−−−−−−−−−┘
    """
    # Define node functions
    def start_run(state, config, node, graph):
        # Initialize 'x' and 'y' in the state
        state.setdefault('x', 3)
        state.setdefault('y', 4)
        print(f"start_run: Initialized 'x' = {state['x']}, 'y' = {state['y']}")
        return {}

    def flow1_main_run(state, config, node, graph):
        # Build and invoke the subgraph with 'x'
        subgraph = build_fan_out_fan_in_example()
        initial_state = {'value': state.get('x', 0)}
        execution = subgraph.invoke(initial_state)
        final_output = next((output for output in execution if output['type'] == 'final'), None)
        if final_output:
            subgraph_result = final_output['state'].get('final_result', 0)
            return {'subgraph_result': subgraph_result}
        else:
            return {}

    def flow2_main_run(state, config, node, graph):
        # Build and invoke the subgraph with 'y'
        subgraph = build_fan_out_fan_in_example()
        initial_state = {'value': state.get('y', 0)}
        execution = subgraph.invoke(initial_state)
        final_output = next((output for output in execution if output['type'] == 'final'), None)
        if final_output:
            subgraph_result = final_output['state'].get('final_result', 0)
            return {'subgraph_result': subgraph_result}
        else:
            return {}

    def fan_in_main_run(state, config, node, graph):
        # Aggregate results from both subgraphs
        parallel_results = state.get('parallel_results', [])
        total = sum(result.get('subgraph_result', 0) for result in parallel_results)
        print(f"fan_in_main_run: Aggregated subgraph results -> total = {total}")
        return {'result': total}

    def end_run(state, config, node, graph):
        # Finalize the result
        final_result = state.get('result', 0)
        print(f"end_run: Final aggregated result = {final_result}")
        return {'final_result': final_result}

    # Create the StateGraph instance
    state_graph = StateGraph(state_schema={"x": int, "y": int})

    # Add nodes
    state_graph.add_node("start", run=start_run)
    state_graph.add_node("flow1_main", run=flow1_main_run)
    state_graph.add_node("flow2_main", run=flow2_main_run)
    state_graph.add_fanin_node("fan_in_main", run=fan_in_main_run)
    state_graph.add_node("end", run=end_run)

    # Set entry and finish points
    state_graph.set_entry_point("start")
    state_graph.set_finish_point("end")

    # Add edges
    state_graph.add_parallel_edge("start", "flow1_main", "fan_in_main")
    state_graph.add_parallel_edge("start", "flow2_main", "fan_in_main")

    # From fan_in_main to end
    state_graph.add_edge("fan_in_main", "end")

    # Invoke the graph with an initial state
    initial_state = {'x': 3, 'y': 4}
    print("Starting the two-flows nested fan-out and fan-in example with initial state:", initial_state)

    # Execute the graph
    execution = state_graph.invoke(initial_state)

    # Collect the final output
    final_output = next((output for output in execution if output['type'] == 'final'), None)

    # Print the final result
    if final_output:
        print(f"Final output: {final_output['state']['final_result']}")
    else:
        print("No final output was produced.")

# Run the example
run_two_flows_nested_fan_out_fan_in_example()



