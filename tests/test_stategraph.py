import unittest
import pygraphviz as pgv


from unittest.mock import patch
from parameterized import parameterized
from hypothesis import given, strategies as st, settings
from sololgraph import StateGraph, START, END

class TestStateGraph(unittest.TestCase):

    def setUp(self):
        self.graph = StateGraph({"test": "schema"})

    def test_init(self):
        """
        Verify that the StateGraph is initialized correctly with the given state schema.
        """
        self.assertEqual(self.graph.state_schema, {"test": "schema"})

    @parameterized.expand([
        ("simple_node", "test_node", None),
        ("node_with_function", "func_node", lambda: None),
    ])
    def test_add_node(self, name, node, run):
        """
        Test adding nodes to the graph, both with and without associated functions.
        Verify that nodes are correctly added and their run functions are properly set.
        """
        self.graph.add_node(node, run)
        node_data = self.graph.node(node)
        if run:
            self.assertEqual(node_data["run"], run)

    def test_add_node_duplicate(self):
        """
        Ensure that adding a duplicate node raises a ValueError.
        """
        self.graph.add_node("test_node")
        with self.assertRaises(ValueError):
            self.graph.add_node("test_node")

    def test_add_edge(self):
        """
        Verify that edges can be added between existing nodes in the graph.
        """
        self.graph.add_node("node1")
        self.graph.add_node("node2")
        self.graph.add_edge("node1", "node2")
        self.assertIn("node2", self.graph.successors("node1"))

    def test_add_edge_nonexistent_node(self):
        """
        Check that attempting to add an edge between non-existent nodes raises a ValueError.
        """
        with self.assertRaises(ValueError):
            self.graph.add_edge("nonexistent1", "nonexistent2")

    def test_add_conditional_edges(self):
        """
        Test the addition of conditional edges based on a given function and condition map.
        Verify that the edges are correctly added to the graph.
        """
        self.graph.add_node("node1")
        self.graph.add_node("node2")
        self.graph.add_node("node3")

        def condition_func(state):
            return state["value"]

        self.graph.add_conditional_edges("node1", condition_func, {True: "node2", False: "node3"})

        self.assertIn("node2", self.graph.successors("node1"))
        self.assertIn("node3", self.graph.successors("node1"))

    def test_add_conditional_edges_nonexistent_node(self):
        """
        Ensure that adding conditional edges from a non-existent node raises a ValueError.
        """
        with self.assertRaises(ValueError):
            self.graph.add_conditional_edges("nonexistent", lambda: True, {True: "node2"})

    def test_set_entry_point(self):
        """
        Verify that the entry point of the graph can be set correctly.
        """
        self.graph.add_node("start_node")
        self.graph.set_entry_point("start_node")
        self.assertIn("start_node", self.graph.successors(START))

    def test_set_entry_point_nonexistent_node(self):
        """
        Check that attempting to set a non-existent node as the entry point raises a ValueError.
        """
        with self.assertRaises(ValueError):
            self.graph.set_entry_point("nonexistent")

    def test_set_finish_point(self):
        """
        Ensure that the finish point of the graph can be set correctly.
        """
        self.graph.add_node("end_node")
        self.graph.set_finish_point("end_node")
        self.assertIn(END, self.graph.successors("end_node"))

    def test_set_finish_point_nonexistent_node(self):
        """
        Verify that setting a non-existent node as the finish point raises a ValueError.
        """
        with self.assertRaises(ValueError):
            self.graph.set_finish_point("nonexistent")

    def test_compile(self):
        """
        Test the compilation of the graph, including setting interrupt points.
        Ensure that interrupt points are correctly set in the compiled graph.
        """
        self.graph.add_node("node1")
        self.graph.add_node("node2")
        compiled_graph = self.graph.compile(interrupt_before=["node1"], interrupt_after=["node2"])
        self.assertEqual(compiled_graph.interrupt_before, ["node1"])
        self.assertEqual(compiled_graph.interrupt_after, ["node2"])

    def test_compile_nonexistent_node(self):
        """
        Check that compiling with non-existent interrupt nodes raises a ValueError.
        """
        with self.assertRaises(ValueError):
            self.graph.compile(interrupt_before=["nonexistent"])

    def test_node(self):
        """
        Verify that node data can be retrieved correctly for existing nodes.
        """
        self.graph.add_node("test_node", run=lambda: None)
        node_data = self.graph.node("test_node")
        self.assertIsNotNone(node_data["run"])

    def test_node_nonexistent(self):
        """
        Ensure that attempting to retrieve data for a non-existent node raises a KeyError.
        """
        with self.assertRaises(KeyError):
            self.graph.node("nonexistent_node")

    def test_edge(self):
        """
        Test that edge data can be retrieved correctly for existing edges.
        """
        self.graph.add_node("node1")
        self.graph.add_node("node2")
        self.graph.add_edge("node1", "node2")
        edge_data = self.graph.edge("node1", "node2")
        self.assertIn("cond", edge_data)

    def test_edge_nonexistent(self):
        """
        Verify that attempting to retrieve data for a non-existent edge raises a KeyError.
        """
        with self.assertRaises(KeyError):
            self.graph.edge("nonexistent1", "nonexistent2")

    def test_successors(self):
        """
        Check that the successors of a node can be correctly retrieved.
        """
        self.graph.add_node("node1")
        self.graph.add_node("node2")
        self.graph.add_node("node3")
        self.graph.add_edge("node1", "node2")
        self.graph.add_edge("node1", "node3")
        successors = self.graph.successors("node1")
        self.assertIn("node2", successors)
        self.assertIn("node3", successors)

    def test_successors_nonexistent(self):
        """
        Ensure that attempting to get successors of a non-existent node raises a KeyError.
        """
        with self.assertRaises(KeyError):
            self.graph.successors("nonexistent_node")

    def test_stream(self):
        """
        Test the stream method of StateGraph, verifying that it yields the correct intermediate states.
        """
        self.graph.add_node("node1", run=lambda state: {"value": state["value"] + 1})
        self.graph.add_node("node2", run=lambda state: {"value": state["value"] * 2})
        self.graph.set_entry_point("node1")
        self.graph.add_edge("node1", "node2")
        self.graph.set_finish_point("node2")

        stream = list(self.graph.stream({"value": 1}))
        self.assertEqual(len(stream), 1)
        self.assertEqual(stream[0]["value"], 4)

    def test_invoke_simple(self):
        """
        Verify the basic functionality of the invoke method for a simple graph.
        """
        self.graph.add_node("node1", run=lambda state: {"value": state["value"] + 1})
        self.graph.add_node("node2", run=lambda state: {"value": state["value"] * 2})
        self.graph.set_entry_point("node1")
        self.graph.add_edge("node1", "node2")
        self.graph.set_finish_point("node2")

        result = list(self.graph.invoke({"value": 1}))
        self.assertEqual(len(result), 1)
        self.assertEqual(result[0]["type"], "final")
        self.assertEqual(result[0]["state"]["value"], 4)

    def test_invoke_with_interrupts(self):
        """
        Test the invoke method with interrupts, ensuring that interrupts are correctly handled and yielded.
        """
        self.graph.add_node("node1", run=lambda state: {"value": state["value"] + 1})
        self.graph.add_node("node2", run=lambda state: {"value": state["value"] * 2})
        self.graph.set_entry_point("node1")
        self.graph.add_edge("node1", "node2")
        self.graph.set_finish_point("node2")
        self.graph.compile(interrupt_before=["node2"])

        result = list(self.graph.invoke({"value": 1}))
        self.assertEqual(len(result), 2)
        self.assertEqual(result[0]["type"], "interrupt")
        self.assertEqual(result[0]["node"], "node2")
        self.assertEqual(result[1]["type"], "final")
        self.assertEqual(result[1]["state"]["value"], 4)

    def test_complex_workflow(self):
        """
        This test verifies the execution of a complex workflow with conditional routing.

        Test setup:
        1. Define a condition function that returns True if the state's "value" is greater than 10.
        2. Add three nodes to the graph:
           - "start": Increases the state's "value" by 5
           - "process": Multiplies the state's "value" by 2
           - "end": Formats a result string with the final value
        3. Set "start" as the entry point.
        4. Add conditional edges from "start":
           - If condition is True (value > 10), go to "end"
           - If condition is False (value <= 10), go to "process"
        5. Add an edge from "process" back to "start" (creating a loop).
        6. Set "end" as the finish point.

        Test execution:
        - Invoke the graph with an initial state of {"value": 1}.

        Assertions:
        1. Verify that the invoke method returns only one result (the final state).
        2. Check that the result type is "final".
        3. Ensure the result contains a "state" key.
        4. Confirm that the state contains a "result" key.
        5. Validate that the final result string is "Final value: 17".

        Expected workflow:
        1. Start: 1 -> 6 (add 5)
        2. Process: 6 -> 12 (multiply by 2)
        3. Start: 12 -> 17 (add 5)
        4. End: (17 > 10, so it goes to end)

        This test ensures that the graph correctly handles conditional routing,
        state updates across multiple nodes, and produces the expected final result.
        """


        """
        Test a more complex workflow with conditional routing, verifying that the graph executes correctly.
        """
        def condition_func(state):
            return state["value"] > 10

        self.graph.add_node("start", run=lambda state: {"value": state["value"] + 5})
        self.graph.add_node("process", run=lambda state: {"value": state["value"] * 2})
        self.graph.add_node("end", run=lambda state: {"result": f"Final value: {state['value']}"})

        self.graph.set_entry_point("start")
        self.graph.add_conditional_edges("start", condition_func, {True: "end", False: "process"})
        self.graph.add_edge("process", "start")
        self.graph.set_finish_point("end")

        invoke_result = list(self.graph.invoke({"value": 1}))

        self.assertEqual(len(invoke_result), 1)
        self.assertEqual(invoke_result[0]["type"], "final")
        self.assertIn("state", invoke_result[0])
        self.assertIn("result", invoke_result[0]["state"])
        self.assertEqual(invoke_result[0]["state"]["result"], "Final value: 17")

    @given(st.dictionaries(st.text(), st.integers()))
    @settings(max_examples=100)  # You can adjust this number as needed
    def test_invoke_property(self, input_state):
        """
        Property-based test to ensure that invoke always produces a final state
        regardless of the input state.
        """
        # Create a new graph for each test run
        graph = StateGraph(state_schema={})

        graph.add_node("start", run=lambda state: {"value": sum(state.values())})
        graph.add_node("end", run=lambda state: {"result": state["value"] * 2})
        graph.set_entry_point("start")
        graph.add_edge("start", "end")
        graph.set_finish_point("end")

        result = list(graph.invoke(input_state))
        self.assertEqual(len(result), 1)
        self.assertEqual(result[0]["type"], "final")
        self.assertIn("result", result[0]["state"])

    def test_complex_workflow_with_interrupts(self):
        """
        This test extends the complex workflow test to include interrupt handling.

        Test setup:
        (Same as test_complex_workflow, with the following addition)
        7. Compile the graph with interrupts:
           - Interrupt before executing the "process" node
           - Interrupt after executing the "end" node

        Test execution:
        - Invoke the graph with an initial state of {"value": 1}.

        Assertions:
        1. Verify that the invoke method returns more than one result (due to interrupts).
        2. Check that the last result is of type "final".
        3. Ensure the final state contains a "result" key.
        4. Validate that the final result string is "Final value: 17".
        5. Check for interrupts:
           - Verify that there's at least one interrupt before the "process" node.
           - Confirm that there's exactly one interrupt after the "end" node.

        Expected workflow with interrupts:
        1. Start: 1 -> 6 (add 5)
        2. Interrupt before "process"
        3. Process: 6 -> 12 (multiply by 2)
        4. Start: 12 -> 17 (add 5)
        5. End: (17 > 10, so it goes to end)
        6. Interrupt after "end"

        This test ensures that the graph correctly handles conditional routing,
        state updates across multiple nodes, produces the expected final result,
        and properly yields interrupt states at the specified points in the workflow.
        It verifies the graph's ability to pause execution at designated points,
        which is crucial for workflows that may require external intervention or
        additional processing at specific stages.
        """
        def condition_func(state):
            return state["value"] > 10

        self.graph.add_node("start", run=lambda state: {"value": state["value"] + 5})
        self.graph.add_node("process", run=lambda state: {"value": state["value"] * 2})
        self.graph.add_node("end", run=lambda state: {"result": f"Final value: {state['value']}"})

        self.graph.set_entry_point("start")
        self.graph.add_conditional_edges("start", condition_func, {True: "end", False: "process"})
        self.graph.add_edge("process", "start")
        self.graph.set_finish_point("end")

        self.graph.compile(interrupt_before=["process"], interrupt_after=["end"])

        invoke_result = list(self.graph.invoke({"value": 1}))

        self.assertGreater(len(invoke_result), 1)  # Should have at least one interrupt and one final state
        self.assertEqual(invoke_result[-1]["type"], "final")
        self.assertIn("result", invoke_result[-1]["state"])
        self.assertEqual(invoke_result[-1]["state"]["result"], "Final value: 17")

        # Check for interrupts
        interrupt_before = [r for r in invoke_result if r["type"] == "interrupt" and r["node"] == "process"]
        self.assertGreater(len(interrupt_before), 0)

        interrupt_after = [r for r in invoke_result if r["type"] == "interrupt" and r["node"] == "end"]
        self.assertEqual(len(interrupt_after), 1)

    def test_cyclic_graph(self):
        """
        Verify that cyclic graphs are handled correctly, with the execution terminating appropriately.
        """
        self.graph.add_node("node1", run=lambda state: {"count": state.get("count", 0) + 1})
        self.graph.add_node("node2", run=lambda state: {"count": state["count"] * 2})
        self.graph.set_entry_point("node1")
        self.graph.add_conditional_edges("node1", lambda state: state["count"] >= 3, {True: "node2", False: "node1"})
        self.graph.set_finish_point("node2")

        result = list(self.graph.invoke({"count": 0}))
        self.assertEqual(result[-1]["state"]["count"], 6)

    def test_error_handling_in_node_function(self):
        """
        Test error handling in node functions, both with and without exception raising enabled.
        """
        def error_func(state):
            raise ValueError("Test error")

        # Create a new graph with raise_exceptions=True
        error_graph = StateGraph(state_schema={}, raise_exceptions=True)
        error_graph.add_node("error_node", run=error_func)
        error_graph.set_entry_point("error_node")
        error_graph.set_finish_point("error_node")

        with self.assertRaises(RuntimeError) as context:
            list(error_graph.invoke({}))

        self.assertIn("Test error", str(context.exception))

        # Test the default behavior (not raising exceptions)
        normal_graph = StateGraph(state_schema={})
        normal_graph.add_node("error_node", run=error_func)
        normal_graph.set_entry_point("error_node")
        normal_graph.set_finish_point("error_node")

        results = list(normal_graph.invoke({}))
        self.assertEqual(len(results), 1)
        self.assertEqual(results[0]["type"], "error")
        self.assertIn("Test error", results[0]["error"])

    @patch('sololgraph.StateGraph._get_next_node')
    def test_no_valid_next_node(self, mock_get_next_node):
        """
        Verify that the graph correctly handles the case where no valid next node is found during execution.
        """
        mock_get_next_node.side_effect = RuntimeError("No valid next node")

        self.graph.add_node("start")
        self.graph.set_entry_point("start")
        self.graph.set_finish_point("start")

        with self.assertRaises(RuntimeError):
            list(self.graph.invoke({}))

    def test_complex_conditional_routing(self):
        """
        Test complex conditional routing scenarios, ensuring that the graph correctly follows the specified paths.
        """
        def route_func(state):
            if state["value"] < 0:
                return "negative"
            elif state["value"] == 0:
                return "zero"
            else:
                return "positive"

        self.graph.add_node("start", run=lambda state: state)
        self.graph.add_node("negative", run=lambda state: {"result": "Negative"})
        self.graph.add_node("zero", run=lambda state: {"result": "Zero"})
        self.graph.add_node("positive", run=lambda state: {"result": "Positive"})

        self.graph.set_entry_point("start")
        self.graph.add_conditional_edges("start", route_func, {
            "negative": "negative",
            "zero": "zero",
            "positive": "positive"
        })
        self.graph.set_finish_point("negative")
        self.graph.set_finish_point("zero")
        self.graph.set_finish_point("positive")

        result_neg = list(self.graph.invoke({"value": -1}))
        result_zero = list(self.graph.invoke({"value": 0}))
        result_pos = list(self.graph.invoke({"value": 1}))

        self.assertEqual(result_neg[-1]["state"]["result"], "Negative")
        self.assertEqual(result_zero[-1]["state"]["result"], "Zero")
        self.assertEqual(result_pos[-1]["state"]["result"], "Positive")

    def test_state_persistence(self):
        """
        Verify that state is correctly persisted and updated across multiple node executions.
        """
        def accumulate(state):
            return {"sum": state.get("sum", 0) + state["value"], "value": state["value"]}

        self.graph.add_node("start", run=accumulate)
        self.graph.add_node("check", run=lambda state: state)
        self.graph.set_entry_point("start")
        self.graph.add_conditional_edges("start", lambda state: state["sum"] >= 10, {True: "check",
            False: "start"
        })
        self.graph.set_finish_point("check")

        result = list(self.graph.invoke({"value": 3}))
        self.assertEqual(result[-1]["state"]["sum"], 12)
        self.assertEqual(len(result), 1)  # Only final state
        self.assertEqual(result[0]["type"], "final")

    def test_config_usage(self):
        """
        Test the usage of configuration in node functions, ensuring that config parameters are correctly passed and used.
        """
        def configurable_func(state, config):
            return {"result": state["value"] * config["multiplier"]}

        self.graph.add_node("start", run=configurable_func)
        self.graph.set_entry_point("start")
        self.graph.set_finish_point("start")

        result = list(self.graph.invoke({"value": 5}, config={"multiplier": 3}))
        self.assertEqual(result[-1]["state"]["result"], 15)

    def test_multiple_entry_points(self):
        """
        Verify that graphs with multiple entry points function correctly, with execution starting from any of the entry points.
        """
        self.graph.add_node("entry1", run=lambda state: {"value": state["value"] + 1})
        self.graph.add_node("entry2", run=lambda state: {"value": state["value"] * 2})
        self.graph.add_node("end", run=lambda state: state)

        self.graph.set_entry_point("entry1")
        self.graph.set_entry_point("entry2")
        self.graph.add_edge("entry1", "end")
        self.graph.add_edge("entry2", "end")
        self.graph.set_finish_point("end")

        result1 = list(self.graph.invoke({"value": 5}))
        result2 = list(self.graph.invoke({"value": 5}))

        self.assertIn(result1[-1]["state"]["value"], [6, 10])
        self.assertIn(result2[-1]["state"]["value"], [6, 10])

    def test_dynamic_node_addition_during_execution(self):
        """
        Test the dynamic addition of nodes during graph execution, ensuring that new nodes are correctly integrated into the workflow.
        """
        def start_node(state):
            state['path'] = ['start']
            return state

        def dynamic_add_node(state, graph):
            if 'dynamic1' not in graph.successors('dynamic_adder'):
                graph.add_node('dynamic1', run=lambda state: {**state, 'value': state['value'] * 2, 'path': state['path'] + ['dynamic1']})
                graph.add_edge('dynamic_adder', 'dynamic1')

            if 'dynamic2' not in graph.successors('dynamic1'):
                graph.add_node('dynamic2', run=lambda state: {**state, 'value': state['value'] + 5, 'path': state['path'] + ['dynamic2']})
                graph.add_edge('dynamic1', 'dynamic2')

            graph.set_finish_point('dynamic2')
            return state

        self.graph.add_node('start', run=start_node)
        self.graph.add_node('dynamic_adder', run=dynamic_add_node)
        self.graph.add_edge('start', 'dynamic_adder')
        self.graph.set_entry_point('start')

        result = list(self.graph.invoke({"value": 5, "path": []}))
        final_state = result[-1]['state']

        self.assertEqual(final_state['value'], 15)  # (5 * 2) + 5
        self.assertEqual(final_state['path'], ['start', 'dynamic1', 'dynamic2'])
        self.assertIn('dynamic1', self.graph.successors('dynamic_adder'))
        self.assertIn('dynamic2', self.graph.successors('dynamic1'))

    def test_exception_in_conditional_edge(self):
        """
        Verify that exceptions in conditional edge functions are correctly propagated and handled.
        """
        def faulty_condition(state):
            raise ValueError("Conditional error")

        self.graph.add_node("start", run=lambda state: state)
        self.graph.set_entry_point("start")
        self.graph.add_conditional_edges("start", faulty_condition, {True: END, False: "start"})

        with self.assertRaises(ValueError):
            list(self.graph.invoke({"value": 0}))

    def test_render_graphviz(self):
        """
        Test the render_graphviz method to ensure it correctly creates a PyGraphviz AGraph
        object representing the StateGraph.
        """
        # Create a simple graph
        self.graph = StateGraph({"test": "schema"})
        self.graph.add_node("node1", run=lambda state: state)
        self.graph.add_node("node2", run=lambda state: {"value": state["value"] * 2})
        self.graph.set_entry_point("node1")
        self.graph.add_edge("node1", "node2")
        self.graph.set_finish_point("node2")
        self.graph.compile(interrupt_before=["node1"], interrupt_after=["node2"])

        # Call render_graphviz
        result = self.graph.render_graphviz()

        # Assert that the result is a PyGraphviz AGraph object
        self.assertIsInstance(result, pgv.AGraph)

        # Check the number of nodes (including START and END)
        self.assertEqual(len(result.nodes()), 4)

        # Check the number of edges
        self.assertEqual(len(result.edges()), 3)

        # Check that all nodes are present
        node_names = [node.get_name() for node in result.nodes()]
        self.assertIn(START, node_names)
        self.assertIn("node1", node_names)
        self.assertIn("node2", node_names)
        self.assertIn(END, node_names)

        # Check node attributes
        node1 = result.get_node("node1")
        self.assertIn("interrupt_before: True", node1.attr['label'])
        self.assertIn("interrupt_after: False", node1.attr['label'])

        node2 = result.get_node("node2")
        self.assertIn("interrupt_before: False", node2.attr['label'])
        self.assertIn("interrupt_after: True", node2.attr['label'])

        # Check edge attributes
        edge = result.get_edge("node1", "node2")
        self.assertEqual(edge.attr['label'], "")  # No condition on this edge

        # Check graph attributes
        self.assertEqual(result.graph_attr['rankdir'], "TB")
        self.assertEqual(result.graph_attr['size'], "8,8")

        # Check node and edge default attributes
        self.assertEqual(result.node_attr['shape'], "rectangle")
        self.assertEqual(result.node_attr['style'], "filled")
        self.assertEqual(result.node_attr['fillcolor'], "white")
        self.assertEqual(result.edge_attr['color'], "black")

    def test_save_graph_image(self):
        """
        Test the save_graph_image method to ensure it correctly saves the graph as an image file.
        """
        import os

        # Create a simple graph
        self.graph = StateGraph({"test": "schema"})
        self.graph.add_node("node1")
        self.graph.add_node("node2")
        self.graph.add_edge("node1", "node2")

        # Define a test filename
        test_filename = "test_graph.png"

        # Save the graph image
        self.graph.save_graph_image(test_filename)

        # Check if the file was created
        self.assertTrue(os.path.exists(test_filename))

        # Check if the file is not empty
        self.assertGreater(os.path.getsize(test_filename), 0)

        # Clean up: remove the test file
        os.remove(test_filename)

    def test_interrupt_handling(self):
        """
        Verify the handling of interrupts, including the ability to resume execution after an interrupt.
        """
        def interruptible_func(state):
            if state.get("interrupt", False):
                raise InterruptedError("Function interrupted")
            return {"value": state["value"] + 1}

        self.graph.add_node("start", run=interruptible_func)
        self.graph.add_node("end", run=lambda state: state)
        self.graph.set_entry_point("start")
        self.graph.add_edge("start", "end")
        self.graph.set_finish_point("end")
        self.graph.compile(interrupt_before=["start"])

        def interrupt_handler(interrupted_state):
            interrupted_state["interrupt"] = True
            return interrupted_state

        results = list(self.graph.invoke({"value": 0}))
        self.assertEqual(len(results), 2)  # Interrupt + final state
        self.assertEqual(results[0]["type"], "interrupt")

        # Simulate handling the interrupt
        interrupt_state = interrupt_handler(results[0]["state"])

        # Continue execution with handled state
        final_results = list(self.graph.invoke(interrupt_state))
        self.assertEqual(final_results[-1]["state"]["value"], 1)

    def test_parallel_execution_simulation(self):
        """
        This test simulates a parallel execution scenario within the StateGraph framework.
        It demonstrates how to structure a graph to represent parallel processing and result aggregation.

        Test setup:
        1. Define two custom node functions:
           a. parallel_process:
              - Simulates parallel processing by creating multiple results.
              - For each of 3 parallel "tasks", it adds the input value to the task index.
              - Returns a list of these results in the state under 'parallel_results'.

           b. aggregate_results:
              - Simulates the aggregation of parallel processing results.
              - Sums up all the 'result' values from the 'parallel_results' list.
              - Returns this sum as 'final_result' in the state.

        2. Create a graph with three nodes:
           - "start": A pass-through node that doesn't modify the state.
           - "parallel": Uses the parallel_process function to simulate parallel execution.
           - "aggregate": Uses the aggregate_results function to combine parallel results.

        3. Set up the graph structure:
           - Set "start" as the entry point.
           - Add an edge from "start" to "parallel".
           - Add an edge from "parallel" to "aggregate".
           - Set "aggregate" as the finish point.

        Test execution:
        - Invoke the graph with an initial state of {"value": 5}.

        Expected workflow:
        1. Start node: Passes the initial state {"value": 5} unchanged.
        2. Parallel node:
           - Simulates three parallel tasks: 5+0, 5+1, 5+2
           - Produces state: {"parallel_results": [{"result": 5}, {"result": 6}, {"result": 7}]}
        3. Aggregate node:
           - Sums up the results: 5 + 6 + 7 = 18
           - Produces final state: {"final_result": 18}

        Assertion:
        - Verify that the final state contains "final_result" with a value of 18.

        Key concepts demonstrated:
        1. Simulation of parallel processing within a sequential graph structure.
        2. State transformation to represent parallel task results.
        3. Aggregation of parallel results into a final output.
        4. Flexibility of the StateGraph in modeling complex workflows.

        Limitations and considerations:
        - This test simulates parallel execution but does not actually perform parallel processing.
        - In a real-world scenario, true parallelism would require additional infrastructure outside the graph.
        - The test showcases how to structure a graph to represent and handle results from conceptually parallel operations.

        This test ensures that the graph can correctly simulate a parallel processing workflow,
        including the generation of multiple results and their subsequent aggregation,
        all within the sequential execution model of the StateGraph.
        """
        """
        Test a simulated parallel execution scenario, ensuring that results from parallel processes are correctly aggregated.
        """
        def parallel_process(state):
            # Simulate parallel processing
            results = [
                {"result": state["value"] + i}
                for i in range(3)
            ]
            return {"parallel_results": results}

        def aggregate_results(state):
            return {"final_result": sum(r["result"] for r in state["parallel_results"])}

        self.graph.add_node("start", run=lambda state: state)
        self.graph.add_node("parallel", run=parallel_process)
        self.graph.add_node("aggregate", run=aggregate_results)
        self.graph.set_entry_point("start")
        self.graph.add_edge("start", "parallel")
        self.graph.add_edge("parallel", "aggregate")
        self.graph.set_finish_point("aggregate")

        result = list(self.graph.invoke({"value": 5}))
        self.assertEqual(result[-1]["state"]["final_result"], 18)  # 5+0 + 5+1 + 5+2 = 18

    def test_document_writer_graph_structure_with_interruptions(self):
        
        def render_and_save_graph(graph, filename="state_graph.png"):
            try:
                graph.save_graph_image(filename)
                print(f"Graph image saved as {filename}")
            except Exception as e:
                print(f"Error saving graph image: {str(e)}")

        def create_node_function(name):
            def node_function(state, config=None):
                new_state = state.copy()
                new_state["state"] = name
                new_state["result"] = f"Executed {name}"
                return new_state
            node_function.__name__ = f"{name}_run"
            return node_function

        state_nodes = {
            "A": create_node_function("A"),
            "B": create_node_function("B"),
            "C": create_node_function("C"),
            "D": create_node_function("D"),
            "E": create_node_function("E"),
            "F": create_node_function("F"),
            "G": create_node_function("G"),
            "H": create_node_function("H"),
            "I": create_node_function("I"),
            "J": create_node_function("J"),
            "K": create_node_function("K"),
            "L": create_node_function("L"),
        }

        for name, func in state_nodes.items():
            self.graph.add_node(name, func)

        def is_b_complete(state):
            return "ahead" if not state.get("messages") else "behind"

        def is_e_complete(state, config):
            return "E" if config["configurable"]["instruction"] else "F"

        def is_g_complete(state, config):
            state["revision_number"] += 1
            if config["configurable"]["instruction"]:
                return "loop"
            elif state["revision_number"] <= state["max_revisions"]:
                return "condition2"
            else:
                return "condition"

        self.graph.add_conditional_edges("B", is_b_complete, {"ahead": "C", "behind": "A"})
        self.graph.add_conditional_edges("E", is_e_complete, {"E": "E", "F": "F"})
        self.graph.add_conditional_edges("G", is_g_complete, {"loop": "G", "condition2": "H", "condition": "I"})

        self.graph.add_edge("A", "B")
        self.graph.add_edge("C", "D")
        self.graph.add_edge("D", "E")
        self.graph.add_edge("F", "G")
        self.graph.add_edge("H", "F")
        self.graph.add_edge("I", "J")
        self.graph.add_edge("J", "K")
        self.graph.add_edge("K", "L")
        self.graph.add_edge("L", END)

        self.graph.set_entry_point("A")

        interrupt_before = ["B", "E", "G", "K"]
        interrupt_after = []
        compiled_graph = self.graph.compile(interrupt_before=interrupt_before, interrupt_after=interrupt_after)

        render_and_save_graph(compiled_graph, "anonymized_graph.png")

        initial_state = {"state": "A", "values": {"test": "initial"}, "revision_number": 0, "max_revisions": 3}
        config = {"configurable": {"instruction": ""}}

        max_iterations = 100
        events = []
        for i, event in enumerate(compiled_graph.stream(initial_state, config)):
            events.append(event)
            if i >= max_iterations:
                raise RuntimeError(f"Maximum number of iterations ({max_iterations}) reached. Possible infinite loop detected.")

        interrupt_points = [event for event in events if event.get("type") == "interrupt"]
        assert len(interrupt_points) > 0, "No interrupts occurred"

        interrupt_nodes = [event["node"] for event in interrupt_points]
        assert "B" in interrupt_nodes
        assert "E" in interrupt_nodes
        assert "G" in interrupt_nodes

        for interrupt in interrupt_points:
            assert interrupt["node"] in interrupt_before

        final_event = events[-1]
        print("Test completed successfully!")
        print(f"Total events: {len(events)}")
        print(f"Final event: {final_event}")

        assert "result" in final_event, "Final event should contain a 'result' key"
        assert final_event["result"] == "Executed L", "Unexpected final state"
        assert final_event["revision_number"] == 4, "Unexpected revision number in final state"
        assert final_event["state"] == "L", "Unexpected final state name"

if __name__ == '__main__':
    unittest.main()
