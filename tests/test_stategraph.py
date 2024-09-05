import unittest
import networkx as nx
from graphviz import Digraph
from unittest.mock import Mock, patch
from parameterized import parameterized
from hypothesis import given, strategies as st, settings, assume
from sololgraph import StateGraph, START, END

class TestStateGraph(unittest.TestCase):

    def setUp(self):
        self.graph = StateGraph({"test": "schema"})

    def test_init(self):
        self.assertEqual(self.graph.state_schema, {"test": "schema"})

    @parameterized.expand([
        ("simple_node", "test_node", None),
        ("node_with_function", "func_node", lambda: None),
    ])
    def test_add_node(self, name, node, run):
        self.graph.add_node(node, run)
        node_data = self.graph.node(node)
        if run:
            self.assertEqual(node_data["run"], run)

    def test_add_node_duplicate(self):
        self.graph.add_node("test_node")
        with self.assertRaises(ValueError):
            self.graph.add_node("test_node")

    def test_add_edge(self):
        self.graph.add_node("node1")
        self.graph.add_node("node2")
        self.graph.add_edge("node1", "node2")
        self.assertIn("node2", self.graph.successors("node1"))

    def test_add_edge_nonexistent_node(self):
        with self.assertRaises(ValueError):
            self.graph.add_edge("nonexistent1", "nonexistent2")

    def test_add_conditional_edges(self):
        self.graph.add_node("node1")
        self.graph.add_node("node2")
        self.graph.add_node("node3")

        def condition_func(state):
            return state["value"]

        self.graph.add_conditional_edges("node1", condition_func, {True: "node2", False: "node3"})

        self.assertIn("node2", self.graph.successors("node1"))
        self.assertIn("node3", self.graph.successors("node1"))

    def test_add_conditional_edges_nonexistent_node(self):
        with self.assertRaises(ValueError):
            self.graph.add_conditional_edges("nonexistent", lambda: True, {True: "node2"})

    def test_set_entry_point(self):
        self.graph.add_node("start_node")
        self.graph.set_entry_point("start_node")
        self.assertIn("start_node", self.graph.successors(START))

    def test_set_entry_point_nonexistent_node(self):
        with self.assertRaises(ValueError):
            self.graph.set_entry_point("nonexistent")

    def test_set_finish_point(self):
        self.graph.add_node("end_node")
        self.graph.set_finish_point("end_node")
        self.assertIn(END, self.graph.successors("end_node"))

    def test_set_finish_point_nonexistent_node(self):
        with self.assertRaises(ValueError):
            self.graph.set_finish_point("nonexistent")

    def test_compile(self):
        self.graph.add_node("node1")
        self.graph.add_node("node2")
        compiled_graph = self.graph.compile(interrupt_before=["node1"], interrupt_after=["node2"])
        self.assertEqual(compiled_graph.interrupt_before, ["node1"])
        self.assertEqual(compiled_graph.interrupt_after, ["node2"])

    def test_compile_nonexistent_node(self):
        with self.assertRaises(ValueError):
            self.graph.compile(interrupt_before=["nonexistent"])

    def test_node(self):
        self.graph.add_node("test_node", run=lambda: None)
        node_data = self.graph.node("test_node")
        self.assertIsNotNone(node_data["run"])

    def test_node_nonexistent(self):
        with self.assertRaises(KeyError):
            self.graph.node("nonexistent_node")

    def test_edge(self):
        self.graph.add_node("node1")
        self.graph.add_node("node2")
        self.graph.add_edge("node1", "node2")
        edge_data = self.graph.edge("node1", "node2")
        self.assertIn("cond", edge_data)

    def test_edge_nonexistent(self):
        with self.assertRaises(KeyError):
            self.graph.edge("nonexistent1", "nonexistent2")

    def test_successors(self):
        self.graph.add_node("node1")
        self.graph.add_node("node2")
        self.graph.add_node("node3")
        self.graph.add_edge("node1", "node2")
        self.graph.add_edge("node1", "node3")
        successors = self.graph.successors("node1")
        self.assertIn("node2", successors)
        self.assertIn("node3", successors)

    def test_successors_nonexistent(self):
        with self.assertRaises(KeyError):
            self.graph.successors("nonexistent_node")

    def test_stream(self):
        self.graph.add_node("node1", run=lambda state: {"value": state["value"] + 1})
        self.graph.add_node("node2", run=lambda state: {"value": state["value"] * 2})
        self.graph.set_entry_point("node1")
        self.graph.add_edge("node1", "node2")
        self.graph.set_finish_point("node2")

        stream = list(self.graph.stream({"value": 1}))
        self.assertEqual(len(stream), 1)
        self.assertEqual(stream[0]["value"], 4)

    def test_invoke_simple(self):
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

    def test_render_graphviz(self):
        self.graph.add_node("node1")
        self.graph.add_node("node2")
        self.graph.add_edge("node1", "node2")
        dot = self.graph.render_graphviz()
        self.assertIsInstance(dot, Digraph)

    def test_complex_workflow(self):
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
        self.graph.add_node("node1", run=lambda state: {"count": state.get("count", 0) + 1})
        self.graph.add_node("node2", run=lambda state: {"count": state["count"] * 2})
        self.graph.set_entry_point("node1")
        self.graph.add_conditional_edges("node1", lambda state: state["count"] >= 3, {True: "node2", False: "node1"})
        self.graph.set_finish_point("node2")

        result = list(self.graph.invoke({"count": 0}))
        self.assertEqual(result[-1]["state"]["count"], 6)

    def test_error_handling_in_node_function(self):
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
        mock_get_next_node.side_effect = RuntimeError("No valid next node")

        self.graph.add_node("start")
        self.graph.set_entry_point("start")
        self.graph.set_finish_point("start")

        with self.assertRaises(RuntimeError):
            list(self.graph.invoke({}))

    def test_complex_conditional_routing(self):
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
        def configurable_func(state, config):
            return {"result": state["value"] * config["multiplier"]}

        self.graph.add_node("start", run=configurable_func)
        self.graph.set_entry_point("start")
        self.graph.set_finish_point("start")

        result = list(self.graph.invoke({"value": 5}, config={"multiplier": 3}))
        self.assertEqual(result[-1]["state"]["result"], 15)

    def test_multiple_entry_points(self):
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
        def faulty_condition(state):
            raise ValueError("Conditional error")

        self.graph.add_node("start", run=lambda state: state)
        self.graph.set_entry_point("start")
        self.graph.add_conditional_edges("start", faulty_condition, {True: END, False: "start"})

        with self.assertRaises(ValueError):
            list(self.graph.invoke({"value": 0}))

    def test_complex_graph_visualization(self):
        # Create a more complex graph
        self.graph.add_node("start", run=lambda state: state)
        self.graph.add_node("process1", run=lambda state: {"value": state["value"] + 1})
        self.graph.add_node("process2", run=lambda state: {"value": state["value"] * 2})
        self.graph.add_node("decision", run=lambda state: state)
        self.graph.add_node("end", run=lambda state: {"result": f"Final: {state['value']}"})

        self.graph.set_entry_point("start")
        self.graph.add_edge("start", "process1")
        self.graph.add_edge("process1", "process2")
        self.graph.add_edge("process2", "decision")
        self.graph.add_conditional_edges("decision", lambda state: state["value"] > 10, {True: "end", False: "process1"})
        self.graph.set_finish_point("end")

        # Render the graph
        dot = self.graph.render_graphviz()

        # Check if all nodes are in the rendered graph
        for node in ["start", "process1", "process2", "decision", "end"]:
            self.assertIn(node, dot.source)

        # Check if conditional edge is properly labeled
        self.assertIn("condition", dot.source)

    def test_interrupt_handling(self):
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

if __name__ == '__main__':
    unittest.main()
