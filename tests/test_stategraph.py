import networkx as nx
from graphviz import Digraph
import unittest
from unittest.mock import Mock
from agents.stategraph import StateGraph, START, END

class TestStateGraph(unittest.TestCase):

    def setUp(self):
        self.graph = StateGraph({"test": "schema"})

    def test_init(self):
        self.assertIsInstance(self.graph.graph, nx.DiGraph)
        self.assertIn(START, self.graph.graph.nodes)
        self.assertIn(END, self.graph.graph.nodes)

    def test_add_node(self):
        self.graph.add_node("test_node")
        self.assertIn("test_node", self.graph.graph.nodes)

    def test_add_node_duplicate(self):
        self.graph.add_node("test_node")
        with self.assertRaises(ValueError):
            self.graph.add_node("test_node")

    def test_add_edge(self):
        self.graph.add_node("node1")
        self.graph.add_node("node2")
        self.graph.add_edge("node1", "node2")
        self.assertTrue(self.graph.graph.has_edge("node1", "node2"))

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
        
        self.assertTrue(self.graph.graph.has_edge("node1", "node2"))
        self.assertTrue(self.graph.graph.has_edge("node1", "node3"))

    def test_set_entry_point(self):
        self.graph.add_node("start_node")
        self.graph.set_entry_point("start_node")
        self.assertTrue(self.graph.graph.has_edge(START, "start_node"))

    def test_set_finish_point(self):
        self.graph.add_node("end_node")
        self.graph.set_finish_point("end_node")
        self.assertTrue(self.graph.graph.has_edge("end_node", END))

    def test_compile(self):
        self.graph.add_node("node1")
        self.graph.add_node("node2")
        compiled_graph = self.graph.compile(interrupt_before=["node1"], interrupt_after=["node2"])
        self.assertEqual(compiled_graph.interrupt_before, ["node1"])
        self.assertEqual(compiled_graph.interrupt_after, ["node2"])

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

    def test_invoke(self):
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
    
        # Test invoke method
        invoke_result = list(self.graph.invoke({"value": 1}))
        
        print("Invoke result:", invoke_result)  # For debugging
        
        # Check that we get only one result (the final state)
        self.assertEqual(len(invoke_result), 1)
        
        # Check the final result
        self.assertTrue("state" in invoke_result[0])
        self.assertTrue("result" in invoke_result[0]["state"])
        self.assertEqual(invoke_result[0]["state"]["result"], "Final value: 17")
    
        # Test stream method
        stream_result = list(self.graph.stream({"value": 1}))
        
        print("Stream result:", stream_result)  # For debugging
        
        # Check that we get only one result (the final state)
        self.assertEqual(len(stream_result), 1)
        
        # Check the final result in the stream
        self.assertTrue("result" in stream_result[0])
        self.assertEqual(stream_result[0]["result"], "Final value: 17")


if __name__ == '__main__':
    unittest.main()
