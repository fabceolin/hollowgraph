import inspect
import networkx as nx
from graphviz import Digraph
from typing import Any, Callable, Dict, List, Optional, Union, Generator
from langchain_core.runnables.utils import AddableDict

# Copyright (c) 2024 Claudionor Coelho Jr, Fabrício Ceolin

START = "__start__"
END = "__end__"


class StateGraph:
    """
    A graph-based state machine for managing complex workflows.

    This class allows defining states, transitions, and conditions for state changes,
    as well as executing the workflow based on the defined graph structure.
    """

    def __init__(self, state_schema: Dict[str, Any]):
        """
        Initialize the StateGraph.

        Args:
            state_schema (Dict[str, Any]): The schema defining the structure of the state.
        """
        self.state_schema = state_schema
        self.graph = nx.DiGraph()
        self.graph.add_node(START, run=None)
        self.graph.add_node(END, run=None)
        self.interrupt_before: List[str] = []
        self.interrupt_after: List[str] = []

    def add_node(self, node: str, run: Optional[Callable] = None) -> None:
        """
        Add a node to the graph.

        Args:
            node (str): The name of the node.
            run (Optional[Callable]): The function to run when this node is active.

        Raises:
            ValueError: If the node already exists in the graph.
        """
        if node in self.graph.nodes:
            raise ValueError(f"Node '{node}' already exists in the graph.")
        self.graph.add_node(node, run=run)

    def add_edge(self, in_node: str, out_node: str) -> None:
        """
        Add an unconditional edge between two nodes.

        Args:
            in_node (str): The source node.
            out_node (str): The target node.

        Raises:
            ValueError: If either node doesn't exist in the graph.
        """
        if in_node not in self.graph.nodes or out_node not in self.graph.nodes:
            raise ValueError(f"Both nodes must exist in the graph.")
        self.graph.add_edge(in_node, out_node, cond=lambda _: True)

    def add_conditional_edges(self, in_node: str, func: Callable, cond: Dict[Any, str]) -> None:
        """
        Add conditional edges from a node based on a function's output.

        Args:
            in_node (str): The source node.
            func (Callable): The function to determine the next node.
            cond (Dict[Any, str]): Mapping of function outputs to target nodes.

        Raises:
            ValueError: If the source node doesn't exist or if any target node is invalid.
        """
        if in_node not in self.graph.nodes:
            raise ValueError(f"Node '{in_node}' does not exist in the graph.")

        for cond_value, out_node in cond.items():
            if out_node not in self.graph.nodes:
                raise ValueError(f"Target node '{out_node}' does not exist in the graph.")
            self.graph.add_edge(
                in_node,
                out_node,
                cond=lambda state, f=func, cv=cond_value: f(state) == cv
            )

    def set_entry_point(self, init_state: str) -> None:
        """
        Set the entry point of the graph.

        Args:
            init_state (str): The initial state node.

        Raises:
            ValueError: If the initial state node doesn't exist in the graph.
        """
        if init_state not in self.graph.nodes:
            raise ValueError(f"Node '{init_state}' does not exist in the graph.")
        self.graph.add_edge(START, init_state, cond=lambda _: True)

    def set_finish_point(self, final_state: str) -> None:
        """
        Set the finish point of the graph.

        Args:
            final_state (str): The final state node.

        Raises:
            ValueError: If the final state node doesn't exist in the graph.
        """
        if final_state not in self.graph.nodes:
            raise ValueError(f"Node '{final_state}' does not exist in the graph.")
        self.graph.add_edge(final_state, END, cond=lambda _: True)

    def compile(self, interrupt_before: List[str] = [], interrupt_after: List[str] = []) -> 'StateGraph':
        """
        Compile the graph and set interruption points.

        Args:
            interrupt_before (List[str]): Nodes to interrupt before execution.
            interrupt_after (List[str]): Nodes to interrupt after execution.

        Returns:
            StateGraph: The compiled graph instance.

        Raises:
            ValueError: If any interrupt node doesn't exist in the graph.
        """
        for node in interrupt_before + interrupt_after:
            if node not in self.graph.nodes:
                raise ValueError(f"Interrupt node '{node}' does not exist in the graph.")

        self.interrupt_before = interrupt_before
        self.interrupt_after = interrupt_after
        return self

    def node(self, node_name: str) -> Dict[str, Any]:
        """
        Get the attributes of a specific node.

        Args:
            node_name (str): The name of the node.

        Returns:
            Dict[str, Any]: The node's attributes.

        Raises:
            KeyError: If the node is not found in the graph.
        """
        if node_name not in self.graph.nodes:
            raise KeyError(f"Node '{node_name}' not found in the graph")
        return self.graph.nodes[node_name]

    def edge(self, in_node: str, out_node: str) -> Dict[str, Any]:
        """
        Get the attributes of a specific edge.

        Args:
            in_node (str): The source node.
            out_node (str): The target node.

        Returns:
            Dict[str, Any]: The edge's attributes.

        Raises:
            KeyError: If the edge is not found in the graph.
        """
        if not self.graph.has_edge(in_node, out_node):
            raise KeyError(f"Edge from '{in_node}' to '{out_node}' not found in the graph")
        return self.graph.edges[in_node, out_node]

    def successors(self, node: str) -> List[str]:
        """
        Get the list of successors for a given node.

        Args:
            node (str): The name of the node.

        Returns:
            List[str]: A list of successor node names.

        Raises:
            KeyError: If the node is not found in the graph.
        """
        if node not in self.graph.nodes:
            raise KeyError(f"Node '{node}' not found in the graph")
        return list(self.graph.successors(node))

    def stream(self, input_state: Dict[str, Any] = {}, config: Dict[str, Any] = {}) -> Generator[Dict[str, Any], None, None]:
        """
        Stream the execution of the graph, yielding intermediate states.

        Args:
            input_state (Dict[str, Any]): The initial state.
            config (Dict[str, Any]): Configuration for the execution.

        Yields:
            Dict[str, Any]: Intermediate states and interrupts during execution.
        """
        for result in self.invoke(input_state, config):
            if result["type"] == "interrupt":
                yield result
            elif result["type"] == "final":
                yield result["state"]
                return
            else:
                yield result

    def invoke(self, input_state: Dict[str, Any] = {}, config: Dict[str, Any] = {}) -> Dict[str, Any]:
        current_state = {"values": input_state, "next": START}

        while current_state["next"] != END:
            current_node = current_state["next"]
            node_data = self.node(current_node)

            if "run" in node_data and node_data["run"]:
                node_function = node_data["run"]
                available_params = {
                    "state": current_state["values"],
                    "config": config,
                    "node": current_node,
                    "graph": self
                }
                function_params = self._prepare_function_params(node_function, available_params)
                result = node_function(**function_params)

                if isinstance(result, dict):
                    current_state["values"].update(result)
                else:
                    current_state["values"]["result"] = result

            next_node = self._get_next_node(current_node, current_state["values"], config)
            current_state["next"] = next_node

        return current_state["values"]

    def _prepare_function_params(
            self, 
            func: Callable, 
            available_params: Dict[str, Any]
        ) -> Dict[str, Any]:
            """
            Prepare the parameters for a node function based on its signature.
    
            Args:
                func (Callable): The function to prepare parameters for.
                available_params (Dict[str, Any]): Dictionary of available parameters.
    
            Returns:
                Dict[str, Any]: The prepared parameters for the function.
            """
            sig = inspect.signature(func)
            function_params = {}
    
            for param_name, param in sig.parameters.items():
                if param_name in available_params:
                    function_params[param_name] = available_params[param_name]
                elif param.default is not inspect.Parameter.empty:
                    function_params[param_name] = param.default
                else:
                    raise ValueError(f"Required parameter '{param_name}' not provided for function '{func.__name__}'")
    
            return function_params

    def _get_next_node(self, current_node: str, state: Dict[str, Any], config: Dict[str, Any]) -> str:
        """
        Determine the next node based on the current node's successors and conditions.

        Args:
            current_node (str): The current node.
            state (Dict[str, Any]): The current state.
            config (Dict[str, Any]): The configuration.

        Returns:
            str: The name of the next node.

        Raises:
            RuntimeError: If no valid next node is found.
        """
        for successor in self.successors(current_node):
            edge_data = self.edge(current_node, successor)
            if "cond" in edge_data:
                cond_func = edge_data["cond"]
                available_params = {"state": state, "config": config}
                cond_params = self._prepare_function_params(cond_func, available_params)
                if cond_func(**cond_params):
                    return successor

        raise RuntimeError(f"No valid next node found for '{current_node}'")

    def render_graphviz(self) -> Digraph:
        """
        Render the graph using Graphviz.

        Returns:
            Digraph: A Graphviz representation of the graph.
        """
        dot = Digraph()

        for node in self.graph.nodes:
            label = node
            if node in self.interrupt_before:
                label += "\ninterrupt_before: True"
            else:
                label += "\ninterrupt_before: False"

            if node in self.interrupt_after:
                label += "\ninterrupt_after: True"
            else:
                label += "\ninterrupt_after: False"

            dot.node(node, label=label)

        for u, v in self.graph.edges:
            edge_label = ""
            cond_func = self.graph.edges[u, v]["cond"]
            if cond_func:
                edge_label = "condition"
            dot.edge(u, v, label=edge_label)

        return dot
