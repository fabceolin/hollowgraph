import inspect
from typing import Any, Callable, Dict, List, Optional, Union, Generator
import networkx as nx
from networkx.drawing.nx_agraph import to_agraph


# Copyright (c) 2024 Claudionor Coelho Jr, Fabrício Ceolin

START = "__start__"
END = "__end__"

class StateGraph:
    """
    A graph-based state machine for managing complex workflows.

    This class allows defining states, transitions, and conditions for state changes,
    as well as executing the workflow based on the defined graph structure.

    Attributes:
        state_schema (Dict[str, Any]): The schema defining the structure of the state.
        graph (nx.DiGraph): The directed graph representing the state machine.
        interrupt_before (List[str]): Nodes to interrupt before execution.
        interrupt_after (List[str]): Nodes to interrupt after execution.

    Example:
        >>> graph = StateGraph({"value": int})
        >>> graph.add_node("start", run=lambda state: {"value": state["value"] + 1})
        >>> graph.add_node("end", run=lambda state: {"result": f"Final value: {state['value']}"})
        >>> graph.set_entry_point("start")
        >>> graph.set_finish_point("end")
        >>> graph.add_edge("start", "end")
        >>> result = list(graph.invoke({"value": 1}))
        >>> print(result[-1]["state"]["result"])
        Final value: 2
    """

    def __init__(self, state_schema: Dict[str, Any], raise_exceptions: bool = False):
        """
        Initialize the StateGraph.

        Args:
            state_schema (Dict[str, Any]): The schema defining the structure of the state.
            raise_exceptions (bool): If True, exceptions in node functions will be raised instead of being handled internally.
        """
        self.state_schema = state_schema
        self.graph = nx.DiGraph()
        self.graph.add_node(START, run=None)
        self.graph.add_node(END, run=None)
        self.interrupt_before: List[str] = []
        self.interrupt_after: List[str] = []
        self.raise_exceptions = raise_exceptions

    def add_node(self, node: str, run: Optional[Callable[..., Any]] = None) -> None:
        """
        Add a node to the graph.

        Args:
            node (str): The name of the node.
            run (Optional[Callable[..., Any]]): The function to run when this node is active.

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
            raise ValueError("Both nodes must exist in the graph.")
        self.graph.add_edge(in_node, out_node, cond=lambda **kwargs: True, cond_map={True: out_node})

    def add_conditional_edges(self, in_node: str, func: Callable[..., Any], cond: Dict[Any, str]) -> None:
        """
        Add conditional edges from a node based on a function's output.

        Args:
            in_node (str): The source node.
            func (Callable[..., Any]): The function to determine the next node.
            cond (Dict[Any, str]): Mapping of function outputs to target nodes.

        Raises:
            ValueError: If the source node doesn't exist or if any target node is invalid.
        """
        if in_node not in self.graph.nodes:
            raise ValueError(f"Node '{in_node}' does not exist in the graph.")
        for cond_value, out_node in cond.items():
            if out_node not in self.graph.nodes:
                raise ValueError(f"Target node '{out_node}' does not exist in the graph.")
            self.graph.add_edge(in_node, out_node, cond=func, cond_map=cond)

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
        self.graph.add_edge(START, init_state, cond=lambda **kwargs: True, cond_map={True: init_state})

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
        self.graph.add_edge(final_state, END, cond=lambda **kwargs: True, cond_map={True: END})

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

    def invoke(self, input_state: Dict[str, Any] = {}, config: Dict[str, Any] = {}) -> Generator[Dict[str, Any], None, None]:
        """
        Execute the graph, yielding intermediate and final states.

        Args:
            input_state (Dict[str, Any]): The initial state.
            config (Dict[str, Any]): Configuration for the execution.

        Yields:
            Dict[str, Any]: Intermediate states, interrupts, and the final state during execution.

        Raises:
            RuntimeError: If no valid next node is found during execution or if an exception occurs in a node function when raise_exceptions is True.
        """
        current_state = {"values": input_state, "next": START}

        while current_state["next"] != END:
            current_node = current_state["next"]
            node_data = self.node(current_node)

            if current_node in self.interrupt_before:
                yield {"type": "interrupt", "node": current_node, "state": current_state["values"]}

            if node_data.get("run"):
                result = self._execute_node_function(node_data["run"], current_state["values"], config, current_node)
                if "error" in result:
                    yield {"type": "error", "node": current_node, "error": result["error"], "state": current_state["values"]}
                    return
                current_state["values"].update(result)

            if current_node in self.interrupt_after:
                yield {"type": "interrupt", "node": current_node, "state": current_state["values"]}

            current_state["next"] = self._get_next_node(current_node, current_state["values"], config)

        yield {"type": "final", "state": current_state["values"]}

    def _execute_node_function(self, func: Callable[..., Any], state: Dict[str, Any], config: Dict[str, Any], node: str) -> Dict[str, Any]:
        """
        Execute the function associated with a node.

        Args:
            func (Callable[..., Any]): The function to execute.
            state (Dict[str, Any]): The current state.
            config (Dict[str, Any]): The configuration.
            node (str): The current node name.

        Returns:
            Dict[str, Any]: The result of the function execution or an error state.

        Raises:
            RuntimeError: If raise_exceptions is True and an exception occurs during function execution.
        """
        available_params = {"state": state, "config": config, "node": node, "graph": self}
        function_params = self._prepare_function_params(func, available_params)
        
        try:
            result = func(**function_params)
        except Exception as e:
            if self.raise_exceptions:
                raise RuntimeError(f"Error in node {node}: {str(e)}") from e
            return {"error": str(e), "node": node}

        if isinstance(result, dict):
            return result
        else:
            return {"result": result}

    def _prepare_function_params(self, func: Callable[..., Any], available_params: Dict[str, Any]) -> Dict[str, Any]:
        """
        Prepare the parameters for a node function based on its signature.

        Args:
            func (Callable[..., Any]): The function to prepare parameters for.
            available_params (Dict[str, Any]): Dictionary of available parameters.

        Returns:
            Dict[str, Any]: The prepared parameters for the function.

        Raises:
            ValueError: If required parameters for the function are not provided.
        """
        sig = inspect.signature(func)
        function_params = {}

        if len(sig.parameters) == 0:
            return {}

        for param_name, param in sig.parameters.items():
            if param_name in available_params:
                function_params[param_name] = available_params[param_name]
            elif param.default is not inspect.Parameter.empty:
                function_params[param_name] = param.default
            elif param.kind == inspect.Parameter.VAR_KEYWORD:
                function_params.update({k: v for k, v in available_params.items() if k not in function_params})
                break
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
                condition_result = cond_func(**cond_params)

                if "cond_map" in edge_data:
                    if condition_result in edge_data["cond_map"]:
                        return edge_data["cond_map"][condition_result]
                else:
                    if condition_result:
                        return successor

        raise RuntimeError(f"No valid next node found for '{current_node}'")

    def render_graphviz(self):
        """
        Render the graph using NetworkX and Graphviz.

        Returns:
            pygraphviz.AGraph: A PyGraphviz graph object representing the StateGraph.
        """
        # Create a new directed graph
        G = nx.DiGraph()

        # Add nodes with attributes
        for node in self.graph.nodes():
            label = f"{node}\n"
            label += f"interrupt_before: {node in self.interrupt_before}\n"
            label += f"interrupt_after: {node in self.interrupt_after}"
            G.add_node(node, label=label)

        # Add edges with attributes
        for u, v, data in self.graph.edges(data=True):
            edge_label = ""
            if 'cond' in data:
                cond = data['cond']
                if callable(cond) and cond.__name__ != '<lambda>':
                    edge_label = "condition"
                elif isinstance(cond, dict) and len(cond) > 1:
                    edge_label = "condition"
            G.add_edge(u, v, label=edge_label)

        # Convert to a PyGraphviz graph
        A = to_agraph(G)

        # Set graph attributes
        A.graph_attr.update(rankdir="TB", size="8,8")
        A.node_attr.update(shape="rectangle", style="filled", fillcolor="white")
        A.edge_attr.update(color="black")

        return A


    def save_graph_image(self, filename="state_graph.png"):
        """
        Save the graph as an image file.

        Args:
            filename (str): The name of the file to save the graph image to.
        """
        A = self.render_graphviz()
        A.layout(prog='dot')
        A.draw(filename)
