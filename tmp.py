# a minimum automatic differentiation.

import typing

class Node:
    def __init__(self, inputs: list["Node"], op: "Op | None", value: float | None = None):
        self.inputs = inputs  # List of input Nodes
        self.op = op          # Operation that produced this Node
        self.value: float | None = value    # Value of the Node (if it's a leaf node)

    def __add__(self, other: "Node") -> "Node":
        return addOp(self, other)
    
    def __sub__(self, other: "Node") -> "Node":
        return subOp(self, other)
    
    def __mul__(self, other: "Node") -> "Node":
        return mulOp(self, other)
    
    def __truediv__(self, other: "Node") -> "Node":
        return divOp(self, other)
    
    def eval(self) -> float:
        if self.value is not None:
            return self.value
        input_values = [input_node.eval() for input_node in self.inputs]
        if self.op is None:
            raise ValueError("Node is not connected to an operation.")
        self.value = self.op.forward(input_values)
        return self.value
    
    def __str__(self) -> str:
        return f"{self.eval()}"
    
    def __repr__(self) -> str:
        return self.__str__()

class Op(typing.Protocol):
    def forward(self, inputs: list[float]) -> float:
        ...
    
    def backward(self, out_grad: Node, node: Node) -> list[Node]:
        ...

class AddOp:
    def __call__(self, node_a: Node, node_b: Node) -> Node:
        return Node([node_a, node_b], self)

    def forward(self, inputs: list[float]) -> float:
        return inputs[0] + inputs[1]
    
    def backward(self, out_grad: Node, node: Node) -> list[Node]:
        return [out_grad, out_grad]
    
class SubOp:
    def __call__(self, node_a: Node, node_b: Node) -> Node:
        return Node([node_a, node_b], self)

    def forward(self, inputs: list[float]) -> float:
        return inputs[0] - inputs[1]
    
    def backward(self, out_grad: Node, node: Node) -> list[Node]:
        return [out_grad, Node([], None, value=-1) * out_grad]
    
class MulOp:
    def __call__(self, node_a: Node, node_b: Node) -> Node:
        return Node([node_a, node_b], self)

    def forward(self, inputs: list[float]) -> float:
        return inputs[0] * inputs[1]
    
    def backward(self, out_grad: Node, node: Node) -> list[Node]:
        return [node.inputs[1] * out_grad, node.inputs[0] * out_grad]
    
class DivOp:
    def __call__(self, node_a: Node, node_b: Node) -> Node:
        return Node([node_a, node_b], self)

    def forward(self, inputs: list[float]) -> float:
        return inputs[0] / inputs[1]
    
    def backward(self, out_grad: Node, node: Node) -> list[Node]:
        return [out_grad * (Node([], None, value=1) / node.inputs[1]),
                out_grad * (Node([], None, value=-1) * node.inputs[0] / (node.inputs[1] * node.inputs[1]))]
    
    
addOp = AddOp()
subOp = SubOp()
mulOp = MulOp()
divOp = DivOp()

def gradients(output_node: Node, input_nodes: list[Node]) -> list[Node]:
    grads = {output_node: Node([], None, value=1)}  # Gradient of output w.r.t itself is 1
    topo_order: list[Node] = []
    visited: set[Node] = set()

    def build_topo(node: Node):
        if node not in visited:
            visited.add(node)
            for inp in node.inputs:
                build_topo(inp)
            topo_order.append(node)

    build_topo(output_node)

    for node in reversed(topo_order):
        if node.op is not None:
            out_grad = grads[node]
            in_grads = node.op.backward(out_grad, node)
            for inp, in_grad in zip(node.inputs, in_grads):
                if inp in grads:
                    grads[inp] = grads[inp] + in_grad
                else:
                    grads[inp] = in_grad

    return [grads.get(inp, Node([], None, value=0)) for inp in input_nodes]

if __name__ == "__main__":
    a = Node([], None, value=2)
    b = Node([], None, value=3)

    c = a * a - b * b + a / b

    # grad_a = 2*a + 1/b
    # grad_b = -2*b - a/(b*b)
    print(gradients(c, [a, b]))
