"""A minimal, safe calculator tool -- reused from rag-from-scratch's
ReAct-agent demo, kept here because arithmetic-in-agent-loop is a classic,
easy-to-verify tool-use example that's useful for the benchmark harness too
(it has an unambiguous ground truth)."""

from __future__ import annotations

import ast
import operator
from typing import Any

from src.core.interfaces import Tool

_OPS = {
    ast.Add: operator.add,
    ast.Sub: operator.sub,
    ast.Mult: operator.mul,
    ast.Div: operator.truediv,
    ast.Pow: operator.pow,
    ast.USub: operator.neg,
}


def _safe_eval(node: ast.AST) -> float:
    if isinstance(node, ast.BinOp) and type(node.op) in _OPS:
        return _OPS[type(node.op)](_safe_eval(node.left), _safe_eval(node.right))
    if isinstance(node, ast.UnaryOp) and type(node.op) in _OPS:
        return _OPS[type(node.op)](_safe_eval(node.operand))
    if isinstance(node, ast.Constant) and isinstance(node.value, (int, float)):
        return node.value
    raise ValueError("Unsupported expression")


class CalculatorTool(Tool):
    name = "calculator"
    description = "Evaluate a basic arithmetic expression, e.g. '12 * (4 + 3)'."

    def input_schema(self) -> dict[str, Any]:
        return {
            "type": "object",
            "properties": {"expression": {"type": "string", "description": "A basic arithmetic expression."}},
            "required": ["expression"],
        }

    def run(self, **kwargs: Any) -> str:
        expression = kwargs["expression"]
        try:
            tree = ast.parse(expression, mode="eval")
            result = _safe_eval(tree.body)
            return str(result)
        except Exception as exc:  # noqa: BLE001
            return f"Error evaluating expression: {exc}"
