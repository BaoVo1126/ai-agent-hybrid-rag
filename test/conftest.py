from __future__ import annotations

import pytest

from src.core.interfaces import Document
from src.retrieval.hybrid import HybridRetriever
from src.tools.calculator_tool import CalculatorTool
from src.tools.rag_tool import DocumentSearchTool
from src.tools.registry import ToolRegistry
from src.tools.summarize_tool import SummarizeTool


@pytest.fixture
def sample_documents() -> list[Document]:
    return [
        Document(
            id="doc1::chunk_0",
            text=(
                "Retrieval augmented generation combines a retriever with a language model. "
                "The retriever finds relevant passages and the model conditions its answer on them."
            ),
            metadata={"source": "sample.txt", "page": 1},
        ),
        Document(
            id="doc2::chunk_0",
            text=(
                "An AI agent is a system that perceives its environment through tools and takes "
                "actions to achieve a goal, often using a reasoning loop such as ReAct."
            ),
            metadata={"source": "sample.txt", "page": 2},
        ),
        Document(
            id="doc3::chunk_0",
            text="Evaluation metrics for retrieval include precision at k, recall at k, and mean reciprocal rank.",
            metadata={"source": "sample.txt", "page": 3},
        ),
    ]


@pytest.fixture
def hybrid_retriever(sample_documents) -> HybridRetriever:
    retriever = HybridRetriever()
    retriever.fit(sample_documents)
    return retriever


@pytest.fixture
def tool_registry(hybrid_retriever) -> ToolRegistry:
    return ToolRegistry(
        [
            DocumentSearchTool(hybrid_retriever),
            CalculatorTool(),
            SummarizeTool(),
        ]
    )
