# AI Engineering Notes

Retrieval augmented generation (RAG) combines a retrieval system with a
generative language model. The retriever searches a knowledge base for
passages relevant to the user's query, and the generator conditions its
answer on those passages, which reduces hallucination compared to a model
answering from parametric memory alone.

An AI agent extends a language model with the ability to take actions in
an environment via tools, and to reason about which action to take next.
The ReAct pattern interleaves Thought, Action, and Observation steps in a
loop until the model decides it has enough information to produce a Final
Answer.

Evaluation of retrieval systems commonly uses precision at k, recall at k,
and mean reciprocal rank. Evaluation of agents additionally considers the
number of tool calls, latency, and token cost per query, since two agents
that reach the same final answer can differ enormously in how expensive it
was to get there.
