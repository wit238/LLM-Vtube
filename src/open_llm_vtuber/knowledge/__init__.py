"""File knowledge base (RAG) over a folder of markdown files.

Modules:
  - chunker: markdown -> chunks
  - embedder: lazy transformers embedding model
  - knowledge_base: index + retrieval + persistent cache
  - base: shared instance accessor used by run_server / conversations
"""
