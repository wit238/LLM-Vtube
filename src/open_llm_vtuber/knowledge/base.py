"""Module-level access to the shared file knowledge base instance.

`run_server` builds a KnowledgeBase at startup and registers it here;
`single_conversation` reads it to retrieve context for each user turn.
"""

import threading
from typing import Optional

from .knowledge_base import KnowledgeBase

_knowledge_base: Optional[KnowledgeBase] = None
_lock = threading.Lock()


def set_knowledge_base(kb: Optional[KnowledgeBase]) -> None:
    global _knowledge_base
    with _lock:
        _knowledge_base = kb


def get_knowledge_base() -> Optional[KnowledgeBase]:
    with _lock:
        return _knowledge_base
