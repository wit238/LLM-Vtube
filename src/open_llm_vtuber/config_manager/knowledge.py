# config_manager/knowledge.py
from pydantic import Field
from typing import Dict, ClassVar
from .i18n import I18nMixin, Description


class KnowledgeConfig(I18nMixin):
    """Configuration for the file knowledge base (RAG over markdown files).

    Drop `.md` files into `folder_path`; on every server start they are
    indexed (chunked + embedded) and at conversation time the most relevant
    chunks are retrieved and injected into the LLM context so answers are
    grounded in those documents.
    """

    enabled: bool = Field(True, alias="enabled")
    folder_path: str = Field("knowledge_md", alias="folder_path")
    embedding_model: str = Field("BAAI/bge-m3", alias="embedding_model")
    embedding_device: str = Field("cpu", alias="embedding_device")
    chunk_size: int = Field(1000, alias="chunk_size")
    chunk_overlap: int = Field(150, alias="chunk_overlap")
    top_k: int = Field(4, alias="top_k")
    min_score: float = Field(0.2, alias="min_score")
    cache_dir: str = Field("cache/knowledge", alias="cache_dir")
    extension: str = Field(".md", alias="extension")
    include_sources: bool = Field(True, alias="include_sources")

    DESCRIPTIONS: ClassVar[Dict[str, Description]] = {
        "enabled": Description(
            en="Enable the file knowledge base (RAG)", zh="启用文件知识库（RAG）"
        ),
        "folder_path": Description(
            en="Folder to scan for markdown files; drop .md files in here",
            zh="扫描 Markdown 文件的文件夹；把 .md 文件放进来",
        ),
        "embedding_model": Description(
            en="Sentence-embedding model used to index/search the documents",
            zh="用于索引/搜索文档的句子嵌入模型",
        ),
        "embedding_device": Description(
            en="Device for the embedding model ('cpu' recommended so it doesn't fight the JaiTTS CUDA pipeline)",
            zh="嵌入模型的设备（建议 'cpu'，避免与 JaiTTS 的 CUDA 管线争抢）",
        ),
        "chunk_size": Description(
            en="Target characters per chunk", zh="每个文本块的预期字符数"
        ),
        "chunk_overlap": Description(
            en="Characters of overlap between consecutive chunks",
            zh="相邻文本块之间的重叠字符数",
        ),
        "top_k": Description(
            en="Number of most relevant chunks injected into the LLM context",
            zh="注入 LLM 上下文的最相关文本块数量",
        ),
        "min_score": Description(
            en="Minimum cosine similarity for a chunk to be retrieved (0-1)",
            zh="检索文本块的最低余弦相似度（0-1）",
        ),
        "cache_dir": Description(
            en="Directory for the persistent chunk+embedding cache",
            zh="持久化文本块与嵌入缓存的目录",
        ),
        "extension": Description(
            en="File extension to index (e.g. '.md', '.txt')",
            zh="要索引的文件扩展名（如 '.md'、'.txt'）",
        ),
        "include_sources": Description(
            en="Attach source filename/heading to each retrieved chunk",
            zh="在每个检索到的文本块上附加来源文件名/标题",
        ),
    }
