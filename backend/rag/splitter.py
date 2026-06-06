"""
文本分割器模块
使用 RecursiveCharacterTextSplitter 对文档进行语义感知的切割
"""
import logging
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.documents import Document
from backend.config.settings import get_settings

logger = logging.getLogger("ai_rd_agent")


class TextSplitter:
    """递归字符文本分割器

    特点：
    - 按自然分隔符（换行、句号、空格）逐级尝试切割
    - 保持语义完整性，尽量不在句子中间切割
    - 通过 chunk_overlap 保持上下文连续性

    使用示例：
        splitter = TextSplitter()
        chunks = splitter.split(documents)
    """

    def __init__(self, chunk_size: int | None = None, chunk_overlap: int | None = None):
        """
        Args:
            chunk_size: 每个文本块的最大字符数（默认从配置读取）
            chunk_overlap: 相邻块之间的重叠字符数（默认从配置读取）
        """
        settings = get_settings()
        self.chunk_size = chunk_size or settings.CHUNK_SIZE
        self.chunk_overlap = chunk_overlap or settings.CHUNK_OVERLAP

        # 创建 LangChain 内置的分割器
        self._splitter = RecursiveCharacterTextSplitter(
            chunk_size=self.chunk_size,
            chunk_overlap=self.chunk_overlap,
            # 按优先级从高到低的分隔符列表
            separators=[
                "\n\n",    # 空行（段落边界）
                "\n",      # 换行
                "。",      # 中文句号
                "；",      # 中文分号
                ". ",      # 英文句号
                "! ",      # 英文感叹号
                "? ",      # 英文问号
                " ",       # 空格
                "",        # 字符级切割（最后手段）
            ],
            length_function=len,  # 使用字符数衡量长度
            add_start_index=True, # 记录每个块在原文档中的起始位置
        )

        logger.info(
            f"文本分割器已初始化 | "
            f"chunk_size={self.chunk_size}, "
            f"chunk_overlap={self.chunk_overlap}"
        )

    def split(self, documents: list[Document]) -> list[Document]:
        """对文档列表进行切割

        Args:
            documents: 待切割的 LangChain Document 列表

        Returns:
            切割后的 Document 列表（每个块继承原文档的 metadata）
        """
        if not documents:
            return []

        total_chars = sum(len(doc.page_content) for doc in documents)
        chunks = self._splitter.split_documents(documents)

        logger.info(
            f"文本切割完成: {len(documents)} 个文档 → "
            f"{len(chunks)} 个块 "
            f"(原始 {total_chars} 字符)"
        )

        return chunks

    def split_text(self, text: str) -> list[str]:
        """对纯文本字符串进行切割（不包含 metadata）

        Args:
            text: 待切割的文本字符串

        Returns:
            切割后的文本片段列表
        """
        if not text:
            return []
        return self._splitter.split_text(text)
