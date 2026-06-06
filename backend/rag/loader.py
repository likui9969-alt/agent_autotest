"""
文档加载器模块
支持加载 txt、pdf、docx 格式的文档，统一输出为 Document 列表
"""
import logging
from pathlib import Path
from langchain_core.documents import Document

logger = logging.getLogger("ai_rd_agent")


# 支持的文件格式及其对应的加载器
SUPPORTED_EXTENSIONS = {
    ".txt": "text",
    ".pdf": "pdf",
    ".docx": "docx",
}


class DocumentLoader:
    """统一文档加载器 — 根据文件扩展名自动选择解析方式

    使用示例：
        loader = DocumentLoader()
        docs = loader.load("data/docs/test_report.txt")
        docs = loader.load_directory("data/docs/")
    """

    # 允许加载的最大文件大小（10MB），防止内存溢出
    MAX_FILE_SIZE = 10 * 1024 * 1024

    def load(self, file_path: str) -> list[Document]:
        """加载单个文档文件

        Args:
            file_path: 文档文件的绝对路径

        Returns:
            LangChain Document 列表（每个 Document 包含 page_content 和 metadata）
        """
        path = Path(file_path)

        # 检查文件是否存在
        if not path.exists():
            raise FileNotFoundError(f"文档不存在: {file_path}")

        # 检查文件大小
        file_size = path.stat().st_size
        if file_size > self.MAX_FILE_SIZE:
            raise ValueError(
                f"文件过大: {self._format_size(file_size)}，"
                f"最大允许 {self._format_size(self.MAX_FILE_SIZE)}"
            )

        # 根据扩展名选择加载方式
        ext = path.suffix.lower()
        if ext not in SUPPORTED_EXTENSIONS:
            supported = ", ".join(SUPPORTED_EXTENSIONS.keys())
            raise ValueError(f"不支持的文件格式 '{ext}'。支持的格式: {supported}")

        logger.info(f"加载文档: {path.name} ({self._format_size(file_size)})")

        if ext == ".txt":
            return self._load_text(path)
        elif ext == ".pdf":
            return self._load_pdf(path)
        elif ext == ".docx":
            return self._load_docx(path)
        else:
            return []

    def load_directory(self, dir_path: str) -> list[Document]:
        """加载目录下所有支持的文档

        Args:
            dir_path: 文档目录路径

        Returns:
            所有文档的 Document 列表
        """
        all_docs = []
        dir_path = Path(dir_path)

        if not dir_path.exists():
            logger.warning(f"文档目录不存在: {dir_path}")
            return all_docs

        for ext in SUPPORTED_EXTENSIONS:
            for file_path in dir_path.glob(f"*{ext}"):
                try:
                    docs = self.load(str(file_path))
                    all_docs.extend(docs)
                except Exception as e:
                    logger.error(f"加载失败 {file_path.name}: {e}")

        logger.info(f"目录加载完成: {len(all_docs)} 个文档片段")
        return all_docs

    # ---- 各格式加载器 ----

    def _load_text(self, path: Path) -> list[Document]:
        """加载纯文本文件"""
        with open(path, "r", encoding="utf-8", errors="replace") as f:
            content = f.read()

        return [Document(
            page_content=content,
            metadata={
                "source": str(path),
                "filename": path.name,
                "file_type": "txt",
            }
        )]

    def _load_pdf(self, path: Path) -> list[Document]:
        """加载 PDF 文件 — 逐页提取文本"""
        try:
            from pypdf import PdfReader
        except ImportError:
            raise ImportError("请安装 pypdf: pip install pypdf")

        reader = PdfReader(str(path))
        docs = []

        for i, page in enumerate(reader.pages):
            text = page.extract_text()
            if text and text.strip():
                docs.append(Document(
                    page_content=text.strip(),
                    metadata={
                        "source": str(path),
                        "filename": path.name,
                        "file_type": "pdf",
                        "page": i + 1,
                    }
                ))

        logger.info(f"  PDF 解析完成: {len(docs)} 页")
        return docs

    def _load_docx(self, path: Path) -> list[Document]:
        """加载 Word 文档"""
        try:
            from docx import Document as DocxDocument
        except ImportError:
            raise ImportError("请安装 python-docx: pip install python-docx")

        doc = DocxDocument(str(path))

        # 提取所有段落文本
        paragraphs = []
        for para in doc.paragraphs:
            if para.text.strip():
                paragraphs.append(para.text)

        # 也提取表格中的文本
        for table in doc.tables:
            for row in table.rows:
                row_text = " | ".join(
                    cell.text for cell in row.cells if cell.text.strip()
                )
                if row_text.strip():
                    paragraphs.append(row_text)

        full_text = "\n".join(paragraphs)

        return [Document(
            page_content=full_text,
            metadata={
                "source": str(path),
                "filename": path.name,
                "file_type": "docx",
            }
        )]

    @staticmethod
    def _format_size(size_bytes: int) -> str:
        """将字节数格式化为人类可读的大小"""
        for unit in ["B", "KB", "MB"]:
            if size_bytes < 1024:
                return f"{size_bytes:.1f} {unit}"
            size_bytes /= 1024
        return f"{size_bytes:.1f} GB"
