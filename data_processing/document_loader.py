import os
import fitz
from PIL import Image
import io
import docx2txt
import logging

logger = logging.getLogger(__name__)


class DocumentLoader:
    SUPPORTED_FORMATS = [".pdf", ".docx", ".txt"]

    def __init__(self):
        logger.info("Initializing DocumentLoader")

    def load_document(self, file_path):
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"File not found: {file_path}")

        file_ext = os.path.splitext(file_path)[1].lower()

        if file_ext not in self.SUPPORTED_FORMATS:
            raise ValueError(f"Unsupported file format: {file_ext}")

        logger.info(f"Loading document: {file_path}")

        if file_ext == ".pdf":
            return self._load_pdf(file_path)
        elif file_ext == ".docx":
            return self._load_docx(file_path)
        elif file_ext == ".txt":
            return self._load_txt(file_path)

    def _load_pdf(self, file_path):
        try:
            doc = fitz.open(file_path)
            metadata = {
                "file_path": file_path,
                "file_type": "pdf",
                "page_count": len(doc),
                "title": doc.metadata.get("title", ""),
                "author": doc.metadata.get("author", ""),
                "creation_date": doc.metadata.get("creationDate", ""),
            }

            return {"metadata": metadata, "doc_obj": doc, "images": None}
        except Exception as e:
            logger.error(f"Error loading PDF {file_path}: {str(e)}")
            raise

    def _load_docx(self, file_path):
        try:
            text = docx2txt.process(file_path)
            metadata = {
                "file_path": file_path,
                "file_type": "docx",
                "page_count": None,
                "title": os.path.basename(file_path),
                "author": "",
                "creation_date": "",
            }

            return {"metadata": metadata, "text_content": text, "doc_obj": None}
        except Exception as e:
            logger.error(f"Error loading DOCX {file_path}: {str(e)}")
            raise

    def _load_txt(self, file_path):
        try:
            with open(file_path, "r", encoding="utf-8") as file:
                text = file.read()

            metadata = {
                "file_path": file_path,
                "file_type": "txt",
                "page_count": 1,
                "title": os.path.basename(file_path),
                "author": "",
                "creation_date": "",
            }

            return {"metadata": metadata, "text_content": text, "doc_obj": None}
        except Exception as e:
            logger.error(f"Error loading TXT {file_path}: {str(e)}")
            raise

    def batch_load(self, directory_path, file_types=None):
        if file_types is None:
            file_types = self.SUPPORTED_FORMATS

        loaded_docs = []

        for filename in os.listdir(directory_path):
            file_path = os.path.join(directory_path, filename)
            if os.path.isfile(file_path):
                file_ext = os.path.splitext(file_path)[1].lower()
                if file_ext in file_types:
                    try:
                        doc = self.load_document(file_path)
                        loaded_docs.append(doc)
                    except Exception as e:
                        logger.warning(f"Failed to load {file_path}: {str(e)}")

        logger.info(f"Loaded {len(loaded_docs)} documents from {directory_path}")
        return loaded_docs
