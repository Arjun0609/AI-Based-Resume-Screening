from PIL import Image
import io
import logging

logger = logging.getLogger(__name__)


class TextExtractor:
    def __init__(self, ocr_enabled=False, min_confidence=60):
        self.ocr_enabled = ocr_enabled
        self.min_confidence = min_confidence
        logger.info(f"Initializing TextExtractor (OCR: {ocr_enabled})")

    def extract_text(self, document):
        doc_type = document["metadata"]["file_type"]

        if "text_content" in document and document["text_content"]:
            logger.info("Text content already available, skipping extraction")
            return document["text_content"]

        if doc_type == "pdf":
            return self._extract_from_pdf(document)
        else:
            raise ValueError(f"Text extraction not implemented for {doc_type}")

    def _extract_from_pdf(self, document):
        doc_obj = document["doc_obj"]
        text = ""

        for page_num in range(len(doc_obj)):
            page = doc_obj[page_num]

            page_text = page.get_text()

            if not page_text.strip() and self.ocr_enabled:
                page_text = self._ocr_page(page)

            text += page_text + "\n\n"

        return text.strip()

    def _ocr_page(self, page):
        import pytesseract
        
        try:
            pix = page.get_pixmap(dpi=300)
            img = Image.open(io.BytesIO(pix.tobytes("png")))

            ocr_data = pytesseract.image_to_data(
                img, output_type=pytesseract.Output.DICT
            )

            visible_text = []
            for i in range(len(ocr_data["text"])):
                if (
                    int(ocr_data["conf"][i]) >= self.min_confidence
                    and ocr_data["text"][i].strip()
                ):
                    visible_text.append(ocr_data["text"][i])

            return " ".join(visible_text)
        except Exception as e:
            logger.error(f"OCR error: {str(e)}")
            return ""
