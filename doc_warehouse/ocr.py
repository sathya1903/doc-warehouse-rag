import pdfplumber
import pytesseract
from pdf2image import convert_from_path


def extract_text_from_pdf(pdf_path: str) -> str:
    text = ""

    # Try text-based extraction
    with pdfplumber.open(pdf_path) as pdf:
        for page in pdf.pages:
            page_text = page.extract_text()
            if page_text:
                text += page_text + "\n"

    # If no text found → OCR
    if not text.strip():
        images = convert_from_path(pdf_path)
        for img in images:
            text += pytesseract.image_to_string(img)

    return text
