import fitz  # PyMuPDF
import os

def extract_text_from_pdf(pdf_path):
    doc = fitz.open(pdf_path)
    text = ""

    for page in doc:
        text += page.get_text()

    return text

if __name__ == "__main__":
    pdf_path = os.path.join(os.path.dirname(__file__), "../uploads/resume.pdf")
    extracted_text = extract_text_from_pdf(pdf_path)
    print(extracted_text)