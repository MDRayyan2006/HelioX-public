import os
import fitz  # PyMuPDF


def extract_pdf(pdf_path):
    doc = fitz.open(pdf_path)

    pages = []
    full_text = []

    for i, page in enumerate(doc):
        text = page.get_text("text")

        if text.strip():  # skip empty pages
            pages.append({
                "page_num": i + 1,
                "text": text
            })
            full_text.append(text)

    metadata = {
        "file_name": os.path.basename(pdf_path),
        "file_path": pdf_path,
        "num_pages": len(pages),
        "file_size": os.path.getsize(pdf_path),
        "source": "pdf"
    }

    doc.close()

    return {
        "text": "\n".join(full_text),
        "pages": pages,
        "metadata": metadata
    }


def load_folder(folder_path):
    documents = []

    for file in os.listdir(folder_path):
        if file.lower().endswith(".pdf"):
            path = os.path.join(folder_path, file)

            try:
                doc = extract_pdf(path)
                documents.append(doc)

            except Exception as e:
                print(f"❌ Failed: {file} → {e}")

    return documents


if __name__ == "__main__":
    folder_path = os.path.join(os.path.dirname(__file__), "../uploads")
    docs = load_folder(folder_path)

    print(f"✅ Loaded {len(docs)} documents")
    for doc in docs:
        print(f"\n--- {doc['metadata']['file_name']} ---")
        print(f"Pages: {doc['metadata']['num_pages']}, Size: {doc['metadata']['file_size']} bytes")
        print(f"Sample Text:\n{doc['text'][:500]}...")