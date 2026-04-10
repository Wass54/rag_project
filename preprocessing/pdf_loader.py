import fitz
import re
from pathlib import Path


def extract_text_from_pdf(pdf_path: str) -> str:
    """
    Extract raw text from a PDF file, page by page.

    Each page is prefixed with its page number so we can later
    trace back which page a chunk came from.

    """
    doc = fitz.open(pdf_path)
    pages_text = []

    for page_num, page in enumerate(doc):
        text = page.get_text("text")
        text = clean_text(text)

        if text.strip():
            pages_text.append(f"[Page {page_num + 1}]\n{text}")

    doc.close()

    return "\n\n".join(pages_text)


def clean_text(text: str) -> str:
    """
    Basic text cleaning to remove PDF extraction artifacts.

    """
    lines = text.split("\n")

    # Discard lines that are too short to carry meaningful content
    # (page numbers, stray characters, artefacts from tables, etc.)
    lines = [line for line in lines if len(line.strip()) > 2]

    text = "\n".join(lines)

    # Collapse multiple consecutive spaces into one
    text = re.sub(r" {2,}", " ", text)

    # Collapse more than 2 consecutive newlines into exactly 2
    # (preserves paragraph structure without excessive blank space)
    text = re.sub(r"\n{3,}", "\n\n", text)

    return text.strip()


def load_pdfs_from_folder(folder_path: str) -> list[dict]:
    """
    Load all PDF files found in a given folder.
    """
    folder = Path(folder_path)

    if not folder.exists():
        raise FileNotFoundError(f"Folder not found: {folder_path}")

    pdf_files = list(folder.glob("*.pdf"))

    if not pdf_files:
        print(f"No PDF files found in {folder_path}")
        return []

    documents = []

    for pdf_file in pdf_files:
        print(f"Loading: {pdf_file.name}")
        text = extract_text_from_pdf(str(pdf_file))

        documents.append({
            "filename": pdf_file.name,
            "source": str(pdf_file),
            "text": text,
            "num_chars": len(text)
        })

    print(f"\n{len(documents)} document(s) loaded successfully.")
    for doc in documents:
        print(f"  - {doc['filename']}: {doc['num_chars']:,} characters")

    return documents


# ---------------------------------------------------------------------------
# Quick test — run directly with: python preprocessing/pdf_loader.py <path>
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    import sys

    if len(sys.argv) < 2:
        print("Usage: python pdf_loader.py <path_to_pdf>")
        sys.exit(1)

    path = sys.argv[1]
    print(f"Extracting text from: {path}\n")

    extracted = extract_text_from_pdf(path)

    # Print first 2000 characters as a preview
    print(extracted[:2000])
    print(f"\n--- Total: {len(extracted):,} characters extracted ---")