import pdfplumber
import os
import json
from tqdm import tqdm


def extract_text_from_pdf(pdf_path: str, is_constitution: bool = False) -> str:
    """Extract text from a PDF, applying special cropping for the Constitution."""
    text_pages = []

    with pdfplumber.open(pdf_path) as pdf:
        print(f"  Pages: {len(pdf.pages)}")

        for page_num, page in enumerate(pdf.pages):
            try:
                # Apply the Diglot & Footnote cropping ONLY for the Constitution
                if is_constitution:
                    width = page.width
                    height = page.height
                    # Crops right half, cuts top 50px, cuts bottom 120px
                    bounding_box = (width * 0.5, 50, width, height - 120)
                    try:
                        page = page.crop(bounding_box)
                    except ValueError:
                        # Failsafe for pages with weird dimensions
                        pass

                text = page.extract_text()
                if text and len(text.strip()) > 50:
                    text_pages.append(f"[Page {page_num + 1}]\n{text}")
            except Exception as e:
                print(f"  Error on page {page_num}: {e}")
                continue

    return "\n\n".join(text_pages)


def extract_all_pdfs(acts_dir: str = "data/raw/acts", output_dir: str = "data/cleaned/acts"):
    """Extract text from all downloaded PDFs and save as JSON."""
    os.makedirs(output_dir, exist_ok=True)

    pdf_files = [f for f in os.listdir(acts_dir) if f.endswith(".pdf")]
    print(f"Found {len(pdf_files)} PDFs to extract")

    for pdf_file in tqdm(pdf_files):
        pdf_path = os.path.join(acts_dir, pdf_file)
        output_file = os.path.join(
            output_dir, pdf_file.replace(".pdf", ".json"))

        if os.path.exists(output_file):
            print(f"  Already extracted: {pdf_file}")
            continue

        print(f"\nExtracting: {pdf_file}")

        # Check if this specific file is the constitution to trigger cropping
        is_const = "constitution" in pdf_file.lower()
        text = extract_text_from_pdf(pdf_path, is_constitution=is_const)

        if len(text) < 100:
            print(f"  WARNING: Very little text extracted — may be scanned PDF")

        data = {
            "filename": pdf_file,
            "text": text,
            "source": "legislative.gov.in",
            "doc_type": "constitution" if is_const else "act"
        }

        with open(output_file, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=2)

    print(f"\nExtraction complete. Files saved to {output_dir}")


if __name__ == "__main__":
    # This block ensures the function actually runs when you execute the file
    extract_all_pdfs()
