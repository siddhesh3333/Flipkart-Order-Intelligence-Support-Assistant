"""
Part 3 - Policy Sentence-Level Chunker

Reads the policy documents from:

    Part-3/data/policies/

and creates sentence-level chunks with stable parent-document
metadata for document-level retrieval evaluation.
"""

from pathlib import Path
import json
import re


# ============================================================
# PATHS
# ============================================================

ROOT_DIR = Path(__file__).resolve().parents[2]

POLICY_DIR = ROOT_DIR / "Part-3" / "data" / "policies"

OUTPUT_DIR = ROOT_DIR / "Part-3" / "data" / "processed"

OUTPUT_FILE = OUTPUT_DIR / "policy_chunks.jsonl"


# ============================================================
# SENTENCE SPLITTING
# ============================================================

def split_sentences(text: str) -> list[str]:
    """
    Split policy text into sentence-level units.

    Markdown headings and blank lines are ignored.
    """

    lines = []

    for line in text.splitlines():

        line = line.strip()

        if not line:
            continue

        # Skip markdown headings.
        if line.startswith("#"):
            continue

        lines.append(line)

    clean_text = " ".join(lines)

    # Sentence boundary:
    # ., !, or ? followed by whitespace/end of text.
    sentences = re.split(
        r"(?<=[.!?])\s+",
        clean_text
    )

    sentences = [
        sentence.strip()
        for sentence in sentences
        if sentence.strip()
    ]

    return sentences


# ============================================================
# DOCUMENT ID
# ============================================================

def extract_document_id(text: str) -> str:
    """
    Extract the stable document ID from:

        # Document ID: POL-001
    """

    match = re.search(
        r"#\s*Document ID:\s*([A-Z]+-\d+)",
        text
    )

    if not match:
        raise ValueError(
            "Document ID not found in policy document."
        )

    return match.group(1)


# ============================================================
# CHUNK CREATION
# ============================================================

def build_chunks() -> list[dict]:

    if not POLICY_DIR.exists():
        raise FileNotFoundError(
            f"Policy directory not found:\n{POLICY_DIR}"
        )

    policy_files = sorted(
        POLICY_DIR.glob("*.md")
    )

    if not policy_files:
        raise FileNotFoundError(
            f"No policy documents found in:\n{POLICY_DIR}"
        )

    chunks = []

    for policy_file in policy_files:

        text = policy_file.read_text(
            encoding="utf-8"
        )

        document_id = extract_document_id(
            text
        )

        sentences = split_sentences(
            text
        )

        for sentence_number, sentence in enumerate(
            sentences,
            start=1
        ):

            chunk = {
                "chunk_id": (
                    f"{document_id}-S{sentence_number:03d}"
                ),
                "document_id": document_id,
                "source_file": policy_file.name,
                "text": sentence,
            }

            chunks.append(chunk)

    return chunks


# ============================================================
# SAVE JSONL
# ============================================================

def save_chunks(chunks: list[dict]) -> None:

    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True
    )

    with OUTPUT_FILE.open(
        "w",
        encoding="utf-8"
    ) as file:

        for chunk in chunks:

            file.write(
                json.dumps(
                    chunk,
                    ensure_ascii=False
                )
                + "\n"
            )


# ============================================================
# MAIN
# ============================================================

def main():

    print("=" * 70)
    print("PART 3 - POLICY SENTENCE CHUNKER")
    print("=" * 70)

    print("\nPolicy directory:")
    print(POLICY_DIR)

    chunks = build_chunks()

    save_chunks(chunks)

    document_ids = sorted(
        {
            chunk["document_id"]
            for chunk in chunks
        }
    )

    print("\nDocuments processed:")
    print(len(document_ids))

    print("\nDocument IDs:")
    for document_id in document_ids:
        print(" -", document_id)

    print("\nTotal sentence chunks:")
    print(len(chunks))

    print("\nOutput:")
    print(OUTPUT_FILE)

    print("\nFirst 5 chunks:")
    print("-" * 70)

    for chunk in chunks[:5]:

        print(
            json.dumps(
                chunk,
                ensure_ascii=False
            )
        )

    print("\n" + "=" * 70)
    print("POLICY CHUNKING: PASSED")
    print("=" * 70)


if __name__ == "__main__":
    main()