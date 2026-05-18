def chunk_text(text: str, chunk_size: int = 300) -> list[str]:
    chunks = []

    current_chunk = ""

    sentences = text.split(".")

    for sentence in sentences:
        sentence = sentence.strip()

        if not sentence:
            continue

        candidate = current_chunk + sentence + ". "

        if len(candidate) <= chunk_size:
            current_chunk = candidate
        else:
            chunks.append(current_chunk.strip())
            current_chunk = sentence + ". "

    if current_chunk:
        chunks.append(current_chunk.strip())

    return chunks
