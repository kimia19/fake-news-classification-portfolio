"""Stream a binary word2vec-converted GloVe 6B 100d ZIP into a small Keras matrix.

This reads only needed vectors into RAM rather than storing 400K vectors.
The archive is verified separately by scripts/download_glove_100d.py.

Word2vec binary record format:
    <utf8 token><space><100 float32 little-endian values> (no delimiter)
"""
import hashlib
import zipfile
from pathlib import Path
import numpy as np

from download_glove_100d import EXPECTED_MD5


def read_exact(stream, size):
    parts = bytearray()
    while len(parts) < size:
        part = stream.read(size - len(parts))
        if not part:
            raise EOFError("Truncated binary GloVe stream")
        parts.extend(part)
    return bytes(parts)


def read_word(stream, max_bytes=512):
    word = bytearray()
    for _ in range(max_bytes):
        character = stream.read(1)
        if character == b" ":
            if not word:
                raise ValueError("Empty word in GloVe binary vector")
            return word.decode("utf-8")
        if not character or character == b"\n":
            raise ValueError("Truncated or malformed GloVe word entry")
        word.extend(character)
    raise ValueError("Word token exceeds maximum length")


def build_glove_matrix(
    archive_path, word_index, vocab_size=5000, embedding_dim=100,
    check_checksum=True, expected_count=400000
):
    archive_path = Path(archive_path)
    if check_checksum:
        checksum = hashlib.md5(archive_path.read_bytes()).hexdigest()
        if checksum != EXPECTED_MD5:
            raise ValueError("GloVe binary ZIP checksum is different from Zenodo's")
    if vocab_size < 2 or embedding_dim < 1:
        raise ValueError("Invalid embedding configuration")

    wanted = {word: int(idx) for word, idx in word_index.items()
              if 0 < int(idx) < vocab_size}
    matrix = np.zeros((vocab_size, embedding_dim), dtype=np.float32)
    found = set()
    with zipfile.ZipFile(archive_path) as zf:
        members = [info for info in zf.infolist() if not info.is_dir()]
        # This exact release contains a single .bin file. Do not follow
        # arbitrary archive paths or extract any files to disk.
        candidates = [x for x in members if x.filename.lower().endswith(".bin")]
        if len(candidates) != 1:
            if len(members) == 1:
                candidates = members
            else:
                raise ValueError(f"Ambiguous GloVe archive members: {[m.filename for m in members]}")
        member = candidates[0]
        if member.file_size > 600_000_000:
            raise ValueError("Unreasonable expanded GloVe size")
        with zf.open(member, "r") as stream:
            header = stream.readline(128).decode("ascii").strip().split()
            if len(header) != 2:
                raise ValueError(f"Unexpected binary word2vec header: {header}")
            total_words, dim = map(int, header)
            if dim != embedding_dim:
                raise ValueError(f"Expected {embedding_dim}d, found {dim}d")
            if expected_count is not None and total_words != expected_count:
                raise ValueError(f"Expected {expected_count} words, found {total_words}")
            if total_words > 1_000_000:
                raise ValueError("Unexpectedly large vocabulary")
            for _ in range(total_words):
                token = read_word(stream)
                vector = np.frombuffer(
                    read_exact(stream, 4 * dim), dtype="<f4"
                )
                if token in wanted:
                    matrix[wanted[token]] = vector
                    found.add(token)
                # The binary word2vec format has no separator after the
                # packed floats; the next word begins at the next byte.

    hits = len(found)
    misses = len(wanted) - hits
    info = {
        "source": "https://doi.org/10.5281/zenodo.4925376",
        "format": "binary word2vec conversion of Stanford GloVe 6B 100d",
        "zip_md5": EXPECTED_MD5 if check_checksum else "not checked (test only)",
        "vectors_in_source": int(total_words),
        "embedding_dimensions": embedding_dim,
        "train_vocabulary_capped": len(wanted),
        "matched_tokens": hits,
        "unmatched_tokens": misses,
        "coverage": float(hits / len(wanted)) if wanted else 0.0,
        "padding_and_oov": "unmatched and padded rows remain zero; no pretrained OOV vector",
    }
    return matrix, info
