"""Download a checksum-verified, 100d GloVe 6B archive for coursework.

This is Liebl's independently hosted *binary word2vec conversion* of the
Stanford GloVe 6B 100-dimensional vectors, not Stanford's original 822 MB ZIP.
Source: https://doi.org/10.5281/zenodo.4925376
Original vectors: https://nlp.stanford.edu/projects/glove/
The downstream experiment records the source and conversion explicitly.
No pretrained vectors are committed into the Git repository.
"""
import argparse
import hashlib
import os
from pathlib import Path
import tempfile
from urllib.request import Request, urlopen

URL = "https://zenodo.org/records/4925376/files/glove.6B.100d.zip?download=1"
EXPECTED_MD5 = "f19871e3053750198004fb2acc1f8d44"


def download_verified_glove(destination, overwrite=False):
    destination = Path(destination)
    if destination.exists() and not overwrite:
        current = hashlib.md5(destination.read_bytes()).hexdigest()
        if current == EXPECTED_MD5:
            print("Using existing checksum-verified GloVe:", destination)
            return destination
        raise ValueError("Existing GloVe archive has the wrong checksum. Refusing to reuse.")
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="wb", dir=destination.parent, prefix=".glove_",
            suffix=".part", delete=False
        ) as output:
            temporary = Path(output.name)
            checksum = hashlib.md5()
            request = Request(URL, headers={"User-Agent": "Academic-reproducibility/1.0"})
            with urlopen(request, timeout=180) as response:
                while True:
                    part = response.read(1024 * 1024)
                    if not part:
                        break
                    output.write(part)
                    checksum.update(part)
        if checksum.hexdigest() != EXPECTED_MD5:
            raise ValueError(
                f"GloVe download checksum mismatch ({checksum.hexdigest()}); "
                "not installing downloaded data."
            )
        os.replace(temporary, destination)
        temporary = None
        print(f"Verified GloVe zip: {destination} (md5={EXPECTED_MD5})")
        return destination
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--output", type=Path, default=Path("data/glove.6B.100d.zip")
    )
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()
    download_verified_glove(args.output, overwrite=args.overwrite)
