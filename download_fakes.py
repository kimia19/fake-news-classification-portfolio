"""Download the officially published FA-KES dataset with integrity verification.

Citation:
Abu Salem, F. K., Al Feel, R., Elbassuoni, S., Jaber, M., & Farah, M.
(2019). FA-KES: A Fake News Dataset around the Syrian War.
Zenodo, https://doi.org/10.5281/zenodo.2607278
License: CC BY 4.0, attribution required.

Source: https://zenodo.org/records/2607278
This script does not upload or commit the downloaded data.
"""
import argparse
import hashlib
import os
from pathlib import Path
import tempfile
from urllib.request import Request, urlopen

URL = "https://zenodo.org/records/2607278/files/FA-KES-Dataset.csv?download=1"
EXPECTED_MD5 = "55d0ec452440a4ed46aac1e330b40e83"


def download_verified_dataset(destination: Path, overwrite: bool = False) -> Path:
    destination = Path(destination)
    if destination.exists() and not overwrite:
        raise FileExistsError(
            f"{destination} already exists; use --overwrite explicitly to replace"
        )
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="wb", dir=str(destination.parent), prefix=".fakes_",
            suffix=".part", delete=False
        ) as file:
            temporary = Path(file.name)
            checksum = hashlib.md5()  # published upstream checksum, not security authentication
            req = Request(URL, headers={"User-Agent": "Academic-reproducibility/1.0"})
            with urlopen(req, timeout=45) as response:
                while True:
                    chunk = response.read(1024 * 1024)
                    if not chunk:
                        break
                    file.write(chunk)
                    checksum.update(chunk)
        received_md5 = checksum.hexdigest()
        if received_md5 != EXPECTED_MD5:
            raise ValueError(
                f"Zenodo checksum mismatch: received {received_md5}, "
                f"expected {EXPECTED_MD5}. No file installed."
            )
        os.replace(temporary, destination)
        temporary = None
        return destination
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output", type=Path, default=Path("data/FA-KES-Dataset.csv")
    )
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()
    saved = download_verified_dataset(args.output, overwrite=args.overwrite)
    print(f"Verified and saved dataset to: {saved}")
