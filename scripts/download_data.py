"""Download the CLINC150 dataset (Larson et al., 2019) into data/."""
import urllib.request
from pathlib import Path

BASE = "https://raw.githubusercontent.com/clinc/oos-eval/master/data/"
FILES = ["data_full.json", "domains.json"]


def main() -> None:
    out = Path("data")
    out.mkdir(exist_ok=True)
    for name in FILES:
        target = out / name
        if target.exists():
            print(f"{target} already exists")
            continue
        urllib.request.urlretrieve(BASE + name, target)
        print(f"saved {target}")


if __name__ == "__main__":
    main()
