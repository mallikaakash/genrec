"""Download a trained GenRec checkpoint from the Modal volume.

The `modal volume get` CLI trips over the nested backbone/ directory
("[Errno 21] Is a directory"), so we walk the volume with the Python API and
write files ourselves.

    python scripts/fetch_model.py                      # -> ./genrec_model_v2
    python scripts/fetch_model.py --remote v2/model --local ./genrec_model_v2
"""
import argparse
import pathlib

import modal
from modal.volume import FileEntryType


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--volume", default="genrec-out")
    ap.add_argument("--remote", default="v2/model")
    ap.add_argument("--local", default="./genrec_model_v2")
    args = ap.parse_args()

    vol = modal.Volume.from_name(args.volume)
    dest = pathlib.Path(args.local)
    dest.mkdir(parents=True, exist_ok=True)

    n = 0
    for entry in vol.iterdir(args.remote, recursive=True):
        rel = entry.path[len(args.remote):].lstrip("/")
        if not rel:
            continue
        target = dest / rel
        if entry.type != FileEntryType.FILE:
            target.mkdir(parents=True, exist_ok=True)
            continue
        target.parent.mkdir(parents=True, exist_ok=True)
        with open(target, "wb") as f:
            for chunk in vol.read_file(entry.path):
                f.write(chunk)
        n += 1
        print(f"  {rel}  ({target.stat().st_size/1e6:.1f} MB)")
    print(f"\n[fetch] wrote {n} files to {dest}")


if __name__ == "__main__":
    main()
