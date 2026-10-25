"""Self-test: python test_inference.py"""
import io, os, sys
import numpy as np
from PIL import Image
import inference as inf

FAILS = []

def check(name, ok, detail=""):
    print(("PASS  " if ok else "FAIL  ") + name + (f"  [{detail}]" if detail else ""))
    if not ok:
        FAILS.append(name)

def enc(a, fmt="PNG"):
    b = io.BytesIO()
    Image.fromarray(a).save(b, format=fmt)
    return b.getvalue()

def main():
    model, gate = inf.load_model(device="cpu"), inf.load_gate()
    yy, xx = np.mgrid[:128, :128]
    cell = np.zeros((128, 128, 3), np.uint8)
    cell[(yy - 64) ** 2 + (xx - 64) ** 2 < 60 ** 2] = (200, 140, 160)
    good = enc(cell)

    r = inf.predict_bytes(good, model, gate)
    check("accepts cell-like PNG", r["status"] == "ok")
    check("accepts JPEG", inf.predict_bytes(enc(cell, "JPEG"), model, gate)["status"] == "ok")
    check("deterministic", inf.predict_bytes(good, model, gate) == r)

    cases = {
        "empty": b"",
        "random": os.urandom(2000),
        "text": b"hello" * 100,
        "truncated": good[:len(good) // 2],
        "too big": good + b"\0" * (6 * 1024 * 1024),
        "tiny": enc(np.zeros((16, 16, 3), np.uint8)),
    }
    for name, raw in cases.items():
        check(f"rejects {name}", inf.predict_bytes(raw, model, gate)["status"] == "rejected")

    black = enc(np.zeros((128, 128, 3), np.uint8))
    check("withholds all-black", inf.predict_bytes(black, model, gate)["status"] == "withheld")

    print(f"\n{len(FAILS)} failure(s)")
    sys.exit(1 if FAILS else 0)

if __name__ == "__main__":
    main()
