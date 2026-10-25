"""Validated inference for untrusted uploads. Research use only."""
import io, json, warnings
from pathlib import Path
import numpy as np, torch, torch.nn as nn
from PIL import Image
from torchvision.models import resnet18
from safetensors.torch import load_file

HERE = Path(__file__).resolve().parent
IMG, MEAN, STD = 128, (0.485, 0.456, 0.406), (0.229, 0.224, 0.225)
MAX_BYTES, MIN_SIDE, MAX_SIDE = 5*1024*1024, 32, 4096
ALLOWED, THRESHOLD, REFER_BAND = {"PNG", "JPEG"}, 0.5, (0.20, 0.80)

class UploadError(Exception):
    pass

def load_upload(raw: bytes) -> Image.Image:
    if not raw:
        raise UploadError("empty upload")
    if len(raw) > MAX_BYTES:
        raise UploadError("file too large")
    old = Image.MAX_IMAGE_PIXELS
    Image.MAX_IMAGE_PIXELS = MAX_SIDE * MAX_SIDE
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("error", Image.DecompressionBombWarning)
            with Image.open(io.BytesIO(raw)) as im:
                if im.format not in ALLOWED:
                    raise UploadError(f"unsupported format: {im.format}")
                w, h = im.size
                if not (MIN_SIDE <= w <= MAX_SIDE and MIN_SIDE <= h <= MAX_SIDE):
                    raise UploadError(f"size {w}x{h} out of range")
                im.verify()
            with Image.open(io.BytesIO(raw)) as im:
                rgb = im.convert("RGB")
            return Image.frombytes("RGB", rgb.size, rgb.tobytes())
    except UploadError:
        raise
    except Exception as e:
        raise UploadError(f"could not decode ({type(e).__name__})") from None
    finally:
        Image.MAX_IMAGE_PIXELS = old

def load_gate(path=None):
    return json.loads(Path(path or HERE / "model" / "gate.json").read_text())

def load_model(weights=None, device=None):
    weights = Path(weights or HERE / "model" / "resnet18_malaria.safetensors")
    device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
    m = resnet18(weights=None)
    m.fc = nn.Linear(m.fc.in_features, 1)
    m.load_state_dict(load_file(str(weights), device="cpu"))
    return m.to(device).eval()

@torch.no_grad()
def predict_bytes(raw: bytes, model, gate=None) -> dict:
    gate = gate or load_gate()
    try:
        im = load_upload(raw)
    except UploadError as e:
        return {"status": "rejected", "reason": str(e)}
    a = np.asarray(im.resize((IMG, IMG), Image.BILINEAR), dtype=np.uint8)
    bg = float((a.max(axis=2) <= 10).mean())
    if not (gate["bg_lo"] <= bg <= gate["bg_hi"]):
        return {"status": "withheld", "background_fraction": round(bg, 3),
                "reason": f"bg fraction {bg:.3f} outside training range"}
    x = torch.from_numpy(a.copy()).permute(2, 0, 1).float().div(255)
    x = (x - torch.tensor(MEAN).view(3, 1, 1)) / torch.tensor(STD).view(3, 1, 1)
    p = float(torch.sigmoid(model(x.unsqueeze(0).to(next(model.parameters()).device)).squeeze(1)))
    return {"status": "ok", "p_parasitized": round(p, 4),
            "label": "parasitized" if p >= THRESHOLD else "uninfected",
            "refer_to_human": REFER_BAND[0] < p < REFER_BAND[1],
            "background_fraction": round(bg, 3)}
