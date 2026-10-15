from pathlib import Path
import hashlib
import tempfile
import sys

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "src"))
from analysis import run_analysis

shipments = ROOT / "data" / "shipment_records.csv"
notes = ROOT / "data" / "context_notes.csv"
with tempfile.TemporaryDirectory() as tmp:
    outputs = []
    for i in range(3):
        out = Path(tmp) / f"run{i+1}.csv"
        audit = Path(tmp) / f"audit{i+1}.json"
        run_analysis(shipments, notes, out, audit)
        outputs.append(out.read_bytes())
    hashes = [hashlib.sha256(b).hexdigest() for b in outputs]
    if len(set(hashes)) != 1:
        raise SystemExit(f"Reproducibility FAILED: {hashes}")
    print("PASS: 3/3 CSV outputs are byte-identical.")
    print("SHA-256:", hashes[0])
