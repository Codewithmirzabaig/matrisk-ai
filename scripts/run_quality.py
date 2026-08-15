"""Generate a machine-readable quality report for all supplied datasets."""
from __future__ import annotations
import json
from pathlib import Path
from src.data.loaders import load_csv, validate_materials

root=Path(__file__).resolve().parents[1]; raw=root/"data"/"raw"; out=root/"artifacts"
report={}
for path in sorted(raw.glob("DS*.csv")):
    frame=load_csv(path); item={"rows":len(frame),"columns":len(frame.columns),
        "completeness":float(1-frame.isna().mean().mean()),"duplicates":int(frame.duplicated().sum())}
    if path.name.startswith("DS1"):
        q=validate_materials(frame); item["physics_violations"]=list(q.violations); item["passed"]=q.passed
    report[path.name]=item
out.mkdir(exist_ok=True); (out/"data_quality.json").write_text(json.dumps(report,indent=2))
print(json.dumps(report,indent=2))

