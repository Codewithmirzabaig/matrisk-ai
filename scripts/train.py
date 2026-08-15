"""Train and persist the leakage-safe material baseline."""
from __future__ import annotations
import json
from pathlib import Path
import joblib
from src.data.loaders import load_csv
from src.models.material import train_material_model

root=Path(__file__).resolve().parents[1]; output=root/"artifacts"/"models"; output.mkdir(parents=True,exist_ok=True)
frame=load_csv(root/"data"/"raw"/"DS1_material_properties_5500.csv")
bundle=train_material_model(frame)
joblib.dump(bundle,output/"material_model.joblib")
(output/"material_metrics.json").write_text(json.dumps(bundle.metrics,indent=2))
print(json.dumps(bundle.metrics,indent=2))

