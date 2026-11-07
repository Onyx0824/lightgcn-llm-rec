"""Create Step B config files from configs/baseline.yaml (run once from the project root).
  configs/stepB_es.yaml          : same hyper-parameters as baseline, results go to results/stepB/es
  configs/stepB_tune_<name>.yaml : ONE hyper-parameter changed, results go to results/stepB/tune/<name>
"""
import yaml
from pathlib import Path

base = yaml.safe_load(open("configs/baseline.yaml", encoding="utf-8"))
variants = {  # one-at-a-time changes around the default (lr 1e-3, weight_decay 1e-4, 3 layers, dim 64)
    "lr3e-3": {"lr": 3e-3},
    "wd1e-3": {"weight_decay": 1e-3},
    "layers2": {"n_layers": 2},
    "layers4": {"n_layers": 4},
    "dim128": {"embed_dim": 128},
}
Path("configs").mkdir(exist_ok=True)

c = dict(base); c["results_dir"] = "results/stepB/es"
yaml.safe_dump(c, open("configs/stepB_es.yaml", "w", encoding="utf-8"), sort_keys=False)
for name, ov in variants.items():
    c = dict(base); c.update(ov); c["results_dir"] = f"results/stepB/tune/{name}"
    yaml.safe_dump(c, open(f"configs/stepB_tune_{name}.yaml", "w", encoding="utf-8"), sort_keys=False)
print("configs written:", ["stepB_es"] + [f"stepB_tune_{n}" for n in variants])
