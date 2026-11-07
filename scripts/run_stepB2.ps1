# Step B2: is the default LightGCN baseline clearly under-tuned?
#   Stage 1 (screen): one-at-a-time changes around the default, alpha=0 only, early stopping, seeds 0-1.
#                     Selection uses VALIDATION Recall@20 only (never the test set).
#   Stage 2 (confirm): only if Stage 1 finds a better config, set $best below (e.g. "lr3e-3") and run again:
#                     alpha=0 and LLM item-only with that config, early stopping, seeds 0-4.
# Usage:  .\scripts\run_stepB2.ps1
# Time: Stage 1 = 5 configs x 2 seeds x ~10 min = about 1.7 h.  Stage 2 = 2 conditions x 5 seeds = about 1.7 h.
# NOTE: keep this file ASCII-only.

$best = ""          # fill in after reading the Stage 1 table, e.g. $best = "layers2"
$screenSeeds = 0..1
$confirmSeeds = 0..4
$variants = @("lr3e-3", "wd1e-3", "layers2", "layers4", "dim128")

python scripts/make_stepB_configs.py
if ($LASTEXITCODE -ne 0) { throw "make_stepB_configs failed" }

foreach ($v in $variants) {
    New-Item -ItemType Directory -Force -Path "results/stepB/tune/$v" | Out-Null
    foreach ($s in $screenSeeds) {
        $out = "results/stepB/tune/$v/week4_aug_r1.0_a0.0_s$s.json"
        if (Test-Path $out) { continue }
        python scripts/train_augmented.py --config "configs/stepB_tune_$v.yaml" --ratio 1.0 --alpha 0.0 --seed $s
        if ($LASTEXITCODE -ne 0) { throw "screen $v seed $s failed" }
    }
}
python scripts/stepB_summary.py --tune_only

if ($best -ne "") {
    foreach ($s in $confirmSeeds) {
        $d = "results/stepB/tune/$best"
        New-Item -ItemType Directory -Force -Path $d | Out-Null
        $o0 = "$d/week4_aug_r1.0_a0.0_s$s.json"
        if (-not (Test-Path $o0)) {
            python scripts/train_augmented.py --config "configs/stepB_tune_$best.yaml" --ratio 1.0 --alpha 0.0 --seed $s
            if ($LASTEXITCODE -ne 0) { throw "confirm alpha0 seed $s failed" }
        }
        $o1 = "$d/week4_aug_itemonly_r1.0_a1.0_s$s.json"
        if (-not (Test-Path $o1)) {
            python scripts/train_augmented.py --config "configs/stepB_tune_$best.yaml" --ratio 1.0 --alpha 1.0 --seed $s --aug_side item
            if ($LASTEXITCODE -ne 0) { throw "confirm llm-item seed $s failed" }
        }
    }
    python scripts/stepB_summary.py --confirm $best --seeds ($confirmSeeds -join ",")
}
