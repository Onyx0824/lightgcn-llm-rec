# Step B1: early stopping for BOTH sides (checks that the conclusions do not depend on the fixed 25-epoch protocol)
#   conditions: alpha=0 | LLM item-only (alpha=1.0, 100%) | metadata item-only (alpha=1.0, 100%, needs results/stepA/meta)
#   early stopping = validation Recall@20 every 5 epochs, patience 5 evaluations (configs/baseline.yaml), best-validation weights are tested
# Usage (project root, venv on):  .\scripts\run_stepB1.ps1
# Time: about 8-16 min per run (best epoch 20-65 + 25 patience epochs at ~10.6 s/epoch)
#       3 conditions x 5 seeds = about 2.5 h.  Set $seeds = 0..9 for 10 seeds (about 5 h) or $includeMeta = $false (about 1.7 h).
# Finished runs are skipped, so you can stop and restart.   NOTE: keep this file ASCII-only.

$seeds = 0..4
$includeMeta = $true

python scripts/make_stepB_configs.py
if ($LASTEXITCODE -ne 0) { throw "make_stepB_configs failed" }
New-Item -ItemType Directory -Force -Path "results/stepB/es" | Out-Null

function Run($label, $outName, $extraArgs, $seed) {
    $out = "results/stepB/es/$outName"
    if (Test-Path $out) { return }
    python scripts/train_augmented.py --config configs/stepB_es.yaml --ratio 1.0 --seed $seed @extraArgs
    if ($LASTEXITCODE -ne 0) { throw "$label seed $seed failed" }
}

foreach ($s in $seeds) {
    Run "alpha0"   "week4_aug_r1.0_a0.0_s$s.json"          @("--alpha", "0.0") $s
    Run "llm-item" "week4_aug_itemonly_r1.0_a1.0_s$s.json" @("--alpha", "1.0", "--aug_side", "item") $s
    if ($includeMeta) {
        Run "meta-item" "week4_aug_itemmeta_r1.0_a1.0_s$s.json" @("--alpha", "1.0", "--aug_side", "item", "--aug_dir", "results/stepA/meta", "--tag", "itemmeta") $s
    }
}
python scripts/stepB_summary.py --seeds ($seeds -join ",")
