# Step A: do LLM-generated attributes really matter?
#   meta    : title + genre text only (no LLM) -> same bge model -> same item-only augmentation
#   shuffle : LLM item embeddings randomly permuted among items (content destroyed, distribution kept)
# Item side only, ratio 1.0, alpha 1.0, fixed 25 epochs, seeds 0..9 (same protocol as the main experiment).
# Time: 2 conditions x 10 seeds x ~265 s = about 1.5 hours. Finished runs are skipped, so it is safe to re-run.
# Usage (project root, venv activated):  .\scripts\run_stepA.ps1
# NOTE: keep this file ASCII-only (Windows PowerShell 5.1 misreads UTF-8 without BOM).

$seeds = 0..9
$variants = @(
    @{ name = "meta";    tag = "itemmeta" },
    @{ name = "shuffle"; tag = "itemshuffle" }
)

function Check($what) {
    if ($LASTEXITCODE -ne 0) { throw "$what failed (exit code $LASTEXITCODE)" }
}

# 0. MostPop (seconds)
if (-not (Test-Path "results/mostpop_results.json")) {
    python scripts/mostpop.py
    Check "mostpop"
}

# 1. Build control embeddings
foreach ($v in $variants) {
    $dir = "results/stepA/$($v.name)"
    if (Test-Path "$dir/item_emb.npy") { Write-Host "Skip build: $dir exists"; continue }
    python scripts/build_control_embeddings.py --mode $v.name --out_dir $dir --perm_seed 0
    Check "build_control_embeddings ($($v.name))"
}

# 2. Train (item side only)
foreach ($v in $variants) {
    $dir = "results/stepA/$($v.name)"
    foreach ($s in $seeds) {
        $out = "results/week4fe_aug_$($v.tag)_r1.0_a1.0_s$s.json"
        if (Test-Path $out) { continue }
        python scripts/train_augmented.py --ratio 1.0 --alpha 1.0 --seed $s --aug_side item --fixed_epochs 25 --aug_dir $dir --tag $v.tag
        Check "train_augmented ($($v.tag) seed $s)"
    }
}

# 3a. Each control vs alpha=0 (does it help at all?)
foreach ($v in $variants) {
    Write-Host ""
    Write-Host "########## $($v.tag) vs alpha=0 ##########"
    python scripts/summarize_week4.py --ratio 1.0 --prefix "week4fe_aug_$($v.tag)" --control_prefix week4fe_aug --a0 0.0 --a1 1.0
}

# 3b. KEY TEST: each control vs LLM item-only (does the LLM text add anything?)
foreach ($v in $variants) {
    Write-Host ""
    Write-Host "########## $($v.tag) vs LLM item-only (alpha=1.0) ##########"
    python scripts/summarize_week4.py --ratio 1.0 --prefix "week4fe_aug_$($v.tag)" --control_prefix week4fe_aug_itemonly --a0 1.0 --a1 1.0
}
