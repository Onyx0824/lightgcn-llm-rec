# Week 5 D group: semantic-outlier pruning of item embeddings + random-drop controls
#   prune10 / prune20 : drop the 10% / 20% of items least consistent with their genre centroid
#   rand10  / rand20  : drop the same NUMBER of items at random (control for "fewer augmented items")
# Item side only, ratio 1.0, alpha 1.0, fixed 25 epochs.
# Usage: from the PROJECT ROOT with venv activated:  .\scripts\run_week5_prune.ps1
# Time: 4 variants x seeds x ~265 s. With seeds 0..4 that is about 1.5 hours.
# To extend to 10 seeds later, change $seeds to 0..9 and rerun (finished runs are skipped).
# NOTE: keep this file ASCII-only (Windows PowerShell 5.1 misreads UTF-8 without BOM).

$seeds = 0..4
$variants = @(
    @{ name = "prune10"; mode = "score";  frac = "0.10"; tag = "itemprune10" },
    @{ name = "prune20"; mode = "score";  frac = "0.20"; tag = "itemprune20" },
    @{ name = "rand10";  mode = "random"; frac = "0.10"; tag = "itemrand10" },
    @{ name = "rand20";  mode = "random"; frac = "0.20"; tag = "itemrand20" }
)

function Check($what) {
    if ($LASTEXITCODE -ne 0) { throw "$what failed (exit code $LASTEXITCODE)" }
}

# 1. Build pruned item_has masks (fast, no GPU). Output: results/week5/<name>/ (+ prune_stats.json)
foreach ($v in $variants) {
    $dir = "results/week5/$($v.name)"
    if (Test-Path "$dir/item_has.npy") { Write-Host "Skip prune: $dir already exists"; continue }
    python scripts/prune_items.py --mode $v.mode --frac $v.frac --out_dir $dir --seed 0
    Check "prune_items ($($v.name))"
}

# 2. Train, item side only
foreach ($v in $variants) {
    $dir = "results/week5/$($v.name)"
    foreach ($s in $seeds) {
        $out = "results/week4fe_aug_$($v.tag)_r1.0_a1.0_s$s.json"
        if (Test-Path $out) { continue }
        python scripts/train_augmented.py --ratio 1.0 --alpha 1.0 --seed $s --aug_side item --fixed_epochs 25 --aug_dir $dir --tag $v.tag
        Check "train_augmented ($($v.tag) seed $s)"
    }
}

# 3a. Each variant vs unfiltered item-only
foreach ($v in $variants) {
    Write-Host ""
    Write-Host "########## $($v.tag) vs unfiltered item-only (alpha=1.0) ##########"
    python scripts/summarize_week4.py --ratio 1.0 --prefix "week4fe_aug_$($v.tag)" --control_prefix week4fe_aug_itemonly --a0 1.0 --a1 1.0
}

# 3b. KEY TEST: score-based pruning vs random drop of the same size
foreach ($p in @("10", "20")) {
    Write-Host ""
    Write-Host "########## prune$p vs rand$p (same number dropped): does SELECTION matter? ##########"
    python scripts/summarize_week4.py --ratio 1.0 --prefix "week4fe_aug_itemprune$p" --control_prefix "week4fe_aug_itemrand$p" --a0 1.0 --a1 1.0
}
