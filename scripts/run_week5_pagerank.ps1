# Week 5 PageRank group: drop LLM item embeddings of low-/high-PageRank items
#   prlow30 / prlow50   : drop the 30% / 50% of augmented items with the LOWEST PageRank
#   prhigh30 / prhigh50 : drop the 30% / 50% with the HIGHEST PageRank (diagnostic arm)
#   rand30 / rand50 / rand80 / rand90 : drop the same NUMBER of items at random (control, built by prune_items.py)
#                         rand80 / rand90 cover about the same share of interactions as prhigh30 / prhigh50
# Control already run: unfiltered item-only (week4fe_aug_itemonly).
# Item side only, ratio 1.0, alpha 1.0, fixed 25 epochs.
# Usage: from the PROJECT ROOT with venv activated:  .\scripts\run_week5_pagerank.ps1
# Time: 8 variants x seeds x ~265 s. With seeds 0..4 that is about 3 hours (rand80 / rand90 add about 45 minutes).
# Finished runs are skipped, so it can be stopped and restarted.
# NOTE: keep this file ASCII-only (Windows PowerShell 5.1 misreads UTF-8 without BOM).

$seeds = 0..4
$variants = @(
    @{ name = "prlow30";  drop = "low";  frac = "0.30"; tag = "itemprlow30"  },
    @{ name = "prlow50";  drop = "low";  frac = "0.50"; tag = "itemprlow50"  },
    @{ name = "prhigh30"; drop = "high"; frac = "0.30"; tag = "itemprhigh30" },
    @{ name = "prhigh50"; drop = "high"; frac = "0.50"; tag = "itemprhigh50" }
)
$randVariants = @(
    @{ name = "rand30"; frac = "0.30"; tag = "itemrand30" },
    @{ name = "rand50"; frac = "0.50"; tag = "itemrand50" },
    @{ name = "rand80"; frac = "0.80"; tag = "itemrand80" },
    @{ name = "rand90"; frac = "0.90"; tag = "itemrand90" }
)

function Check($what) {
    if ($LASTEXITCODE -ne 0) { throw "$what failed (exit code $LASTEXITCODE)" }
}

# 1a. PageRank masks (fast, no GPU). Output: results/week5/<name>/ (+ prune_stats.json, pagerank_items.csv)
foreach ($v in $variants) {
    $dir = "results/week5/$($v.name)"
    if (Test-Path "$dir/item_has.npy") { Write-Host "Skip: $dir already exists"; continue }
    python scripts/prune_pagerank.py --drop $v.drop --frac $v.frac --out_dir $dir
    Check "prune_pagerank ($($v.name))"
}

# 1b. Random-drop control masks at the same fractions
foreach ($v in $randVariants) {
    $dir = "results/week5/$($v.name)"
    if (Test-Path "$dir/item_has.npy") { Write-Host "Skip: $dir already exists"; continue }
    python scripts/prune_items.py --mode random --frac $v.frac --out_dir $dir --seed 0
    Check "prune_items random ($($v.name))"
}

# 2. Train, item side only (PageRank variants + random controls)
foreach ($v in ($variants + $randVariants)) {
    $dir = "results/week5/$($v.name)"
    foreach ($s in $seeds) {
        $out = "results/week4fe_aug_$($v.tag)_r1.0_a1.0_s$s.json"
        if (Test-Path $out) { continue }
        python scripts/train_augmented.py --ratio 1.0 --alpha 1.0 --seed $s --aug_side item --fixed_epochs 25 --aug_dir $dir --tag $v.tag
        Check "train_augmented ($($v.tag) seed $s)"
    }
}

# 3a. Each variant vs unfiltered item-only
foreach ($v in ($variants + $randVariants)) {
    Write-Host ""
    Write-Host "########## $($v.tag) vs unfiltered item-only (alpha=1.0) ##########"
    python scripts/summarize_week4.py --ratio 1.0 --prefix "week4fe_aug_$($v.tag)" --control_prefix week4fe_aug_itemonly --a0 1.0 --a1 1.0
}

# 3b. KEY TEST: PageRank-selected drop vs random drop of the same size
foreach ($p in @("30", "50")) {
    foreach ($d in @("low", "high")) {
        Write-Host ""
        Write-Host "########## pr$d$p vs rand$p (same number dropped): does PageRank selection matter? ##########"
        python scripts/summarize_week4.py --ratio 1.0 --prefix "week4fe_aug_itempr$d$p" --control_prefix "week4fe_aug_itemrand$p" --a0 1.0 --a1 1.0
    }
}

# 3c. Matched share of interactions: PageRank-high drop vs random drop that covers about the same share
Write-Host ""
Write-Host "########## prhigh30 (76.5% of interactions) vs rand80 (80.2%) ##########"
python scripts/summarize_week4.py --ratio 1.0 --prefix week4fe_aug_itemprhigh30 --control_prefix week4fe_aug_itemrand80 --a0 1.0 --a1 1.0
Write-Host ""
Write-Host "########## prhigh50 (91.1% of interactions) vs rand90 (90.6%) ##########"
python scripts/summarize_week4.py --ratio 1.0 --prefix week4fe_aug_itemprhigh50 --control_prefix week4fe_aug_itemrand90 --a0 1.0 --a1 1.0
