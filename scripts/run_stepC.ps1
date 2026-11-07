# Step C: popularity-first coverage curve (item side only, LLM embeddings, alpha=1.0, fixed 25 epochs)
#   pop10 / pop20 / pop30 / pop50 : augment only the TOP 10/20/30/50% most popular items
#                                   (popularity = train interaction count; built with prune_pagerank.py --score degree)
#   random-order controls at the same coverage:
#       10% and 30% are run here (--ratio 0.1 / 0.3, default random order, tag itemonly);
#       20% and 50% already exist from the main experiment (week4fe_aug_itemonly_r0.2 / r0.5).
# alpha=0 control already exists (week4fe_aug_r1.0_a0.0).
# Usage: from the PROJECT ROOT with venv activated:  .\scripts\run_stepC.ps1
# Time: 6 variants x 5 seeds x ~265 s = about 2.2 hours. Finished runs are skipped, so it can be stopped and restarted.
# Set $seeds = 0..9 for 10 seeds (about 4.4 hours; the random 20/50% controls already have 10 seeds).
# NOTE: keep this file ASCII-only (Windows PowerShell 5.1 misreads UTF-8 without BOM).

$seeds = 0..4
$pop = @(
    @{ keep = "10"; drop = "0.90" },
    @{ keep = "20"; drop = "0.80" },
    @{ keep = "30"; drop = "0.70" },
    @{ keep = "50"; drop = "0.50" }
)
$randRatios = @("0.1", "0.3")

function Check($what) {
    if ($LASTEXITCODE -ne 0) { throw "$what failed (exit code $LASTEXITCODE)" }
}

# 1. Popularity masks (fast, no GPU). Keeping the top X% = dropping the lowest (100-X)% by degree.
foreach ($v in $pop) {
    $dir = "results/stepC/pop$($v.keep)"
    if (Test-Path "$dir/item_has.npy") { Write-Host "Skip: $dir already exists"; continue }
    python scripts/prune_pagerank.py --score degree --drop low --frac $v.drop --out_dir $dir
    Check "prune_pagerank degree (pop$($v.keep))"
}

# 2. Train popularity-first variants (ratio 1.0 on the filtered item_has)
foreach ($v in $pop) {
    $dir = "results/stepC/pop$($v.keep)"
    $tag = "itempop$($v.keep)"
    foreach ($s in $seeds) {
        $out = "results/week4fe_aug_${tag}_r1.0_a1.0_s$s.json"
        if (Test-Path $out) { continue }
        python scripts/train_augmented.py --ratio 1.0 --alpha 1.0 --seed $s --aug_side item --fixed_epochs 25 --aug_dir $dir --tag $tag
        Check "train_augmented ($tag seed $s)"
    }
}

# 3. Random-order controls at 10% and 30% (item side only)
foreach ($r in $randRatios) {
    foreach ($s in $seeds) {
        $out = "results/week4fe_aug_itemonly_r${r}_a1.0_s$s.json"
        if (Test-Path $out) { continue }
        python scripts/train_augmented.py --ratio $r --alpha 1.0 --seed $s --aug_side item --fixed_epochs 25
        Check "train_augmented (random item ratio $r seed $s)"
    }
}

# 4. Summary table + figure
python scripts/stepC_summary.py --seeds ($seeds -join ",")
