# Week 5 denoising experiments (item side; ratio 1.0, alpha 1.0, fixed 25 epochs, seeds 0-9)
#   A = drop_dirty : drop the 72 items with simplified-Chinese residue
#   B = drop_retry : drop the 342 items with attempts>1 (every dirty item has attempts>1, so B contains A;
#                    "A+B" would be identical to B and is not run)
# Usage: run from the PROJECT ROOT with the venv activated:  .\scripts\run_week5_denoise.ps1
# Time: about 2 x 10 x 265 s = 1.5 hours. Existing result files are skipped, so it is safe to stop and rerun.
# NOTE: keep this file ASCII-only (Windows PowerShell 5.1 misreads UTF-8 without BOM).

$seeds = 0..9
$variants = @(
    @{ name = "drop_dirty"; flag = "--drop_dirty"; tag = "itemdropdirty" },
    @{ name = "drop_retry"; flag = "--drop_retry"; tag = "itemdropretry" }
)

function Check($what) {
    if ($LASTEXITCODE -ne 0) { throw "$what failed (exit code $LASTEXITCODE)" }
}

# 1. Build denoised embeddings into results/week5/<name>/ (does not overwrite results/week4)
foreach ($v in $variants) {
    $dir = "results/week5/$($v.name)"
    if (Test-Path "$dir/item_emb.npy") { Write-Host "Skip build: $dir already exists"; continue }
    python src/llm_augmentation/build_embeddings.py $v.flag --out_dir $dir
    Check "build_embeddings ($($v.name))"
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

# 3. Paired comparisons
foreach ($v in $variants) {
    Write-Host ""
    Write-Host "########## $($v.tag) vs unfiltered item-only (alpha=1.0): does denoising help? ##########"
    python scripts/summarize_week4.py --ratio 1.0 --prefix "week4fe_aug_$($v.tag)" --control_prefix week4fe_aug_itemonly --a0 1.0 --a1 1.0
    Write-Host ""
    Write-Host "########## $($v.tag) vs alpha=0 control: gain over no augmentation ##########"
    python scripts/summarize_week4.py --ratio 1.0 --prefix "week4fe_aug_$($v.tag)" --control_prefix week4fe_aug
}
