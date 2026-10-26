# Week 7: train alpha=0 control and item-only augmentation WITH embeddings/top-20 saved, then run the analyses.
#   control   : --ratio 1.0 --alpha 0   --tag w7      -> results/week4fe_aug_w7_r1.0_a0.0_s<seed>.json
#   treatment : --ratio 1.0 --alpha 1.0 --aug_side item --tag w7item -> results/week4fe_aug_w7item_r1.0_a1.0_s<seed>.json
# Separate tags, so the main-experiment result files (week4fe_aug_*) are NOT overwritten.
# Saved arrays go to results/week7/emb/ (*.npy, not tracked by git).
# Usage: from the PROJECT ROOT with venv activated:  .\scripts\run_week7.ps1
# Time: 2 x seeds x ~265 s. With seeds 0..4 about 45 min; with 0..9 about 1.5 hours.
# Finished runs are skipped, so it is safe to stop and rerun.
# NOTE: keep this file ASCII-only (Windows PowerShell 5.1 misreads UTF-8 without BOM).

$seeds = 0..4

function Check($what) {
    if ($LASTEXITCODE -ne 0) { throw "$what failed (exit code $LASTEXITCODE)" }
}

New-Item -ItemType Directory -Force -Path results/week7/emb | Out-Null

foreach ($s in $seeds) {
    $out = "results/week4fe_aug_w7_r1.0_a0.0_s$s.json"
    if (-not (Test-Path $out)) {
        python scripts/train_augmented.py --ratio 1.0 --alpha 0 --seed $s --fixed_epochs 25 --tag w7 --save_emb
        Check "train control seed $s"
    }
    $out = "results/week4fe_aug_w7item_r1.0_a1.0_s$s.json"
    if (-not (Test-Path $out)) {
        python scripts/train_augmented.py --ratio 1.0 --alpha 1.0 --seed $s --aug_side item --fixed_epochs 25 --tag w7item --save_emb
        Check "train item-only seed $s"
    }
}

# Sanity: the w7 control/treatment should reproduce the main-experiment numbers within seed noise
Write-Host ""
Write-Host "########## sanity: w7 item-only vs w7 control (should be about +0.0027 Recall / +0.0042 NDCG) ##########"
python scripts/summarize_week4.py --ratio 1.0 --prefix week4fe_aug_w7item --control_prefix week4fe_aug_w7 --a0 0.0 --a1 1.0

python scripts/popularity_strata.py --by item
Check "popularity_strata item"
python scripts/popularity_strata.py --by user
Check "popularity_strata user"
python scripts/error_analysis.py
Check "error_analysis"
python scripts/visualize_embeddings.py
Check "visualize_embeddings"

Write-Host ""
Write-Host "Done. Outputs in results/week7/: strata_item.md/.png, strata_user.md/.png, error_cases.md, embeddings_pca.png, embeddings_tsne.png, embedding_cohesion.md"
