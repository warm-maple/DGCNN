# ModelNet40 Point Cloud Classification

This project trains a DGCNN classifier on the teacher-provided ModelNet40 train split and writes the required `id,label` CSV for a held-out test folder.

## Quick Start

```powershell
conda run -n pointnet python -m pointnet_final.prepare_cache
conda run -n pointnet python -m pointnet_final.train --run-dir runs/dgcnn_normals_seed1 --epochs 250 --batch-size 24 --eval-batch-size 32 --workers 4 --seed 1
conda run -n pointnet python -m pointnet_final.predict --test-list modelnet40_normal_resampled/modelnet40_test.txt --cache-name rehearsal_test --checkpoints runs/dgcnn_normals_balanced_ft_seed1/best_class.pt --output runs/rehearsal_submission.csv --votes 20
```

For the on-site test set, replace `--test-root` with the released test directory and set `--output` to the required file name.

Best rehearsal result so far:

- Single balanced DGCNN checkpoint, 20 votes: Instance Accuracy `92.22%`, Class Accuracy `91.10%`.
- 3-checkpoint ensemble, 20 votes: Instance Accuracy `92.83%`, Class Accuracy `90.56%`.

The single-checkpoint setup is faster and has the larger class-accuracy margin. The ensemble has the best instance accuracy.
