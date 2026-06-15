param(
    [Parameter(Mandatory = $true)]
    [string]$TestRoot,

    [Parameter(Mandatory = $true)]
    [string]$Output,

    [ValidateSet("max", "stable", "fast", "legacy")]
    [string]$Mode = "max"
)

$ErrorActionPreference = "Stop"

if (-not (Test-Path -LiteralPath $TestRoot)) {
    throw "测试集目录不存在: $TestRoot"
}

$common = @(
    "run", "-n", "pointnet", "python", "-m", "pointnet_final.predict_advanced",
    "--test-root", $TestRoot,
    "--cache-name", "onsite_test_advanced",
    "--output", $Output,
    "--sampling", "random",
    "--workers", "4",
    "--force-cache"
)

switch ($Mode) {
    "max" {
        $extra = @(
            "--checkpoints",
            "runs/dgcnn_normals_seed1/best.pt",
            "runs/dgcnn_normals_balanced_ft_seed1/best.pt",
            "runs/dgcnn_refine_seed3/best.pt",
            "runs/dgcnn_refine_seed3/last.pt",
            "--model-weights", "0.42", "0.38", "0.10", "0.10",
            "--votes", "3",
            "--batch-size", "20"
        )
    }
    "stable" {
        $extra = @(
            "--checkpoints",
            "runs/dgcnn_normals_seed1/best.pt",
            "runs/dgcnn_normals_balanced_ft_seed1/best.pt",
            "runs/dgcnn_refine_seed3/best.pt",
            "--model-weights", "0.34", "0.36", "0.30",
            "--votes", "3",
            "--batch-size", "24"
        )
    }
    "fast" {
        $extra = @(
            "--checkpoints", "runs/dgcnn_refine_seed3/best.pt",
            "--votes", "1",
            "--batch-size", "64"
        )
    }
    "legacy" {
        & conda run -n pointnet python -m pointnet_final.predict `
            --test-root $TestRoot `
            --cache-name onsite_test_legacy `
            --checkpoints runs/dgcnn_normals_balanced_ft_seed1/best_class.pt `
            --output $Output `
            --votes 20 `
            --batch-size 32 `
            --workers 4 `
            --force-cache
        exit $LASTEXITCODE
    }
}

& conda @common @extra
exit $LASTEXITCODE

