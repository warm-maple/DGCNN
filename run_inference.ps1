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
    "--data-root", "F:\Python Project\pointnet\modelnet40_normal_resampled",
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
            "runs/clean_stage1_seed2026/best.pt",
            "runs/clean_stage2_balanced_seed2026/best.pt",
            "--model-weights", "0.5", "0.5",
            "--votes", "7",
            "--batch-size", "24"
        )
    }
    "stable" {
        $extra = @(
            "--checkpoints",
            "runs/clean_stage1_seed2026/best.pt",
            "--votes", "3",
            "--batch-size", "32"
        )
    }
    "fast" {
        $extra = @(
            "--checkpoints", "runs/clean_stage1_seed2026/best.pt",
            "--votes", "1",
            "--batch-size", "64"
        )
    }
    "legacy" {
        & conda run -n pointnet python -m pointnet_final.predict_advanced `
            --test-root $TestRoot `
            --cache-name onsite_test_stage2 `
            --checkpoints runs/clean_stage2_balanced_seed2026/best.pt `
            --output $Output `
            --votes 3 `
            --sampling random `
            --batch-size 32 `
            --workers 4 `
            --force-cache
        exit $LASTEXITCODE
    }
}

& conda @common @extra
exit $LASTEXITCODE

