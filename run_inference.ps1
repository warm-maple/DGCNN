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

$ResolvedTestRoot = (Resolve-Path -LiteralPath $TestRoot).Path
$ProjectRoot = Split-Path -Parent $PSCommandPath

Push-Location $ProjectRoot
try {
    $common = @(
        "run", "--no-capture-output", "-n", "pointnet", "python", "-m", "pointnet_final.predict_advanced",
        "--test-root", $ResolvedTestRoot,
        "--cache-name", "onsite_test_advanced",
        "--output", $Output,
        "--sampling", "random",
        "--seed", "2026",
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
            $extra = @(
                "--checkpoints",
                "runs/clean_stage2_balanced_seed2026/best.pt",
                "--votes", "3",
                "--batch-size", "32"
            )
        }
    }

    & conda @common @extra
    exit $LASTEXITCODE
}
finally {
    Pop-Location
}
