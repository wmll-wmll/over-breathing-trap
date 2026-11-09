# 论文级数据一键运行：世界 C，4 臂 × 5 seeds（--breath-fixed = γ 修复后代码）
# 用法: powershell -ExecutionPolicy Bypass -File v11_digital_life\run_paper_5seed.ps1
# 预计耗时: 2.5-3 小时（实测每臂×seed 约 5-8 分钟，共 20 单元）
# 输出: data\V11\exp13_fixed\<C_臂_seedN>\summary.json + 聚合 exp11_<时间戳>.json

$ErrorActionPreference = "Stop"
$Root = Resolve-Path "$PSScriptRoot\.."
$Py = "$Root\.venv\Scripts\python.exe"
$OutDir = "$Root\data\V11\exp13_fixed"

Write-Host "========================================" -ForegroundColor Magenta
Write-Host "  论文级 5-seed 重跑（世界 C + --breath-fixed）" -ForegroundColor Magenta
Write-Host "  开始: $(Get-Date -Format 'yyyy-MM-dd HH:mm:ss')" -ForegroundColor Magenta
Write-Host "========================================" -ForegroundColor Magenta

Push-Location $Root

# ── Phase 0: 核心自检（129 项，含 exp(−2γdt) 实测衰减断言）──────
Write-Host "`n[Phase 0] test_core 自检..." -ForegroundColor Yellow
& $Py v11_digital_life\test_core.py
if ($LASTEXITCODE -ne 0) {
    Write-Host "`n[FAIL] 自检未通过，终止。检查上面的输出。" -ForegroundColor Red
    Pop-Location
    exit 1
}
Write-Host "[Phase 0] 自检通过" -ForegroundColor Green

# ── Phase 1: 5-seed 全量重跑（20 单元）──────────────────────────
Write-Host "`n[Phase 1] 启动 4 臂 × 5 seeds = 20 单元（约 2.5-3 小时）..." -ForegroundColor Yellow
New-Item -ItemType Directory -Force -Path $OutDir | Out-Null

& $Py -m v11_digital_life.exp11_evolution `
    --arms EVO_NL,NONE_NL,EVO,LAM `
    --worlds C `
    --seeds 0,1,2,3,4 `
    --breath-fixed `
    --out $OutDir 2>&1 | Tee-Object -FilePath "$OutDir\run.log"

Pop-Location

Write-Host "`n========================================" -ForegroundColor Magenta
Write-Host "  完成: $(Get-Date -Format 'yyyy-MM-dd HH:mm:ss')" -ForegroundColor Magenta
Write-Host "  结果目录: $OutDir" -ForegroundColor Magenta
Write-Host "========================================" -ForegroundColor Magenta
