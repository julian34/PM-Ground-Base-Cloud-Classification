<#
.SYNOPSIS
    Jalankan 3 run Stratified K-Fold CV berurutan untuk ablasi USE_BR x STRONG_AUG,
    lalu bandingkan hasilnya. "Edit flag -> train -> ulang" otomatis.

.DESCRIPTION
    Flag diubah lewat environment variable (CDN_USE_BR / CDN_STRONG_AUG) yang dibaca
    config.py saat impor - jadi TIDAK ada file sumber yang ditulis-ulang, working tree
    tetap bersih, dan tiap proses python memakai nilai yang benar.

    Baseline default: USE_BR=on, STRONG_AUG=off (SCConv+SE selalu on).

    Urutan run:
      baseline : USE_BR=1  STRONG_AUG=0   (acuan)
      br_off   : USE_BR=0  STRONG_AUG=0   (isolasi kontribusi Biru/Merah)
      aug_on   : USE_BR=1  STRONG_AUG=1   (isolasi efek augmentasi kuat)

.PARAMETER Folds
    Jumlah fold CV (default 3).

.PARAMETER AccumSteps
    Gradient Accumulation; naikkan (mis. 4) bila CUDA out of memory. Default 1.

.EXAMPLE
    .\scripts\run_ablation.ps1
    .\scripts\run_ablation.ps1 -AccumSteps 4
#>
param(
    [int]$Folds = 3,
    [int]$AccumSteps = 1
)

$ErrorActionPreference = "Stop"
$env:PYTHONIOENCODING = "utf-8"    # konsol Windows aman untuk +/- dan Delta

# Root repo = folder induk dari scripts\
$Root = Split-Path -Parent $PSScriptRoot
$Results = Join-Path $Root "artifacts\results"
New-Item -ItemType Directory -Force -Path $Results | Out-Null

# Python: utamakan venv proyek, jika ada
$VenvPy = Join-Path $Root ".venv\Scripts\python.exe"
$Py = if (Test-Path $VenvPy) { $VenvPy } else { "python" }
Write-Host "Python  : $Py"
Write-Host "Folds   : $Folds | AccumSteps: $AccumSteps"
Write-Host "Results : $Results`n"

# Definisi 3 run (nama + nilai env flag)
$Runs = @(
    @{ name = "baseline"; br = "1"; aug = "0" },
    @{ name = "br_off";   br = "0"; aug = "0" },
    @{ name = "aug_on";   br = "1"; aug = "1" }
)

$ok = @()
$failed = @()

foreach ($r in $Runs) {
    $name = $r.name
    $log = Join-Path $Results "ablation_$name.log"
    $env:CDN_USE_BR = $r.br
    $env:CDN_STRONG_AUG = $r.aug

    Write-Host ("=" * 70)
    Write-Host ("RUN '$name'  ->  USE_BR={0}  STRONG_AUG={1}" -f $r.br, $r.aug)
    Write-Host ("mulai: {0} | log: {1}" -f (Get-Date -Format "yyyy-MM-dd HH:mm:ss"), $log)
    Write-Host ("=" * 70)

    $t0 = Get-Date
    & $Py (Join-Path $Root "scripts\train_kfold.py") --folds $Folds --run-name $name --accum-steps $AccumSteps 2>&1 |
        Tee-Object -FilePath $log
    $code = $LASTEXITCODE
    $dur = [int]((Get-Date) - $t0).TotalMinutes

    if ($code -eq 0) {
        Write-Host ("[OK] run '$name' selesai dalam {0} menit`n" -f $dur) -ForegroundColor Green
        $ok += $name
    } else {
        Write-Warning ("run '$name' GAGAL (exit {0}) setelah {1} menit - lihat {2}" -f $code, $dur, $log)
        $failed += $name
    }
}

# Bersihkan env agar sesi PowerShell kembali ke default
Remove-Item Env:CDN_USE_BR, Env:CDN_STRONG_AUG -ErrorAction SilentlyContinue

Write-Host ("`n" + ("=" * 70))
Write-Host "RINGKASAN ABLASI"
Write-Host ("  sukses : {0}" -f ($(if ($ok.Count) { $ok -join ', ' } else { '(tak ada)' })))
Write-Host ("  gagal  : {0}" -f ($(if ($failed.Count) { $failed -join ', ' } else { '(tak ada)' })))
Write-Host ("=" * 70)

if ($ok.Count -ge 2) {
    Write-Host "`nMembandingkan hasil (compare_kfold.py):`n"
    & $Py (Join-Path $Root "scripts\compare_kfold.py") @ok
} else {
    Write-Warning "Kurang dari 2 run sukses - lewati perbandingan."
}

Write-Host "`nLangkah berikutnya (keputusan manual):"
Write-Host "  1. Baca verdict compare: pertahankan USE_BR bila 'br_off' NYATA lebih buruk;"
Write-Host "     set STRONG_AUG=True hanya bila 'aug_on' NYATA lebih baik (bukan DERAU)."
Write-Host "  2. Set kombinasi pemenang di src/clouddensenet/config.py (nilai default)."
Write-Host "  3. Latih model final: $Py scripts\train.py --run-name cloud_final   (80 epoch, seluruh train/)."
Write-Host "  4. Evaluasi SEKALI di test/ untuk angka final."
