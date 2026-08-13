$shell = New-Object -ComObject Shell.Application

# Items to remove (to Recycle Bin, recoverable)
$items = @(
    @{Path = "D:\Hermes\models"; Name = "qwen2.5-7b-abliterated-v2-q4_k_m.gguf"; Size = "~4.36 GB"},
    @{Path = "D:\Hermes"; Name = "backup_manager.py"; Size = "small"},
    @{Path = "D:\Hermes"; Name = "test_abliterated.py"; Size = "small"},
    @{Path = "D:\Hermes"; Name = "check_model.py"; Size = "small"},
    @{Path = "D:\Hermes"; Name = "verify_backup.py"; Size = "small"},
    @{Path = "D:\Hermes"; Name = "auto_backup_scheduler.ps1"; Size = "small"},
    @{Path = "D:\Hermes"; Name = "restore_model.ps1"; Size = "small"}
)

Write-Host "=== Moving to Recycle Bin (recoverable) ===" -ForegroundColor Cyan
foreach ($item in $items) {
    $full = Join-Path $item.Path $item.Name
    if (Test-Path $full) {
        $folder = $shell.NameSpace($item.Path)
        $file = $folder.ParseName($item.Name)
        if ($file) {
            $file.InvokeVerb("delete")
            Write-Host "  [x] $($item.Name) ($($item.Size))" -ForegroundColor Green
        } else {
            Write-Host "  [!] Could not parse: $full" -ForegroundColor Yellow
        }
    } else {
        Write-Host "  [-] Not found: $full" -ForegroundColor Gray
    }
}

Write-Host ""
Write-Host "=== Cleanup extra backup zip (created by backup_manager test) ===" -ForegroundColor Cyan
$backupZip = "D:\Hermes\backups\hermes-backup-20260809_132334.zip"
if (Test-Path $backupZip) {
    $shell2 = New-Object -ComObject Shell.Application
    $folder = $shell2.NameSpace("D:\Hermes\backups")
    $file = $folder.ParseName("hermes-backup-20260809_132334.zip")
    if ($file) {
        $file.InvokeVerb("delete")
        Write-Host "  [x] backups\hermes-backup-20260809_132334.zip (~2 GB)" -ForegroundColor Green
    }
}
# Remove empty backups folder
$backupsDir = "D:\Hermes\backups"
if (Test-Path $backupsDir) {
    $remaining = Get-ChildItem $backupsDir -ErrorAction SilentlyContinue
    if (-not $remaining) {
        $shell3 = New-Object -ComObject Shell.Application
        $folder = $shell3.NameSpace("D:\Hermes")
        $file = $folder.ParseName("backups")
        if ($file) { $file.InvokeVerb("delete") }
        Write-Host "  [x] Empty backups/ folder removed" -ForegroundColor Green
    }
}

Write-Host ""
Write-Host "=== Cleanup .cache (huggingface download cache) ===" -ForegroundColor Cyan
$cacheDir = "D:\Hermes\models\.cache"
if (Test-Path $cacheDir) {
    $shell4 = New-Object -ComObject Shell.Application
    $folder = $shell4.NameSpace("D:\Hermes\models")
    $file = $folder.ParseName(".cache")
    if ($file) { $file.InvokeVerb("delete") }
    Write-Host "  [x] models\.cache\ (huggingface download artifacts)" -ForegroundColor Green
}

Write-Host ""
Write-Host "=== Cleanup logs/ ===" -ForegroundColor Cyan
$logsDir = "D:\Hermes\logs"
if (Test-Path $logsDir) {
    $shell5 = New-Object -ComObject Shell.Application
    $folder = $shell5.NameSpace("D:\Hermes")
    $file = $folder.ParseName("logs")
    if ($file) { $file.InvokeVerb("delete") }
    Write-Host "  [x] logs\ folder" -ForegroundColor Green
}

Write-Host ""
Write-Host "=== Cleanup temp .ps1 from earlier setup ===" -ForegroundColor Cyan
$tempPs1 = @(
    "D:\Hermes\update_icons.ps1"
)
foreach ($f in $tempPs1) {
    if (Test-Path $f) {
        $shell6 = New-Object -ComObject Shell.Application
        $folder = $shell6.NameSpace((Split-Path $f -Parent))
        $file = $folder.ParseName((Split-Path $f -Leaf))
        if ($file) { $file.InvokeVerb("delete") }
        Write-Host "  [x] $(Split-Path $f -Leaf)" -ForegroundColor Green
    }
}
