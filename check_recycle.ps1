$shell = New-Object -ComObject Shell.Application
$rb = $shell.NameSpace(0xA)
if ($rb) {
    $count = 0
    $size = 0
    $rb.Items() | ForEach-Object {
        $count++
        $size += $_.Size
        Write-Output ("  Folder: {0} ({1:N1} MB)" -f $_.Name, ($_.Size / 1MB))
    }
    Write-Output ""
    Write-Output ("Total in Recycle Bin: {0} item(s), {1:N1} MB" -f $count, ($size / 1MB))
} else {
    Write-Output "Could not access Recycle Bin"
}
