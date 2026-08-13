$dest = 'D:\OneDrive\Desktop'
Get-ChildItem "$dest\Hermes - *.lnk" -ErrorAction SilentlyContinue | ForEach-Object {
    Remove-Item $_.FullName -Force
    Write-Output "Removed: $($_.Name)"
}
