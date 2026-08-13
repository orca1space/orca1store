$shell = New-Object -ComObject WScript.Shell
$dest = "D:\OneDrive\Desktop"
$hermesRoot = "D:\Hermes"
$ico = "$hermesRoot\assets\hermes_icon.ico"

# Remove old 3 shortcuts/items
Get-ChildItem "$dest\Hermes*" -ErrorAction SilentlyContinue | ForEach-Object {
    $appShell = New-Object -ComObject Shell.Application
    $folder = $appShell.NameSpace($_.DirectoryName)
    $item = $folder.ParseName($_.Name)
    if ($item) { $item.InvokeVerb("delete") }
    Write-Output "Removed: $($_.Name)"
}

# Create ONE shortcut using WScript.Shell
$shortcut = $shell.CreateShortcut("$dest\OrcaMax Code.lnk")
$shortcut.TargetPath = "$hermesRoot\OrcaMax.bat"
$shortcut.WorkingDirectory = "$hermesRoot"
$shortcut.IconLocation = "$ico"
$shortcut.Description = "OrcaMax Code - Local AI Assistant"
$shortcut.WindowStyle = 1
$shortcut.Save()
Write-Output "Created: OrcaMax Code.lnk"

# Verify
Get-ChildItem "$dest\OrcaMax*" | Select-Object Name, Length, LastWriteTime | Format-Table -AutoSize
