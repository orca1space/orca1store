$ws = New-Object -ComObject WScript.Shell
$sc = $ws.CreateShortcut("D:\OneDrive\Desktop\Hermes - Chat.lnk")
$sc.TargetPath = "D:\Hermes\hermes.bat"
$sc.WorkingDirectory = "D:\Hermes"
$sc.IconLocation = "D:\Hermes\assets\orca_icon.ico"
$sc.Description = "Hermes - Local AI Agent (Chat)"
$sc.WindowStyle = 1
$sc.Save()
Write-Output "Chat: OK"

$sc2 = $ws.CreateShortcut("D:\OneDrive\Desktop\Hermes - Train.lnk")
$sc2.TargetPath = "D:\Hermes\train-hermes.bat"
$sc2.WorkingDirectory = "D:\Hermes"
$sc2.IconLocation = "D:\Hermes\assets\orca_icon.ico"
$sc2.Description = "Hermes - Training Mode"
$sc2.WindowStyle = 1
$sc2.Save()
Write-Output "Train: OK"
