$ws = New-Object -ComObject WScript.Shell
$sc = $ws.CreateShortcut("D:\OneDrive\Desktop\Hermes - Train.lnk")
$sc.TargetPath = "C:\Users\Yahia\.minimax\workspace\hermes\train-hermes.bat"
$sc.WorkingDirectory = "C:\Users\Yahia\.minimax\workspace\hermes"
$sc.IconLocation = "C:\Users\Yahia\.minimax\workspace\hermes\assets\orca_icon.ico"
$sc.Description = "Hermes - Training Mode"
$sc.WindowStyle = 1
$sc.Save()
Write-Output "Train: OK"
