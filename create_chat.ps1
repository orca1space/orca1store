$ws = New-Object -ComObject WScript.Shell
$sc = $ws.CreateShortcut("D:\OneDrive\Desktop\Hermes - Chat.lnk")
$sc.TargetPath = "C:\Users\Yahia\.minimax\workspace\hermes\hermes.bat"
$sc.WorkingDirectory = "C:\Users\Yahia\.minimax\workspace\hermes"
$sc.IconLocation = "C:\Users\Yahia\.minimax\workspace\hermes\assets\orca_icon.ico"
$sc.Description = "Hermes - Local AI Agent (Chat)"
$sc.WindowStyle = 1
$sc.Save()
Write-Output "Chat: OK"
