Add-Type -AssemblyName System.Drawing
$iconExtractor = @{
    SourcePath = "D:\Hermes\assets\minimax_exe_tmp.exe"
    DestPath = "D:\Hermes\assets\minimax_icon.ico"
}
$icon = [System.Drawing.Icon]::ExtractAssociatedIcon($iconExtractor.SourcePath)
if ($icon) {
    $fs = [System.IO.File]::Create($iconExtractor.DestPath)
    $icon.Save($fs)
    $fs.Close()
    Write-Output ("Icon extracted: " + $iconExtractor.DestPath)
    $info = Get-Item $iconExtractor.DestPath
    Write-Output ("Size: " + $info.Length + " bytes")
} else {
    Write-Output "Could not extract icon"
}
