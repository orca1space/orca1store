Add-Type -AssemblyName Microsoft.VisualBasic
$shell = New-Object -ComObject Shell.Application

$paths = @(
    "C:\Users\Yahia\.minimax\workspace\hermes",
    "C:\Users\Yahia\.minimax\workspace\hermes-code.zip"
)

foreach ($p in $paths) {
    if (Test-Path $p) {
        $folder = $shell.NameSpace((Split-Path $p -Parent))
        $item = $folder.ParseName((Split-Path $p -Leaf))
        if ($item) {
            $item.InvokeVerb("delete")
            Write-Output "Sent to Recycle Bin: $p"
        } else {
            Write-Output "Could not parse: $p"
        }
    } else {
        Write-Output "Not found: $p"
    }
}
