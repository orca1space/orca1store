"""
Create Windows desktop shortcuts for Hermes.
Requires: pywin32 (or uses PowerShell COM as fallback).
"""
import os
import sys
import shutil
from pathlib import Path

HERMES_ROOT = Path(r"C:\Users\Yahia\.minimax\workspace\hermes")
DESKTOP = Path(os.environ["USERPROFILE"]) / "Desktop"
ICO_FILE = HERMES_ROOT / "assets" / "orca_icon.ico"
LOGO_PNG = HERMES_ROOT / "assets" / "orca_logo.png"


def create_shortcut_via_powershell(target, shortcut_path, icon_path, description):
    """Create a .lnk file using PowerShell COM (no extra deps)."""
    ps_script = f'''
$ws = New-Object -ComObject WScript.Shell
$sc = $ws.CreateShortcut("{shortcut_path}")
$sc.TargetPath = "{target}"
$sc.WorkingDirectory = "{HERMES_ROOT}"
$sc.IconLocation = "{icon_path}"
$sc.Description = "{description}"
$sc.WindowStyle = 1
$sc.Save()
Write-Output "OK: {shortcut_path}"
'''
    import subprocess
    result = subprocess.run(
        ["powershell.exe", "-NoProfile", "-NonInteractive", "-Command", ps_script],
        capture_output=True, text=True, timeout=30,
    )
    return result.returncode == 0, result.stdout, result.stderr


def main():
    print("=" * 60)
    print(" Hermes — Desktop Shortcut Creator")
    print("=" * 60)
    print(f"Desktop: {DESKTOP}")
    print(f"Hermes root: {HERMES_ROOT}")
    print(f"Icon: {ICO_FILE} (exists: {ICO_FILE.exists()})")
    print()

    if not DESKTOP.exists():
        print(f"ERROR: Desktop not found at {DESKTOP}")
        sys.exit(1)

    if not ICO_FILE.exists():
        print(f"ERROR: Icon not found at {ICO_FILE}")
        sys.exit(1)

    shortcuts = [
        {
            "name": "Hermes - Chat.lnk",
            "target": str(HERMES_ROOT / "hermes.bat"),
            "description": "Hermes — Local AI Agent (Chat Mode)",
        },
        {
            "name": "Hermes - Train.lnk",
            "target": str(HERMES_ROOT / "train-hermes.bat"),
            "description": "Hermes — Training Mode (teach the agent)",
        },
    ]

    for sc in shortcuts:
        shortcut_path = DESKTOP / sc["name"]
        print(f"Creating: {sc['name']}")
        ok, out, err = create_shortcut_via_powershell(
            target=sc["target"],
            shortcut_path=str(shortcut_path),
            icon_path=str(ICO_FILE),
            description=sc["description"],
        )
        if ok:
            print(f"  ✅ {shortcut_path}")
            if out.strip():
                print(f"     {out.strip()}")
        else:
            print(f"  ❌ FAILED: {err.strip() or 'unknown error'}")

    # Also copy the full logo to desktop for reference
    if LOGO_PNG.exists():
        dest = DESKTOP / "Hermes-Logo.png"
        shutil.copy2(LOGO_PNG, dest)
        print(f"\n  ✅ Logo copied to: {dest}")

    print("\n" + "=" * 60)
    print(" Done! Check your desktop.")
    print("=" * 60)


if __name__ == "__main__":
    main()
