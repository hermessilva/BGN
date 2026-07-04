# Launch the BidirectionalGaitNet viewer on Windows.
#
# Usage:
#   powershell -ExecutionPolicy Bypass -File scripts\view.ps1                       # loads data\env.xml
#   powershell -ExecutionPolicy Bypass -File scripts\view.ps1 -Arg ..\fgn\<network> # loads a trained network
param(
    [string]$Arg,
    [ValidateSet("Release","Debug")][string]$Config = "Release"
)

$ErrorActionPreference = "Stop"
$root = Split-Path -Parent $PSScriptRoot
$rel  = "$root\build\viewer\$Config"
$vbin = "D:\Tootega\Source\MASS\Deps\vcpkg\installed\x64-windows\bin"
$venv = "D:\Tootega\Source\MASS\Deps\venv\Lib\site-packages"
$pyBase = "C:\Users\Hermes\AppData\Local\Programs\Python\Python310"

if (-not (Test-Path "$rel\viewer.exe")) { throw "viewer.exe not found; run scripts\build.ps1 first." }

# Make sure the runtime DLLs the applocal deploy can miss are next to the exe.
foreach ($d in @("$vbin\freeglut.dll", "$pyBase\python310.dll")) {
    $name = Split-Path $d -Leaf
    if ((Test-Path $d) -and -not (Test-Path "$rel\$name")) { Copy-Item $d $rel }
}
New-Item -ItemType Directory -Path "$root\screenshots" -Force | Out-Null

if (-not $Arg) { $Arg = "$root\data\env.xml" }

$env:KMP_DUPLICATE_LIB_OK = "TRUE"     # torch + DART both load OpenMP
$env:PYTHONPATH = $venv                # torch for the embedded interpreter
$env:PATH = "$rel;$env:PATH"

# The viewer resolves data/networks via ..\ paths, so run from build\.
Set-Location "$root\build"
& "$rel\viewer.exe" $Arg
