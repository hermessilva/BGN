# Launch the GaitNet visual editor.
#
# Usage:
#   powershell -ExecutionPolicy Bypass -File scripts\editor.ps1                 # empty project
#   powershell -ExecutionPolicy Bypass -File scripts\editor.ps1 <project.mass>  # open a project
# Inside the editor use File > "Import env.xml..." to load data\env.xml.
param([string]$Project, [ValidateSet("Release","Debug")][string]$Config = "Release")

$ErrorActionPreference = "Stop"
$root = Split-Path -Parent $PSScriptRoot
$rel  = "$root\build\editor\$Config"
$vbin = "D:\Tootega\Source\MASS\Deps\vcpkg\installed\x64-windows\bin"

if (-not (Test-Path "$rel\editor.exe")) { throw "editor.exe not found; run scripts\build.ps1 first." }
if ((Test-Path "$vbin\glfw3.dll") -and -not (Test-Path "$rel\glfw3.dll")) { Copy-Item "$vbin\glfw3.dll" $rel }

$env:KMP_DUPLICATE_LIB_OK = "TRUE"
$env:PATH = "$rel;$env:PATH"
Set-Location $rel
if ($Project) { & "$rel\editor.exe" $Project } else { & "$rel\editor.exe" }
