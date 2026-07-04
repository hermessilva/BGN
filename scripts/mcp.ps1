# Launch the .mass model-editing MCP server (TCP, newline-delimited JSON-RPC).
#
# Usage:
#   powershell -ExecutionPolicy Bypass -File scripts\mcp.ps1 <project.mass> [port]
# Default port 8766. Connect an MCP client (or the Arena editor's bridge) and call
# tools: describe_model, get_node, scale_bone, rotate_joint, generate_fingers, ...
param([Parameter(Mandatory=$true)][string]$Project, [int]$Port = 8766,
      [ValidateSet("Release","Debug")][string]$Config = "Release")

$ErrorActionPreference = "Stop"
$root = Split-Path -Parent $PSScriptRoot
$rel  = "$root\build\mcp\$Config"
$vbin = "D:\Tootega\Source\MASS\Deps\vcpkg\installed\x64-windows\bin"

if (-not (Test-Path "$rel\gaitnet-mcp.exe")) { throw "gaitnet-mcp.exe not found; run scripts\build.ps1 first." }
if ((Test-Path "$vbin\tinyxml2.dll") -and -not (Test-Path "$rel\tinyxml2.dll")) { Copy-Item "$vbin\tinyxml2.dll" $rel }

& "$rel\gaitnet-mcp.exe" $Project $Port
