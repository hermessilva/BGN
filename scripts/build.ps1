# Configure + build BidirectionalGaitNet on Windows (MSVC + vcpkg).
#
# Usage:
#   powershell -ExecutionPolicy Bypass -File scripts\build.ps1               # Release, viewer on
#   powershell -ExecutionPolicy Bypass -File scripts\build.ps1 -Config Debug
#   powershell -ExecutionPolicy Bypass -File scripts\build.ps1 -NoViewer     # sim + pysim only
#   powershell -ExecutionPolicy Bypass -File scripts\build.ps1 -Fresh        # wipe build/ first
param(
    [ValidateSet("Release","Debug")][string]$Config = "Release",
    [switch]$NoViewer,
    [switch]$Fresh
)

$ErrorActionPreference = "Stop"
$root   = Split-Path -Parent $PSScriptRoot
$build  = "$root\build"

# Reuse the vcpkg installation provisioned for the sibling MASS project
# (dartsim, tinyxml2, pybind11, glfw3, glad, imgui, imguizmo, boost-asio, ...).
$vcpkg  = "D:\Tootega\Source\MASS\Deps\vcpkg"
$pyBase = "C:\Users\Hermes\AppData\Local\Programs\Python\Python310"

if (-not (Test-Path "$vcpkg\scripts\buildsystems\vcpkg.cmake")) {
    throw "vcpkg toolchain not found at $vcpkg. Adjust `$vcpkg in this script."
}

if ($Fresh -and (Test-Path $build)) { Remove-Item $build -Recurse -Force }

$viewer = if ($NoViewer) { "OFF" } else { "ON" }

if (-not (Test-Path "$build\CMakeCache.txt")) {
    Write-Host "[configure] generating build/ (viewer=$viewer)" -ForegroundColor Cyan
    cmake -S $root -B $build -G "Visual Studio 18 2026" -A x64 `
        -DCMAKE_TOOLCHAIN_FILE="$vcpkg\scripts\buildsystems\vcpkg.cmake" `
        -DVCPKG_TARGET_TRIPLET=x64-windows `
        -DGAITNET_BUILD_VIEWER=$viewer `
        -DPython_EXECUTABLE="$pyBase\python.exe" `
        -DPython_INCLUDE_DIR="$pyBase\include" `
        -DPython_LIBRARY="$pyBase\libs\python310.lib"
    if ($LASTEXITCODE -ne 0) { throw "configure failed" }
}

Write-Host "[build] $Config" -ForegroundColor Cyan
cmake --build $build --config $Config
if ($LASTEXITCODE -ne 0) { throw "build failed" }

Write-Host "`n[done] $Config build complete" -ForegroundColor Green
