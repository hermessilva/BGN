# Bidirectional GaitNet on Windows (native, MSVC + vcpkg)

Windows-native port of Bidirectional GaitNet (SIGGRAPH 2023). The original code
targets Linux (GCC, EGL, ray/rllib 1.8, Python 3.6). This port builds the C++
simulation, the `pysim` Python binding and the OpenGL/ImGui viewer with MSVC and
vcpkg, and runs the Bidirectional GaitNet training pipeline on Python 3.10.

## Status

The runtime is now **Python-free**: the simulation and viewer evaluate every
neural network in C++ (Eigen), so no Python interpreter, `pybind11` or `torch` is
needed to run. Only the one-time weight-conversion tool uses Python.

| Component | State |
|-----------|-------|
| `sim` C++ library (DART, muscles, Environment, Eigen NN) | builds (MSVC), no Python |
| `viewer` GUI (GLFW + glad + Dear ImGui + ImPlot) | builds and renders; runs with no `python*.dll` |
| C++ NN inference (`sim/NN`): MuscleNN, PolicyNN, RefNN, GaitVAE | matches torch to ~1e-6 |
| `tools/export_weights.py` (pickle → safetensors) | one-time migration (needs numpy) |
| Offline training (PPO / GaitNet VAE) | **removed** — real-time inference uses the shipped pre-trained nets |
| C3D import | disabled (`#if 0`); needs a C++ C3D reader |

## Prerequisites (already provisioned on this machine)

- **Visual Studio 2026** (VS18) with the C++ toolchain (MSVC 14.51).
- **CMake 3.16+**.
- **Python 3.10** at `C:\Users\Hermes\AppData\Local\Programs\Python\Python310`
  (needs `include\` headers and `libs\python310.lib` from the python.org installer).
- **vcpkg** — reused from the sibling MASS project at
  `D:\Tootega\Source\MASS\Deps\vcpkg` (provides `dartsim[collision-bullet,gui,utils]`,
  `tinyxml2`, `pybind11`, `glfw3`, `glad`, `imgui`, `freeglut`, `assimp`, `eigen3`,
  `bullet3` for the `x64-windows` triplet).
- **Python venv** with torch (cu124) + tensorboard, reused from
  `D:\Tootega\Source\MASS\Deps\venv`. (`pip install torch tensorboard numpy`.)

If the vcpkg path differs on your machine, edit `$vcpkg` in `scripts\build.ps1`
and `scripts\view.ps1`.

## Build

```powershell
# sim + viewer (Release)
powershell -ExecutionPolicy Bypass -File scripts\build.ps1

# sim only (headless / no GUI)
powershell -ExecutionPolicy Bypass -File scripts\build.ps1 -NoViewer
```

Produces `build\viewer\Release\viewer.exe`. No Python is involved in the build or
at runtime.

## Run the viewer

```powershell
powershell -ExecutionPolicy Bypass -File scripts\view.ps1                 # loads data\env.xml
powershell -ExecutionPolicy Bypass -File scripts\view.ps1 -Arg ..\fgn\<network>
```

The viewer must run from `build\` because it resolves `..\data`, `..\fgn`, `..\bgn`,
`..\c3d`, `..\motions` relative to the working directory (`view.ps1` handles this).

## Regenerating the network weights (one-time)

The C++ loads network weights from `.safetensors` files (already committed next to
the original checkpoints). To regenerate them from the pickled checkpoints:

```powershell
python tools\export_weights.py   # needs only numpy
```

## Changes made for the Windows port

- **CMake**: `cmake_minimum_required(VERSION 3.16)`, C++17, MSVC guards
  (`/bigobj /permissive-`, `NOMINMAX`, `_USE_MATH_DEFINES`, `_CRT_SECURE_NO_WARNINGS`),
  vcpkg CONFIG targets (`dart`, `dart-gui`, `dart-collision-bullet`, `dart-utils`,
  `tinyxml2::tinyxml2`, `pybind11::embed`, `glfw`, `FreeGLUT::freeglut`,
  `OpenGL::GL/GLU`, `assimp::assimp`). `pybind11_add_module(pysim)`.
  Viewer/GLUT/EGL gated behind `GAITNET_BUILD_VIEWER` (off for `SERVER_BUILD`).
- **`libs/CMakeLists.txt`**: dropped the Linux-only EGL/`glad_egl` requirement; imgui
  built with `IMGUI_IMPL_OPENGL_LOADER_GLAD`.
- **`sim/DARTHelper`**: `std::experimental::filesystem` → `std::filesystem`; OBJ meshes
  loaded via `MeshShape::loadMesh(Uri::createFromPath(path), LocalResourceRetriever)`
  and `MASS_ROOT_DIR` so Windows drive letters are not mis-parsed as a URI scheme.
- **`sim/Muscle.cpp`**: added `<numeric>`/`<algorithm>` (MSVC does not pull them in).
- **`sim/Character.cpp`**: replaced a `dart::math::clip(vec, Zero, Ones)` (Eigen
  expression deduction failed on MSVC) with `cwiseMax(0).cwiseMin(1)`.
- **`viewer`**: include `<windows.h>` before GL headers (APIENTRY/WINGDIAPI); use glad
  in place of `<GL/gl.h>`; screenshot via `stb_image_write` instead of DART's bundled
  lodepng; guard `directory_iterator` on optional folders (`fgn/bgn/c3d/motions`);
  keep the `scoped_interpreter` alive across the top-level catch and inject
  `MASS_ROOT_DIR/python` into `sys.path`.
- **Python**: `import pickle5 as pickle` → try/except stdlib `pickle`; `ray_model`
  makes ray/rllib optional (stub `TorchModelV2` + local `convert_to_torch_tensor`);
  training/pipeline scripts import that shim instead of `ray.rllib...`; removed the
  py3.10-incompatible `from symbol import parameters`; lazy umap/matplotlib in
  `advanced_vae`.

## Removed: offline training

The whole `python/` package was removed when the runtime went Python-free. The
project runs the **pre-trained** networks shipped in the repo (`fgn/`, `bgn/`,
`data/trained_nn/`) via real-time C++ inference, which needs no training. Creating
new models again would mean re-introducing a training path (the original one used
ray/rllib 1.8, whose API was removed in ray 2.x — see `CHANGELOG.md`). The C++
inference itself is validated to match the original torch modules to ~1e-6.
