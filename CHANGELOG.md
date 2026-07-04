# Changelog

All notable changes made in this fork relative to the upstream
[namjohn10/BidirectionalGaitNet](https://github.com/namjohn10/BidirectionalGaitNet)
are documented here. Upstream code, models and data remain the work of the original
authors (Jungnam Park, Moon Seok Park, Jehee Lee, Jungdam Won).

The format is loosely based on [Keep a Changelog](https://keepachangelog.com/).

## [Unreleased] — Python-free inference (pure C++/Eigen)

Removed the runtime dependency on Python/PyTorch entirely. The simulation and the
viewer now evaluate every neural network in C++ (Eigen); the app runs with no
Python interpreter, no `pybind11`, and no `torch` on the machine. The project
targets real-time inference with the shipped pre-trained networks, so the offline
training code (which is what needed Python) was dropped.

### Added
- `sim/NN.{h,cpp}` — pure-Eigen inference for all networks: `MuscleNN`
  (two-level muscle control), `PolicyNN` (SimulationNN actor + `weight_filter`),
  `RefNN` (Forward GaitNet) and `GaitVAE` (Backward GaitNet), plus a minimal
  `safetensors` reader (JSON header via nlohmann_json + f32 blobs).
- `tools/export_weights.py` — one-time migration tool. Converts the shipped
  pickled ray/torch checkpoints (`fgn/…`, `bgn/…`, `data/trained_nn/…`) into
  `.safetensors` the C++ loads at runtime. The ray "worker" blob is read without
  ray installed via a permissive stub Unpickler. The converted `.safetensors`
  are committed alongside the originals.
- Committed `*.safetensors` for fgn, bgn and the four cascading policies.

### Changed
- `sim/Environment.{h,cpp}`, `sim/Character.h`: dropped the embedded Python
  interpreter; `Network::joint/muscle` and `mMuscleNN` are now `nn::*` objects;
  muscle forward, cascading `get_action`/`weight_filter` and metadata loading call
  the C++ NN. `setMuscleNetwork` takes an `nn::MuscleNN*`.
- `viewer/`: `get_action`, `loading_network`/`loading_metadata`, Forward/Backward
  GaitNet load + `render_forward` all go through `sim/NN`. `viewer/main.cpp` no
  longer starts a `scoped_interpreter`.
- Build: `sim` links `nlohmann_json` instead of `pybind11::embed`; `viewer` no
  longer links pybind11; the `python/` pysim module was removed from the build.

### Verified
- C++/Eigen forward passes match the original torch modules to float precision
  (max abs diff ≈ 1e-6 for MuscleNN, PolicyNN and RefNN).
- Viewer runs and renders with `python310.dll` deleted and no `PYTHONPATH`.

### Removed
- The entire `python/` package (pysim binding `RayEnvManager.cpp`, PPO/GaitNet
  training scripts, ray glue, SLURM launchers). Model creation now requires
  re-introducing a training path; running the shipped models does not.

### Not yet ported
- **C3D import** (`viewer/C3D_Reader.cpp`) used the Python `c3dTobvh` loader and is
  disabled (`#if 0`); a C++ C3D reader is needed to re-enable it.
- Loading pre-rendered `.npz` motion sets in the viewer (a training-data UI) was
  removed; motions can still be captured live via "Add Current Simulation motion".

## [Earlier] — Native Windows port (MSVC + vcpkg)

Ported the project from its Linux-only toolchain (GCC, EGL, ray/rllib 1.8, Python 3.6)
to build and run natively on Windows with Visual Studio 2026 (MSVC 14.51), CMake and
vcpkg, on Python 3.10. Simulation core, the `pysim` Python binding, and the
OpenGL/ImGui viewer build and run; the Bidirectional GaitNet training/data pipeline
runs on Python 3.10. The Generative GaitNet PPO trainer is deferred (see below).

### Added
- `scripts/build.ps1` — configure + build (sim, pysim, viewer) via MSVC + vcpkg;
  flags `-NoViewer`, `-Fresh`, `-Config`.
- `scripts/view.ps1` — launch the viewer with the correct working directory, runtime
  DLLs and environment.
- `README-Windows.md` — Windows build/run guide and full change list.
- `CHANGELOG.md` — this file.
- Fork/attribution + license notice in `README.md`.

### Changed — build system
- `CMakeLists.txt` (root, `sim`, `python`, `libs`, `viewer`): `cmake_minimum_required`
  bumped to 3.16, C++17, MSVC guards (`/bigobj`, `/permissive-`, `NOMINMAX`,
  `_USE_MATH_DEFINES`, `_CRT_SECURE_NO_WARNINGS`), GCC-only flags (`-fPIC`,
  `-fvisibility=hidden`, `stdc++fs`) guarded behind `if(NOT MSVC)`.
- vcpkg CONFIG targets: `dart`, `dart-gui`, `dart-collision-bullet`, `dart-utils`,
  `tinyxml2::tinyxml2`, `pybind11::embed`, `glfw`, `FreeGLUT::freeglut`,
  `OpenGL::GL/GLU`, `assimp::assimp`. `pysim` built with `pybind11_add_module`.
- Viewer/GLUT/EGL gated behind the new `GAITNET_BUILD_VIEWER` option (off for
  `SERVER_BUILD`).
- `libs/CMakeLists.txt`: dropped the Linux-only EGL / `glad_egl` requirement; imgui
  built with `IMGUI_IMPL_OPENGL_LOADER_GLAD`.

### Changed — C++ sources
- `sim/DARTHelper.{h,cpp}`: `std::experimental::filesystem` → `std::filesystem`; OBJ
  meshes loaded via `MeshShape::loadMesh(Uri::createFromPath(path), LocalResourceRetriever)`
  rooted at `MASS_ROOT_DIR`, so Windows drive letters are not mis-parsed as a URI scheme.
- `sim/Muscle.cpp`: added `<numeric>` / `<algorithm>` (MSVC does not include them transitively).
- `sim/Character.cpp`: replaced `dart::math::clip(vec, Zero, Ones)` (Eigen expression
  template deduction failed on MSVC) with `cwiseMax(0).cwiseMin(1)`.
- `viewer/GLFWApp.{h,cpp}`, `viewer/GLfunctions.cpp`: include `<windows.h>` before GL
  headers (APIENTRY/WINGDIAPI); use glad in place of `<GL/gl.h>`; screenshots via
  `stb_image_write` instead of DART's bundled lodepng; guard `directory_iterator` on
  optional folders (`fgn`, `bgn`, `c3d`, `motions`).
- `viewer/main.cpp`: keep the pybind11 `scoped_interpreter` alive across the top-level
  `try`/`catch` (destroying it during unwinding crashed the handler in
  `PyGILState_Ensure`); inject `MASS_ROOT_DIR/python` and the build output dir into
  `sys.path`.

### Changed — Python
- `import pickle5 as pickle` → `try: import pickle5 … except ImportError: import pickle`
  across all scripts (pickle5 is a Python 3.6/3.7 backport; stdlib `pickle` on 3.8+).
- `ray_model.py`: ray/rllib made **optional** — a lightweight `TorchModelV2` stand-in
  and a local `convert_to_torch_tensor` are used when ray is not installed, so the
  embedded interpreter, the viewer and the GaitNet training import without ray.
- `forward_gaitnet.py`, `advanced_vae.py`, `train_forward_gaitnet.py`,
  `train_backward_gaitnet.py`: import `convert_to_torch_tensor` from the `ray_model`
  shim instead of `ray.rllib.utils.torch_ops`.
- `train_backward_gaitnet.py`: removed `from symbol import parameters` (the `symbol`
  module was removed in Python 3.10 and the import was unused).
- `advanced_vae.py`: `umap` / `matplotlib` imported lazily (only inside the plotting
  method) so training does not require the heavy visualization stack.

### Deferred
- Generative GaitNet **PPO training** (`ray_train.py`, `ray_ppo.py`,
  `ray_torch_policy.py`, `ray_env.py`) still targets the ray/rllib **1.8** execution-plan
  API (`build_trainer`, `with_common_config`, `ParallelRollouts`, `TrainOneStep`,
  `StandardMetricsReporting`, `build_policy_class`, `ray.util.iter.LocalIterator`,
  `import gym`), all removed in ray 2.x. Running it requires either a legacy
  Python 3.8 + ray 1.8 environment or a rewrite against ray 2.x (Algorithm/Learner/
  RLModule + gymnasium).
