# mcp — .mass model-editing MCP server

Pure C++17 model-editing library + MCP server over the BidirectionalGaitNet
`.mass` project. No DART, no Python — only nlohmann/json, tinyxml2 (`.osim` atlas)
and Boost.Asio (TCP transport). Copied/adapted from MASS-Easy's `libmassedit` and
re-targeted to this project's `.mass` schema (adds the GaitNet `env.xml` config,
the `<parameter>` block and per-joint `kp`/`kv`, which round-trip losslessly).

## Modules
| File | Role |
|---|---|
| `MassModel.{h,cpp}` | `Model` struct + `.mass` JSON IO + stable `uid` (`assignUids`) |
| `Index.{h,cpp}` | generational-handle ids, reverse indices, group selector |
| `Query.{h,cpp}` | JSON facade for the read tools |
| `Kinematics.*` | FK: rotate joint, propagate subtree, re-anchor waypoints |
| `DofMap.*` | anatomical DOF layer (name → joint/axis/sign/range) |
| `Batch.*` | `scale_bone`, `translate_subtree`, L/R symmetric |
| `Complete.*` | finger/phalanx generation, `list_gaps`, symmetric fill |
| `Atlas.*` | OpenSim `.osim` parse, normalized join, `validate`, `sync` |
| `Groom.*` | hair groom params + PBD guide solver |
| `Mcp.*` | MCP JSON-RPC dispatch (`McpServer`) + single-writer queue (`McpQueue`) |

## Tools (`tools/call`)
`describe_model`, `get_node`, `get_muscle`, `select`, `muscles_of_body`,
`muscles_crossing_joint`, `scale_bone`, `translate_subtree`, `rotate_joint`,
`generate_fingers`, `list_gaps`, `load_atlas`, `validate_anatomy`,
`sync_from_atlas`, `save`, `load`. Mutating tools rebuild the index.

## Build & run
Built by the top-level CMake (`GAITNET_BUILD_MCP=ON`) → `build/mcp/<Config>/gaitnet-mcp.exe`.

```powershell
powershell -ExecutionPolicy Bypass -File scripts\build.ps1
powershell -ExecutionPolicy Bypass -File scripts\mcp.ps1 data\project.mass 8766
```

The server speaks newline-delimited JSON-RPC over TCP:
`initialize`, `tools/list`, `tools/call`. Example:
`{"jsonrpc":"2.0","id":1,"method":"tools/call","params":{"name":"describe_model","arguments":{}}}`

## In-process Arena bridge (next)
`McpQueue` is the co-edit mechanism: any thread `submit`s a request and gets a
`std::future`; the owner (Arena's UI thread) `drain`s once per frame, applying
every request through `McpServer` on that single thread — no model locks. Wiring
this into the Arena editor (Asio transport → queue → drain in `App::frame`) lets
an AI edit the live model while it is open. Not yet wired here.
