#pragma once
// A rigged (skinned) character loaded from FBX/GLB via assimp: mesh + UVs + skin
// weights + bone hierarchy + a baked animation clip + diffuse texture. Plays its
// own animation by CPU linear-blend skinning. No DART; used by the editor to show
// an artist-rigged character walking with its own weights.
#include "Math.h"
#include <string>
#include <vector>
#include <array>
#include <map>

namespace ed {

struct RiggedMesh {
    // ---- static (bind-pose) mesh ----
    std::vector<V3> basePos, baseNrm;         // bind-pose vertices
    std::vector<float> uv;                     // 2 per vertex (u,v)
    std::vector<unsigned> indices;             // triangle list
    std::vector<std::array<int, 4>> vBone;     // up to 4 influencing bones per vertex
    std::vector<std::array<float, 4>> vWeight; // matching weights (sum 1)

    // ---- skeleton (nodes) ----
    struct Node { std::string name; M4 local; int parent = -1; int bone = -1; };
    std::vector<Node> nodes;                    // flattened node tree (parent before child)
    std::map<std::string, int> nodeByName;
    M4 globalInverse;                           // inverse(root node transform)

    // ---- bones that skin the mesh ----
    struct Bone { std::string name; M4 offset; int node = -1; }; // offset = inverse bind
    std::vector<Bone> bones;

    // ---- animation clip ----
    struct Key { double t; V3 v; };
    struct QKey { double t; float x, y, z, w; };
    struct Channel { int node = -1; std::vector<Key> pos, scl; std::vector<QKey> rot; };
    std::vector<Channel> channels;
    double animDurTicks = 0, animTps = 25.0;

    // ---- diffuse texture (decoded RGBA8) ----
    int texW = 0, texH = 0; std::vector<unsigned char> texRGBA;

    bool loaded() const { return !basePos.empty(); }
    double animSeconds() const { return animTps > 0 ? animDurTicks / animTps : 0; }

    bool load(const std::string& path, std::string* err = nullptr);
    // skinned pos/nrm at animation time tSec (loops); if no anim, returns bind pose
    void evaluate(double tSec, std::vector<V3>& outPos, std::vector<V3>& outNrm) const;
};

} // namespace ed
