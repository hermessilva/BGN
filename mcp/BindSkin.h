#pragma once
// Load an external character mesh, fit it to the skeleton and bind each vertex to
// the nearest body (the same rigid nearest-bone scheme the Arena editor renders).
// Stores a Skin descriptor on the model; returns a JSON report of the binding.
#include "MassModel.h"
#include <nlohmann/json.hpp>
#include <string>

namespace mass {

struct BindSkin {
    // When fitBones is true, also morph the skeleton's limb chains to the mesh
    // silhouette (arms/legs) so the bones match the character before binding.
    static nlohmann::json bind(Model& m, const std::string& obj,
                               const Vec3& rotDeg, double userScale, const Vec3& offset,
                               bool fitBones, std::string* err);

    // Resize a single bone's box to the local skin mesh (verts nearest to it),
    // using the Skin descriptor already stored on the model. Keeps the L/R pair
    // symmetric. `margin` inflates the fit; `slack` biases toward/away the box.
    static nlohmann::json fitBone(Model& m, const std::string& bone,
                                  double margin, std::string* err);
};

} // namespace mass
