#pragma once
// libmassedit — bulk skeleton edits. The core primitive scales a bone along an
// axis about its proximal (parent) joint: the bone's own geometry + waypoints
// scale along that axis, while each child subtree keeps its size and translates
// rigidly by the displacement of its attachment. This is the "lengthen the
// femur / shorten the fingers" operation that touches many elements at once.
#include "MassModel.h"
#include "Index.h"
#include <string>

namespace mass {

struct Batch {
    // Scale `bone` by `factor` along world direction `axisWorld`, about the bone's
    // proximal joint center. Scales the bone body (position + Box size along the
    // axis) and its own waypoints; translates every child subtree (+ their
    // waypoints) rigidly by their attachment displacement. False if bone unknown.
    static bool scaleBone(Model& m, const Index& ix, const std::string& bone,
                          const Vec3& axisWorld, double factor);

    // Translate a bone's whole subtree (bodies + joints + waypoints) by `delta`.
    static bool translateSubtree(Model& m, const Index& ix, const std::string& bone,
                                 const Vec3& delta);

    // L/R counterpart of a bone/muscle name ("FemurR"<->"FemurL", "R_X"<->"L_X").
    // Returns "" if the name has no side.
    static std::string counterpartName(const std::string& name);

    // Apply scaleBone to `bone` and to its L/R counterpart (if present), with the
    // axis X-component negated for the counterpart (sagittal mirror). Returns the
    // number of bones actually scaled (0, 1, or 2).
    static int scaleBoneSymmetric(Model& m, const Index& ix, const std::string& bone,
                                  const Vec3& axisWorld, double factor);
};

} // namespace mass
