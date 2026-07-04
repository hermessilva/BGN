#pragma once
// libmassedit — forward kinematics over mass::Model.
// Rotating a joint rigidly transforms its whole bone subtree (bodies + their
// joints) and re-anchors every muscle waypoint on those bodies, using the Index
// for the subtree and reverse waypoint lookup. Kinematic only (no DART).
#include "MassModel.h"
#include "Index.h"
#include <string>

namespace mass {

// ---- small row-major 3x3 math (independent of Arena's Math.h) ----
Vec3 mv(const Mat3& A, const Vec3& x);          // A * x
Mat3 mm(const Mat3& A, const Mat3& B);          // A * B
Mat3 axisAngle(const Vec3& axis, double angle); // Rodrigues -> rotation matrix
Vec3 normalize(const Vec3& v);

struct Kinematics {
    // Rotate the joint of `childBone` by `angle` (radians) about `axisWorld`
    // through the joint's current world center. Transforms subtree(childBone)
    // rigidly and re-anchors the muscle waypoints anchored to those bones.
    // Returns false if the bone is unknown.
    static bool rotateJoint(Model& m, const Index& ix, const std::string& childBone,
                            const Vec3& axisWorld, double angle);
};

} // namespace mass
