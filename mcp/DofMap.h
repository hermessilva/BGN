#pragma once
// libmassedit — anatomical DOF layer. Maps an intuitive named degree of freedom
// (e.g. "elbow_R.flex") to a joint + local rotation axis + sign + range, so
// commands speak anatomy instead of raw joint axes/radians.
#include "MassModel.h"
#include "Index.h"
#include <string>
#include <unordered_map>

namespace mass {

struct Dof {
    std::string name;        // e.g. "elbow_R.flex"
    std::string childBone;   // bone whose joint this DOF drives
    Vec3 axis{0,0,1};        // rotation axis in the joint's LOCAL frame
    double sign = 1.0;       // orientation of positive motion
    double loRad = 0.0;      // angle at value 0
    double hiRad = 0.0;      // angle at value 1
};

class DofMap {
public:
    void add(const Dof& d) { mDofs[d.name] = d; }
    const Dof* find(const std::string& name) const {
        auto it = mDofs.find(name); return it != mDofs.end() ? &it->second : nullptr;
    }
    // World-space axis of a DOF given the current joint frame (local axis mapped
    // through the child bone's joint linear).
    Vec3 worldAxis(const Model& m, const Index& ix, const std::string& name) const;

    // Rotate the DOF's joint by `angleRad` (times its sign) about the current
    // world axis. Relative delta. False if the DOF/bone is unknown.
    bool applyAngle(Model& m, const Index& ix, const std::string& name, double angleRad) const;

    // Map value in [0,1] to [loRad,hiRad] and apply it as a rotation (times sign).
    bool applyValue(Model& m, const Index& ix, const std::string& name, double v01) const;

    size_t size() const { return mDofs.size(); }

private:
    std::unordered_map<std::string, Dof> mDofs;
};

} // namespace mass
