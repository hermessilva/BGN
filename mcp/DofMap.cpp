#include "DofMap.h"
#include "Kinematics.h"

namespace mass {

Vec3 DofMap::worldAxis(const Model& m, const Index& ix, const std::string& name) const {
    const Dof* d = find(name);
    if (!d) return {0,0,0};
    int c = ix.boneSlot(d->childBone);
    if (c < 0) return {0,0,0};
    return normalize(mv(m.skeleton[c].joint.t.linear, d->axis));
}

bool DofMap::applyAngle(Model& m, const Index& ix, const std::string& name, double angleRad) const {
    const Dof* d = find(name);
    if (!d) return false;
    Vec3 ax = worldAxis(m, ix, name);
    return Kinematics::rotateJoint(m, ix, d->childBone, ax, d->sign * angleRad);
}

bool DofMap::applyValue(Model& m, const Index& ix, const std::string& name, double v01) const {
    const Dof* d = find(name);
    if (!d) return false;
    if (v01 < 0.0) v01 = 0.0; else if (v01 > 1.0) v01 = 1.0;
    double angle = d->loRad + v01 * (d->hiRad - d->loRad);
    return applyAngle(m, ix, name, angle);
}

} // namespace mass
