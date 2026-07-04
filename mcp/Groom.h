#pragma once
// libmassedit — hair/fur grooms. Two-tier design (see Docs/MCP-Study.md):
//   * GroomParams: the authored controls persisted in .mass (a handful drive
//     millions of render strands; strands themselves are never stored).
//   * HairSim: the dynamic tier — a Position-Based-Dynamics solver on a single
//     guide curve (root pinned, gravity/wind, inextensible segments, damping).
//     Physics runs on ~hundreds of guides; render strands interpolate from them.
#include "MassModel.h"
#include <vector>

namespace mass {

// GroomParams (the authored, persisted controls) is declared in MassModel.h.

// PBD solver for one guide curve.
class HairSim {
public:
    // Lay out `segments` links of `segLen` from `root` along unit `dir`.
    void init(const Vec3& root, const Vec3& dir, int segments, double segLen);

    // Advance one step: apply (gravity+wind)*dt to free particles, damp, integrate,
    // pin the root, then project the inextensibility constraints `iters` times.
    void step(double dt, const Vec3& gravity, const Vec3& wind, double damping, int iters = 16);

    const std::vector<Vec3>& points() const { return mP; }
    Vec3 tip() const { return mP.empty() ? Vec3{0,0,0} : mP.back(); }
    double segLength(int i) const;   // current length of link i
    double restLen() const { return mRest; }
    int links() const { return mP.empty() ? 0 : (int)mP.size() - 1; }

private:
    std::vector<Vec3> mP, mPrev;
    double mRest = 0.0;
};

} // namespace mass
