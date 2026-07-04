#include "Batch.h"
#include "Kinematics.h"   // normalize, Vec3 helpers
#include <cmath>

namespace mass {

static double dot(const Vec3& a, const Vec3& b) { return a[0]*b[0]+a[1]*b[1]+a[2]*b[2]; }
static Vec3 add(const Vec3& a, const Vec3& b) { return { a[0]+b[0], a[1]+b[1], a[2]+b[2] }; }
static Vec3 sub(const Vec3& a, const Vec3& b) { return { a[0]-b[0], a[1]-b[1], a[2]-b[2] }; }
static Vec3 mul(const Vec3& a, double s) { return { a[0]*s, a[1]*s, a[2]*s }; }

// local direction of a world axis given a (near-orthonormal) row-major linear:
// col i of R is the world image of local axis i, so localDir[i] = dot(col_i, d).
static Vec3 worldToLocalDir(const Mat3& R, const Vec3& d) {
    return { R[0]*d[0]+R[3]*d[1]+R[6]*d[2],
             R[1]*d[0]+R[4]*d[1]+R[7]*d[2],
             R[2]*d[0]+R[5]*d[1]+R[8]*d[2] };
}

bool Batch::translateSubtree(Model& m, const Index& ix, const std::string& bone, const Vec3& delta) {
    if (ix.boneSlot(bone) < 0) return false;
    for (int b : ix.subtreeBones(bone)) {
        Node& n = m.skeleton[b];
        n.body.t.translation  = add(n.body.t.translation, delta);
        n.joint.t.translation = add(n.joint.t.translation, delta);
        for (const WpRef& w : ix.waypointsOfBone(n.id))
            m.muscles[w.muscle].waypoints[w.wp].p = add(m.muscles[w.muscle].waypoints[w.wp].p, delta);
    }
    return true;
}

bool Batch::scaleBone(Model& m, const Index& ix, const std::string& bone,
                      const Vec3& axisWorldIn, double factor) {
    int bi = ix.boneSlot(bone);
    if (bi < 0) return false;
    const Vec3 d = normalize(axisWorldIn);
    const Vec3 p = m.skeleton[bi].joint.t.translation;   // proximal (parent) attach

    // scale a point along d about p by `factor`
    auto S = [&](const Vec3& x) {
        Vec3 r = sub(x, p);
        double along = dot(r, d);
        return add(add(p, r), mul(d, (factor - 1.0) * along));  // p + r + (f-1)*along*d
    };

    Node& bn = m.skeleton[bi];

    // 1. children: move each child attachment along d, translate its subtree rigidly
    for (int cb = 0; cb < ix.boneCount(); cb++) {
        if (ix.parentOf(cb) != bi) continue;
        const std::string& childName = m.skeleton[cb].id;
        Vec3 oldJ = m.skeleton[cb].joint.t.translation;
        Vec3 delta = sub(S(oldJ), oldJ);
        translateSubtree(m, ix, childName, delta);
    }

    // 2. this bone's own body: scale center along d about p
    bn.body.t.translation = S(bn.body.t.translation);
    // Box size: scale the local component most aligned with d
    if (bn.body.type == "Box") {
        Vec3 dl = worldToLocalDir(bn.body.t.linear, d);
        int k = 0; double best = std::fabs(dl[0]);
        if (std::fabs(dl[1]) > best) { best = std::fabs(dl[1]); k = 1; }
        if (std::fabs(dl[2]) > best) { k = 2; }
        bn.body.size[k] *= factor;
    }

    // 3. this bone's own waypoints scale along d about p
    for (const WpRef& w : ix.waypointsOfBone(bone))
        m.muscles[w.muscle].waypoints[w.wp].p = S(m.muscles[w.muscle].waypoints[w.wp].p);

    return true;
}

std::string Batch::counterpartName(const std::string& name) {
    if (name.empty()) return "";
    if (name.back() == 'R') return name.substr(0, name.size()-1) + "L";
    if (name.back() == 'L') return name.substr(0, name.size()-1) + "R";
    if (name.rfind("R_", 0) == 0) return "L_" + name.substr(2);
    if (name.rfind("L_", 0) == 0) return "R_" + name.substr(2);
    return "";
}

int Batch::scaleBoneSymmetric(Model& m, const Index& ix, const std::string& bone,
                              const Vec3& axisWorld, double factor) {
    int n = 0;
    if (scaleBone(m, ix, bone, axisWorld, factor)) n++;
    std::string other = counterpartName(bone);
    if (!other.empty() && ix.boneSlot(other) >= 0) {
        Vec3 mirrored{ -axisWorld[0], axisWorld[1], axisWorld[2] };  // reflect across sagittal (X)
        if (scaleBone(m, ix, other, mirrored, factor)) n++;
    }
    return n;
}

} // namespace mass
