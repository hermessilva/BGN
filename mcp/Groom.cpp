#include "Groom.h"
#include "Kinematics.h"   // normalize
#include <cmath>

namespace mass {

static Vec3 add(const Vec3& a, const Vec3& b) { return { a[0]+b[0], a[1]+b[1], a[2]+b[2] }; }
static Vec3 sub(const Vec3& a, const Vec3& b) { return { a[0]-b[0], a[1]-b[1], a[2]-b[2] }; }
static Vec3 mul(const Vec3& a, double s) { return { a[0]*s, a[1]*s, a[2]*s }; }
static double len(const Vec3& a) { return std::sqrt(a[0]*a[0]+a[1]*a[1]+a[2]*a[2]); }

void HairSim::init(const Vec3& root, const Vec3& dirIn, int segments, double segLen) {
    Vec3 dir = normalize(dirIn);
    mRest = segLen;
    mP.clear(); mPrev.clear();
    for (int i = 0; i <= segments; i++) {
        Vec3 p = add(root, mul(dir, segLen * i));
        mP.push_back(p);
        mPrev.push_back(p);
    }
}

double HairSim::segLength(int i) const {
    if (i < 0 || i + 1 >= (int)mP.size()) return 0.0;
    return len(sub(mP[i+1], mP[i]));
}

void HairSim::step(double dt, const Vec3& gravity, const Vec3& wind, double damping, int iters) {
    if (mP.size() < 2) return;
    const Vec3 accel = add(gravity, wind);
    const double keep = 1.0 - (damping < 0 ? 0 : (damping > 1 ? 1 : damping));

    // Verlet-style integrate free particles (index 0 is the pinned root)
    for (size_t i = 1; i < mP.size(); i++) {
        Vec3 vel = mul(sub(mP[i], mPrev[i]), keep);       // damped velocity
        Vec3 next = add(add(mP[i], vel), mul(accel, dt * dt));
        mPrev[i] = mP[i];
        mP[i] = next;
    }

    // inextensibility: project each link back to rest length, root held fixed
    for (int it = 0; it < iters; it++) {
        for (size_t i = 0; i + 1 < mP.size(); i++) {
            Vec3 d = sub(mP[i+1], mP[i]);
            double L = len(d);
            if (L < 1e-12) continue;
            double diff = (L - mRest) / L;
            if (i == 0) {
                // particle 0 fixed: move only particle 1
                mP[i+1] = sub(mP[i+1], mul(d, diff));
            } else {
                Vec3 corr = mul(d, 0.5 * diff);
                mP[i]   = add(mP[i], corr);
                mP[i+1] = sub(mP[i+1], corr);
            }
        }
    }
}

} // namespace mass
