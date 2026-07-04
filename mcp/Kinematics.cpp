#include "Kinematics.h"
#include <cmath>

namespace mass {

// row-major: A[0..2]=row0, A[3..5]=row1, A[6..8]=row2
Vec3 mv(const Mat3& A, const Vec3& x) {
    return { A[0]*x[0]+A[1]*x[1]+A[2]*x[2],
             A[3]*x[0]+A[4]*x[1]+A[5]*x[2],
             A[6]*x[0]+A[7]*x[1]+A[8]*x[2] };
}
Mat3 mm(const Mat3& A, const Mat3& B) {
    Mat3 C{};
    for (int r = 0; r < 3; r++)
        for (int c = 0; c < 3; c++)
            C[r*3+c] = A[r*3+0]*B[0*3+c] + A[r*3+1]*B[1*3+c] + A[r*3+2]*B[2*3+c];
    return C;
}
Vec3 normalize(const Vec3& v) {
    double n = std::sqrt(v[0]*v[0]+v[1]*v[1]+v[2]*v[2]);
    if (n < 1e-12) return {0,0,0};
    return { v[0]/n, v[1]/n, v[2]/n };
}
// Rodrigues rotation matrix (row-major) for a unit axis and angle in radians.
Mat3 axisAngle(const Vec3& axisIn, double a) {
    Vec3 u = normalize(axisIn);
    double x = u[0], y = u[1], z = u[2];
    double c = std::cos(a), s = std::sin(a), t = 1.0 - c;
    return {
        t*x*x + c,   t*x*y - s*z, t*x*z + s*y,
        t*x*y + s*z, t*y*y + c,   t*y*z - s*x,
        t*x*z - s*y, t*y*z + s*x, t*z*z + c
    };
}

static Vec3 rotAbout(const Mat3& R, const Vec3& pivot, const Vec3& p) {
    Vec3 d{ p[0]-pivot[0], p[1]-pivot[1], p[2]-pivot[2] };
    Vec3 rd = mv(R, d);
    return { pivot[0]+rd[0], pivot[1]+rd[1], pivot[2]+rd[2] };
}

bool Kinematics::rotateJoint(Model& m, const Index& ix, const std::string& childBone,
                             const Vec3& axisWorld, double angle) {
    int c = ix.boneSlot(childBone);
    if (c < 0) return false;

    // pivot = current world center of this bone's joint
    const Vec3 pivot = m.skeleton[c].joint.t.translation;
    const Mat3 R = axisAngle(axisWorld, angle);

    for (int b : ix.subtreeBones(childBone)) {
        Node& n = m.skeleton[b];
        // body frame
        n.body.t.translation = rotAbout(R, pivot, n.body.t.translation);
        n.body.t.linear      = mm(R, n.body.t.linear);
        // joint frame (rotates about the same pivot)
        n.joint.t.translation = rotAbout(R, pivot, n.joint.t.translation);
        n.joint.t.linear      = mm(R, n.joint.t.linear);
        // muscle waypoints anchored to this bone follow rigidly
        for (const WpRef& w : ix.waypointsOfBone(n.id)) {
            Vec3& p = m.muscles[w.muscle].waypoints[w.wp].p;
            p = rotAbout(R, pivot, p);
        }
    }
    return true;
}

} // namespace mass
