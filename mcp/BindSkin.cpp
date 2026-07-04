#include "BindSkin.h"
#include <fstream>
#include <sstream>
#include <cmath>
#include <array>
#include <map>
#include <algorithm>

namespace mass {
using json = nlohmann::json;
using V3 = std::array<double, 3>;

// ---- tiny OBJ reader: triangle soup (positions + per-vertex normals) ----
static bool loadObjSoup(const std::string& path, std::vector<V3>& pos, std::vector<V3>& nrm, std::string* err) {
    std::ifstream f(path);
    if (!f) { if (err) *err = "cannot open " + path; return false; }
    std::vector<V3> V, N;
    std::string line;
    auto parseIdx = [](const std::string& tok, int& vi, int& ni) {
        vi = ni = 0;
        // formats: v, v/vt, v//vn, v/vt/vn
        size_t s1 = tok.find('/');
        if (s1 == std::string::npos) { vi = std::atoi(tok.c_str()); return; }
        vi = std::atoi(tok.substr(0, s1).c_str());
        size_t s2 = tok.find('/', s1 + 1);
        if (s2 != std::string::npos && s2 + 1 < tok.size()) ni = std::atoi(tok.substr(s2 + 1).c_str());
    };
    while (std::getline(f, line)) {
        if (line.size() < 2) continue;
        if (line[0] == 'v' && line[1] == ' ') {
            std::istringstream ss(line.substr(2)); V3 v{}; ss >> v[0] >> v[1] >> v[2]; V.push_back(v);
        } else if (line[0] == 'v' && line[1] == 'n') {
            std::istringstream ss(line.substr(3)); V3 n{}; ss >> n[0] >> n[1] >> n[2]; N.push_back(n);
        } else if (line[0] == 'f' && line[1] == ' ') {
            std::istringstream ss(line.substr(2));
            std::vector<std::pair<int,int>> fv; std::string tok;
            while (ss >> tok) { int vi, ni; parseIdx(tok, vi, ni); fv.push_back({vi, ni}); }
            // fan-triangulate
            for (size_t k = 2; k < fv.size(); k++) {
                std::pair<int,int> tri[3] = { fv[0], fv[k-1], fv[k] };
                V3 tp[3], tn[3];
                for (int j = 0; j < 3; j++) {
                    int vi = tri[j].first; if (vi < 0) vi = (int)V.size() + vi + 1;
                    tp[j] = (vi >= 1 && vi <= (int)V.size()) ? V[vi-1] : V3{0,0,0};
                }
                // face normal fallback if no vn
                V3 e1{ tp[1][0]-tp[0][0], tp[1][1]-tp[0][1], tp[1][2]-tp[0][2] };
                V3 e2{ tp[2][0]-tp[0][0], tp[2][1]-tp[0][1], tp[2][2]-tp[0][2] };
                V3 fn{ e1[1]*e2[2]-e1[2]*e2[1], e1[2]*e2[0]-e1[0]*e2[2], e1[0]*e2[1]-e1[1]*e2[0] };
                double fl = std::sqrt(fn[0]*fn[0]+fn[1]*fn[1]+fn[2]*fn[2]); if (fl < 1e-12) fl = 1;
                fn = { fn[0]/fl, fn[1]/fl, fn[2]/fl };
                for (int j = 0; j < 3; j++) {
                    int ni = tri[j].second; if (ni < 0) ni = (int)N.size() + ni + 1;
                    tn[j] = (ni >= 1 && ni <= (int)N.size()) ? N[ni-1] : fn;
                    pos.push_back(tp[j]); nrm.push_back(tn[j]);
                }
            }
        }
    }
    if (pos.empty()) { if (err) *err = "no triangles in " + path; return false; }
    return true;
}

// rotate a point/dir by Euler XYZ (degrees): R = Rz*Ry*Rx
static V3 rotEuler(const V3& v, const Vec3& deg) {
    const double d2r = 3.14159265358979323846 / 180.0;
    double cx=std::cos(deg[0]*d2r), sx=std::sin(deg[0]*d2r);
    double cy=std::cos(deg[1]*d2r), sy=std::sin(deg[1]*d2r);
    double cz=std::cos(deg[2]*d2r), sz=std::sin(deg[2]*d2r);
    V3 a{ v[0], cx*v[1]-sx*v[2], sx*v[1]+cx*v[2] };
    V3 b{ cy*a[0]+sy*a[2], a[1], -sy*a[0]+cy*a[2] };
    return V3{ cz*b[0]-sz*b[1], sz*b[0]+cz*b[1], b[2] };
}

// morph one limb chain: scale the subtree rooted at `root` about `pivot` by k
static void scaleSubtree(Model& m, const std::string& root, const V3& pivot, double k) {
    std::vector<std::string> frontier{ root };
    while (!frontier.empty()) {
        std::string cur = frontier.back(); frontier.pop_back();
        Node* n = m.findNode(cur);
        if (n) {
            for (Transform* t : { &n->body.t, &n->joint.t })
                for (int a=0;a<3;a++) t->translation[a] = pivot[a] + (t->translation[a]-pivot[a])*k;
            for (int a=0;a<3;a++) n->body.size[a] *= k;
        }
        for (auto& c : m.skeleton) if (c.parent == cur) frontier.push_back(c.id);
    }
}
// mesh reach from pivot along axis (0=X,1=Y) on side `sgn`, within a perpendicular
// band. Returns a high percentile (not the max) of the reach distribution so thin
// outlying geometry (wings, hair, dress flare) does not inflate the limb length.
static double meshReach(const std::vector<V3>& P, const V3& piv, int axis, double sgn, double band) {
    std::vector<double> ds;
    for (const auto& v : P) {
        double dperp1 = std::fabs((axis==1 ? v[0] : v[1]) - (axis==1 ? piv[0] : piv[1]));
        double dperp2 = std::fabs(v[2] - piv[2]);
        if (std::max(dperp1, dperp2) > band) continue;
        double d = (v[axis]-piv[axis]) * sgn;
        if (d > 0) ds.push_back(d);
    }
    if (ds.empty()) return 0;
    std::sort(ds.begin(), ds.end());
    return ds[(size_t)(0.90 * (ds.size() - 1))];   // 90th percentile
}

json BindSkin::bind(Model& m, const std::string& obj, const Vec3& rotDeg,
                    double userScale, const Vec3& offset, bool fitBones, std::string* err) {
    if (m.skeleton.empty()) { if (err) *err = "model has no skeleton"; return json(); }
    std::vector<V3> pos, nrm;
    if (!loadObjSoup(obj, pos, nrm, err)) return json();

    // rotate
    for (auto& p : pos) p = rotEuler(p, rotDeg);

    // mesh bbox
    V3 mn = pos[0], mx = pos[0];
    for (const auto& p : pos) for (int a=0;a<3;a++){ mn[a]=std::min(mn[a],p[a]); mx[a]=std::max(mx[a],p[a]); }
    // skeleton rest bbox
    V3 smn{1e30,1e30,1e30}, smx{-1e30,-1e30,-1e30};
    for (const auto& n : m.skeleton) {
        const auto& t = n.body.t.translation;
        double he = (n.body.type == "Box")
            ? 0.5 * std::max({ n.body.size[0], n.body.size[1], n.body.size[2] })
            : n.body.radius;
        for (int a=0;a<3;a++){ smn[a]=std::min(smn[a], t[a]-he); smx[a]=std::max(smx[a], t[a]+he); }
    }
    double meshH = std::max(1e-4, mx[1]-mn[1]);
    double skelH = std::max(1e-4, smx[1]-smn[1]);
    double sc = (skelH/meshH) * std::max(0.01, userScale);
    V3 mCtr{ (mn[0]+mx[0])*0.5, 0, (mn[2]+mx[2])*0.5 };
    V3 sCtr{ (smn[0]+smx[0])*0.5, 0, (smn[2]+smx[2])*0.5 };
    // place verts (match editor: center X/Z, align feet)
    for (auto& p : pos) {
        p[0] = (p[0]-mCtr[0])*sc + sCtr[0] + offset[0];
        p[2] = (p[2]-mCtr[2])*sc + sCtr[2] + offset[2];
        p[1] = (p[1]-mn[1])*sc + smn[1] + offset[1];
    }

    // adapt the skeleton limb chains to the placed mesh (arms/legs) so the bones
    // match the character; binding below then uses the fitted skeleton
    int fitted = 0;
    if (fitBones) {
        for (const char* side : { "R", "L" }) {
            Node* sh = m.findNode(std::string("Shoulder") + side);
            Node* hand = m.findNode(std::string("Hand") + side);
            if (sh && hand) {
                V3 piv{ sh->joint.t.translation[0], sh->joint.t.translation[1], sh->joint.t.translation[2] };
                double handX = hand->body.t.translation[0];
                double sgn = (handX - piv[0]) >= 0 ? 1.0 : -1.0;
                double reach = std::fabs(handX - piv[0]);
                double mr = meshReach(pos, piv, 0, sgn, 0.28);
                if (reach > 1e-3 && mr > 1e-3) { double k = std::clamp(mr/reach, 0.3, 1.3); scaleSubtree(m, std::string("Shoulder")+side, piv, k); fitted++; }
            }
            Node* fem = m.findNode(std::string("Femur") + side);
            Node* foot = m.findNode(std::string("Talus") + side);
            if (fem && foot) {
                V3 piv{ fem->joint.t.translation[0], fem->joint.t.translation[1], fem->joint.t.translation[2] };
                double footY = foot->body.t.translation[1];
                double reach = std::fabs(footY - piv[1]);
                double mr = meshReach(pos, piv, 1, -1.0, 0.20);
                if (reach > 1e-3 && mr > 1e-3) { double k = std::clamp(mr/reach, 0.5, 1.4); scaleSubtree(m, std::string("Femur")+side, piv, k); fitted++; }
            }
        }
    }

    // precompute rest-inverse (R^T, -R^T t) per body + half-extents
    int nb = (int)m.skeleton.size();
    std::vector<std::array<double,9>> Rt(nb);
    std::vector<V3> Ti(nb), He(nb);
    for (int b=0;b<nb;b++) {
        const auto& L = m.skeleton[b].body.t.linear;      // row-major R
        std::array<double,9> rt{ L[0],L[3],L[6], L[1],L[4],L[7], L[2],L[5],L[8] }; // R^T
        Rt[b] = rt;
        const auto& t = m.skeleton[b].body.t.translation;
        Ti[b] = { -(rt[0]*t[0]+rt[1]*t[1]+rt[2]*t[2]),
                  -(rt[3]*t[0]+rt[4]*t[1]+rt[5]*t[2]),
                  -(rt[6]*t[0]+rt[7]*t[1]+rt[8]*t[2]) };
        const auto& n = m.skeleton[b];
        He[b] = (n.body.type=="Box")
            ? V3{ n.body.size[0]*0.5, n.body.size[1]*0.5, n.body.size[2]*0.5 }
            : V3{ n.body.radius, n.body.radius, n.body.radius };
        for (int a=0;a<3;a++) He[b][a] = std::max(1e-4, He[b][a]);
    }
    // nearest body per vertex (ellipsoid metric) -> tally
    std::vector<int> tally(nb, 0);
    for (const auto& p : pos) {
        int best=0; double bestK=1e30;
        for (int b=0;b<nb;b++) {
            double lx = Rt[b][0]*p[0]+Rt[b][1]*p[1]+Rt[b][2]*p[2] + Ti[b][0];
            double ly = Rt[b][3]*p[0]+Rt[b][4]*p[1]+Rt[b][5]*p[2] + Ti[b][1];
            double lz = Rt[b][6]*p[0]+Rt[b][7]*p[1]+Rt[b][8]*p[2] + Ti[b][2];
            double kx=lx/He[b][0], ky=ly/He[b][1], kz=lz/He[b][2];
            double k = std::sqrt(kx*kx+ky*ky+kz*kz);
            if (k<bestK){ bestK=k; best=b; }
        }
        tally[best]++;
    }

    // store descriptor
    m.skin.obj = obj; m.skin.rotDeg = rotDeg; m.skin.userScale = userScale;
    m.skin.offset = offset; m.skin.present = true;

    json bones = json::object();
    for (int b=0;b<nb;b++) if (tally[b]>0) bones[m.skeleton[b].id] = tally[b];
    return json{
        {"vertices", (int)pos.size()}, {"triangles", (int)pos.size()/3},
        {"autoScale", sc}, {"fittedChains", fitted}, {"bones", bones}
    };
}

} // namespace mass
