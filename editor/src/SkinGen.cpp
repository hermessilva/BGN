#include "SkinGen.h"
#include <cmath>
#include <algorithm>
#include <functional>

namespace ed {

// ---- signed field: smooth-union of per-body ellipsoids, minus skin inflate ----
struct Ellipsoid { V3 c; V3 cx, cy, cz; V3 he; }; // center, rotation columns, half-extents

static float ellipsoidSDF(const Ellipsoid& e, const V3& p) {
    V3 d = p - e.c;
    V3 l{ dot(d, e.cx), dot(d, e.cy), dot(d, e.cz) };   // to local (R^T)
    V3 q{ l.x/e.he.x, l.y/e.he.y, l.z/e.he.z };
    float k = length(q);
    float mn = std::min(e.he.x, std::min(e.he.y, e.he.z));
    return (k - 1.0f) * mn;   // approximate ellipsoid distance
}
static float smin(float a, float b, float k) {
    if (k <= 1e-5f) return std::min(a, b);
    float h = std::max(0.0f, std::min(1.0f, 0.5f + 0.5f*(b - a)/k));
    return b*(1-h) + a*h - k*h*(1-h);
}

// capsule (segment) SDF — used to fold muscle volume into the field
struct Capsule { V3 a, b; float r; };
static float capsuleSDF(const Capsule& c, const V3& p) {
    V3 pa = p - c.a, ba = c.b - c.a;
    float h = std::max(0.0f, std::min(1.0f, dot(pa, ba) / std::max(dot(ba, ba), 1e-8f)));
    return length(pa - ba * h) - c.r;
}

// ---- Marching Cubes tables (Paul Bourke) ----
static const int kEdge[256] = {
0x0,0x109,0x203,0x30a,0x406,0x50f,0x605,0x70c,0x80c,0x905,0xa0f,0xb06,0xc0a,0xd03,0xe09,0xf00,
0x190,0x99,0x393,0x29a,0x596,0x49f,0x795,0x69c,0x99c,0x895,0xb9f,0xa96,0xd9a,0xc93,0xf99,0xe90,
0x230,0x339,0x33,0x13a,0x636,0x73f,0x435,0x53c,0xa3c,0xb35,0x83f,0x936,0xe3a,0xf33,0xc39,0xd30,
0x3a0,0x2a9,0x1a3,0xaa,0x7a6,0x6af,0x5a5,0x4ac,0xbac,0xaa5,0x9af,0x8a6,0xfaa,0xea3,0xda9,0xca0,
0x460,0x569,0x663,0x76a,0x66,0x16f,0x265,0x36c,0xc6c,0xd65,0xe6f,0xf66,0x86a,0x963,0xa69,0xb60,
0x5f0,0x4f9,0x7f3,0x6fa,0x1f6,0xff,0x3f5,0x2fc,0xdfc,0xcf5,0xfff,0xef6,0x9fa,0x8f3,0xbf9,0xaf0,
0x650,0x759,0x453,0x55a,0x256,0x35f,0x55,0x15c,0xe5c,0xf55,0xc5f,0xd56,0xa5a,0xb53,0x859,0x950,
0x7c0,0x6c9,0x5c3,0x4ca,0x3c6,0x2cf,0x1c5,0xcc,0xfcc,0xec5,0xdcf,0xcc6,0xbca,0xac3,0x9c9,0x8c0,
0x8c0,0x9c9,0xac3,0xbca,0xcc6,0xdcf,0xec5,0xfcc,0xcc,0x1c5,0x2cf,0x3c6,0x4ca,0x5c3,0x6c9,0x7c0,
0x950,0x859,0xb53,0xa5a,0xd56,0xc5f,0xf55,0xe5c,0x15c,0x55,0x35f,0x256,0x55a,0x453,0x759,0x650,
0xaf0,0xbf9,0x8f3,0x9fa,0xef6,0xfff,0xcf5,0xdfc,0x2fc,0x3f5,0xff,0x1f6,0x6fa,0x7f3,0x4f9,0x5f0,
0xb60,0xa69,0x963,0x86a,0xf66,0xe6f,0xd65,0xc6c,0x36c,0x265,0x16f,0x66,0x76a,0x663,0x569,0x460,
0xca0,0xda9,0xea3,0xfaa,0x8a6,0x9af,0xaa5,0xbac,0x4ac,0x5a5,0x6af,0x7a6,0xaa,0x1a3,0x2a9,0x3a0,
0xd30,0xc39,0xf33,0xe3a,0x936,0x83f,0xb35,0xa3c,0x53c,0x435,0x73f,0x636,0x13a,0x33,0x339,0x230,
0xe90,0xf99,0xc93,0xd9a,0xa96,0xb9f,0x895,0x99c,0x69c,0x795,0x49f,0x596,0x29a,0x393,0x99,0x190,
0xf00,0xe09,0xd03,0xc0a,0xb06,0xa0f,0x905,0x80c,0x70c,0x605,0x50f,0x406,0x30a,0x203,0x109,0x0 };

#include "SkinGenTriTable.inc"   // static const int kTri[256][16]

// edge endpoints (corner indices)
static const int kEV[12][2] = {{0,1},{1,2},{2,3},{3,0},{4,5},{5,6},{6,7},{7,4},{0,4},{1,5},{2,6},{3,7}};
// corner offsets
static const int kCO[8][3] = {{0,0,0},{1,0,0},{1,1,0},{0,1,0},{0,0,1},{1,0,1},{1,1,1},{0,1,1}};

void GenerateSkin(const Model& m, const SkinParams& p,
                  std::vector<V3>& outPos, std::vector<V3>& outNrm,
                  std::atomic<float>* progress) {
    if (progress) progress->store(0.0f);
    outPos.clear(); outNrm.clear();
    if (m.skeleton.empty()) return;

    // build ellipsoids + bounding box
    std::vector<Ellipsoid> E;
    V3 lo{1e9f,1e9f,1e9f}, hi{-1e9f,-1e9f,-1e9f};
    for (const auto& n : m.skeleton) {
        Ellipsoid e;
        e.c = V3(n.body.t.translation);
        const Mat3& L = n.body.t.linear; // row-major world rotation
        e.cx = {(float)L[0],(float)L[3],(float)L[6]};   // column 0
        e.cy = {(float)L[1],(float)L[4],(float)L[7]};   // column 1
        e.cz = {(float)L[2],(float)L[5],(float)L[8]};   // column 2
        if (n.body.type == "Box")
            e.he = { (float)n.body.size[0]*0.5f*p.bodyScale,
                     (float)n.body.size[1]*0.5f*p.bodyScale,
                     (float)n.body.size[2]*0.5f*p.bodyScale };
        else
            e.he = V3{(float)n.body.radius, (float)n.body.radius, (float)n.body.radius} * (p.bodyScale);
        E.push_back(e);
        float rad = std::max(e.he.x, std::max(e.he.y, e.he.z)) + p.inflate + p.smooth;
        lo = { std::min(lo.x, e.c.x-rad), std::min(lo.y, e.c.y-rad), std::min(lo.z, e.c.z-rad) };
        hi = { std::max(hi.x, e.c.x+rad), std::max(hi.y, e.c.y+rad), std::max(hi.z, e.c.z+rad) };
    }

    // muscle capsules folded into the field (skin wraps around the muscle bulk)
    std::vector<Capsule> C;
    if (p.includeMuscles) {
        for (const auto& mu : m.muscles) {
            if (mu.waypoints.size() < 2) continue;
            double pcsa = mu.pcsa_cm2 > 0.0 ? mu.pcsa_cm2 : mu.f0 / m.meta.specific_tension_N_cm2;
            float r = (float)std::sqrt(std::max(pcsa, 0.01) / 3.14159265) * 0.01f * p.muscleScale;
            if (r < 0.004f) r = 0.004f; if (r > 0.04f) r = 0.04f;
            for (size_t k = 1; k < mu.waypoints.size(); k++) {
                Capsule c; c.a = V3(mu.waypoints[k-1].p); c.b = V3(mu.waypoints[k].p); c.r = r;
                C.push_back(c);
                float rr = r + p.inflate + p.smooth;
                for (const V3& e : {c.a, c.b}) {
                    lo = { std::min(lo.x,e.x-rr), std::min(lo.y,e.y-rr), std::min(lo.z,e.z-rr) };
                    hi = { std::max(hi.x,e.x+rr), std::max(hi.y,e.y+rr), std::max(hi.z,e.z+rr) };
                }
            }
        }
    }

    auto field = [&](const V3& q) -> float {
        float d = 1e9f;
        for (const auto& e : E) d = smin(d, ellipsoidSDF(e, q), p.smooth);
        for (const auto& c : C) d = smin(d, capsuleSDF(c, q), p.smooth);
        return d - p.inflate;   // surface (0) sits inflate outside the union
    };

    float cs = std::max(0.006f, p.cell);
    int nx = (int)std::ceil((hi.x-lo.x)/cs) + 1;
    int ny = (int)std::ceil((hi.y-lo.y)/cs) + 1;
    int nz = (int)std::ceil((hi.z-lo.z)/cs) + 1;
    nx = std::min(nx, 400); ny = std::min(ny, 400); nz = std::min(nz, 400);

    auto at = [&](int i,int j,int k){ return V3{ lo.x+i*cs, lo.y+j*cs, lo.z+k*cs }; };

    // sample one z-slab at a time to bound memory
    std::vector<float> slabA((size_t)nx*ny), slabB((size_t)nx*ny);
    auto sampleSlab = [&](int k, std::vector<float>& s){
        for (int j=0;j<ny;j++) for (int i=0;i<nx;i++) s[(size_t)j*nx+i] = field(at(i,j,k));
    };
    sampleSlab(0, slabA);

    auto grad = [&](const V3& q){
        float h=cs*0.5f;
        return normalize(V3{ field({q.x+h,q.y,q.z})-field({q.x-h,q.y,q.z}),
                             field({q.x,q.y+h,q.z})-field({q.x,q.y-h,q.z}),
                             field({q.x,q.y,q.z+h})-field({q.x,q.y,q.z-h}) });
    };

    for (int k=0;k<nz-1;k++) {
        if (progress) progress->store((float)k / std::max(1, nz-1));
        sampleSlab(k+1, slabB);
        for (int j=0;j<ny-1;j++) for (int i=0;i<nx-1;i++) {
            float val[8]; V3 pos[8];
            for (int c=0;c<8;c++){
                int ci=i+kCO[c][0], cj=j+kCO[c][1], ck=k+kCO[c][2];
                pos[c] = at(ci,cj,ck);
                val[c] = (kCO[c][2]==0? slabA : slabB)[(size_t)cj*nx+ci];
            }
            int idx=0; for (int c=0;c<8;c++) if (val[c] < 0.0f) idx |= (1<<c);
            int eb = kEdge[idx];
            if (eb == 0) continue;
            V3 vert[12];
            for (int e=0;e<12;e++) if (eb & (1<<e)) {
                int a=kEV[e][0], b=kEV[e][1];
                float t = (0.0f - val[a]) / (val[b]-val[a] + 1e-12f);
                vert[e] = pos[a] + (pos[b]-pos[a]) * t;
            }
            for (int t=0; kTri[idx][t] != -1; t+=3) {
                V3 a=vert[kTri[idx][t]], b=vert[kTri[idx][t+1]], c=vert[kTri[idx][t+2]];
                outPos.push_back(a); outNrm.push_back(grad(a));
                outPos.push_back(b); outNrm.push_back(grad(b));
                outPos.push_back(c); outNrm.push_back(grad(c));
            }
        }
        slabA.swap(slabB);
    }
    if (progress) progress->store(1.0f);
}

} // namespace ed
