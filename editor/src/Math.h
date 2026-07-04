#pragma once
// Minimal self-contained 3D math (column-major mat4 for GL/ImGuizmo).
#include <cmath>
#include <array>
#include "MassModel.h"

namespace ed {

struct V3 {
    float x=0,y=0,z=0;
    V3() = default;
    V3(float a,float b,float c):x(a),y(b),z(c){}
    V3(const Vec3& v):x((float)v[0]),y((float)v[1]),z((float)v[2]){}
    V3 operator+(const V3& o) const { return {x+o.x,y+o.y,z+o.z}; }
    V3 operator-(const V3& o) const { return {x-o.x,y-o.y,z-o.z}; }
    V3 operator*(float s) const { return {x*s,y*s,z*s}; }
};
inline float dot(const V3&a,const V3&b){return a.x*b.x+a.y*b.y+a.z*b.z;}
inline V3 cross(const V3&a,const V3&b){return {a.y*b.z-a.z*b.y,a.z*b.x-a.x*b.z,a.x*b.y-a.y*b.x};}
inline float length(const V3&a){return std::sqrt(dot(a,a));}
inline V3 normalize(const V3&a){float l=length(a); return l>1e-8f?a*(1.0f/l):a;}

// column-major 4x4
struct M4 {
    float m[16] = {1,0,0,0, 0,1,0,0, 0,0,1,0, 0,0,0,1};
    float* data(){return m;}
    const float* data() const {return m;}
};

inline M4 mul(const M4& a, const M4& b) {
    M4 r;
    for (int c=0;c<4;c++) for (int row=0;row<4;row++){
        float s=0; for(int k=0;k<4;k++) s += a.m[k*4+row]*b.m[c*4+k];
        r.m[c*4+row]=s;
    }
    return r;
}
// transform direction (w=0, rotation only)
inline V3 mulDir(const M4& a, const V3& p){
    return { a.m[0]*p.x+a.m[4]*p.y+a.m[8]*p.z,
             a.m[1]*p.x+a.m[5]*p.y+a.m[9]*p.z,
             a.m[2]*p.x+a.m[6]*p.y+a.m[10]*p.z };
}
// transform point (w=1)
inline V3 mulPoint(const M4& a, const V3& p){
    float x=a.m[0]*p.x+a.m[4]*p.y+a.m[8]*p.z+a.m[12];
    float y=a.m[1]*p.x+a.m[5]*p.y+a.m[9]*p.z+a.m[13];
    float z=a.m[2]*p.x+a.m[6]*p.y+a.m[10]*p.z+a.m[14];
    return {x,y,z};
}

inline M4 perspective(float fovy, float aspect, float znear, float zfar){
    M4 r; for(int i=0;i<16;i++) r.m[i]=0;
    float t = 1.0f/std::tan(fovy*0.5f);
    r.m[0]=t/aspect; r.m[5]=t; r.m[10]=(zfar+znear)/(znear-zfar);
    r.m[11]=-1.0f; r.m[14]=(2*zfar*znear)/(znear-zfar);
    return r;
}
inline M4 lookAt(const V3& eye, const V3& center, const V3& up){
    V3 f = normalize(center-eye);
    V3 s = normalize(cross(f,up));
    V3 u = cross(s,f);
    M4 r;
    r.m[0]=s.x; r.m[4]=s.y; r.m[8]=s.z;  r.m[12]=-dot(s,eye);
    r.m[1]=u.x; r.m[5]=u.y; r.m[9]=u.z;  r.m[13]=-dot(u,eye);
    r.m[2]=-f.x;r.m[6]=-f.y;r.m[10]=-f.z; r.m[14]=dot(f,eye);
    r.m[3]=0; r.m[7]=0; r.m[11]=0; r.m[15]=1;
    return r;
}

// Build model matrix (column-major) from a MASS Transform (row-major Mat3 + Vec3).
inline M4 fromTransform(const Transform& t){
    M4 r;
    const Mat3& L = t.linear; // row-major: L[row*3+col]
    // column c of rotation = L[:,c]
    r.m[0]=(float)L[0]; r.m[1]=(float)L[3]; r.m[2]=(float)L[6];  r.m[3]=0;
    r.m[4]=(float)L[1]; r.m[5]=(float)L[4]; r.m[6]=(float)L[7];  r.m[7]=0;
    r.m[8]=(float)L[2]; r.m[9]=(float)L[5]; r.m[10]=(float)L[8]; r.m[11]=0;
    r.m[12]=(float)t.translation[0]; r.m[13]=(float)t.translation[1]; r.m[14]=(float)t.translation[2]; r.m[15]=1;
    return r;
}

// inverse of a rigid transform (R|t) -> (R^T | -R^T t)
inline M4 rigidInverse(const M4& a){
    M4 r;
    // R^T
    r.m[0]=a.m[0]; r.m[1]=a.m[4]; r.m[2]=a.m[8];
    r.m[4]=a.m[1]; r.m[5]=a.m[5]; r.m[6]=a.m[9];
    r.m[8]=a.m[2]; r.m[9]=a.m[6]; r.m[10]=a.m[10];
    r.m[3]=0; r.m[7]=0; r.m[11]=0; r.m[15]=1;
    V3 t{a.m[12],a.m[13],a.m[14]};
    r.m[12]=-(r.m[0]*t.x+r.m[4]*t.y+r.m[8]*t.z);
    r.m[13]=-(r.m[1]*t.x+r.m[5]*t.y+r.m[9]*t.z);
    r.m[14]=-(r.m[2]*t.x+r.m[6]*t.y+r.m[10]*t.z);
    return r;
}

// ---- picking ----
struct Ray { V3 o, d; };

// ray vs oriented box centered at model origin with half-extents he, transform model
// returns t of entry (>0) or -1
inline float rayOBB(const Ray& ray, const M4& model, const V3& he){
    // transform ray into box local space via inverse(model). model is rigid (R|t),
    // inverse = (R^T | -R^T t). Extract R columns.
    V3 cx{model.m[0],model.m[1],model.m[2]};
    V3 cy{model.m[4],model.m[5],model.m[6]};
    V3 cz{model.m[8],model.m[9],model.m[10]};
    V3 t{model.m[12],model.m[13],model.m[14]};
    V3 ro = ray.o - t;
    V3 lo{dot(ro,cx),dot(ro,cy),dot(ro,cz)};
    V3 ld{dot(ray.d,cx),dot(ray.d,cy),dot(ray.d,cz)};
    float tmin=-1e30f, tmax=1e30f;
    const float lomin[3]={-he.x,-he.y,-he.z}, lomax[3]={he.x,he.y,he.z};
    const float lop[3]={lo.x,lo.y,lo.z}, ldp[3]={ld.x,ld.y,ld.z};
    for(int i=0;i<3;i++){
        if(std::fabs(ldp[i])<1e-8f){ if(lop[i]<lomin[i]||lop[i]>lomax[i]) return -1; }
        else{
            float t1=(lomin[i]-lop[i])/ldp[i], t2=(lomax[i]-lop[i])/ldp[i];
            if(t1>t2) std::swap(t1,t2);
            tmin=std::max(tmin,t1); tmax=std::min(tmax,t2);
            if(tmin>tmax) return -1;
        }
    }
    return tmin>0?tmin:(tmax>0?tmax:-1);
}

// ray vs triangle (Moller-Trumbore); returns t>0 or -1
inline float rayTriangle(const Ray& ray, const V3& a, const V3& b, const V3& c){
    V3 e1 = b - a, e2 = c - a;
    V3 p = cross(ray.d, e2);
    float det = dot(e1, p);
    if (std::fabs(det) < 1e-9f) return -1;
    float inv = 1.0f/det;
    V3 tv = ray.o - a;
    float u = dot(tv, p) * inv;
    if (u < 0 || u > 1) return -1;
    V3 q = cross(tv, e1);
    float v = dot(ray.d, q) * inv;
    if (v < 0 || u + v > 1) return -1;
    float t = dot(e2, q) * inv;
    return t > 1e-5f ? t : -1;
}

// ray vs sphere at center c, radius r
inline float raySphere(const Ray& ray, const V3& c, float r){
    V3 oc = ray.o - c;
    float b = dot(oc, ray.d);
    float cc = dot(oc,oc) - r*r;
    float disc = b*b - cc;
    if(disc<0) return -1;
    float t = -b - std::sqrt(disc);
    return t>0?t:-1;
}

// distance from point to segment (for muscle picking), returns t param + dist
inline float raySegmentDist(const Ray& ray, const V3& a, const V3& b, float& outT){
    // approximate: sample closest point between ray and segment
    V3 u = ray.d;              // ray dir (normalized)
    V3 v = b - a;
    V3 w = ray.o - a;
    float A=dot(u,u), B=dot(u,v), C=dot(v,v), D=dot(u,w), E=dot(v,w);
    float den = A*C - B*B;
    float sc, tc;
    if(std::fabs(den)<1e-8f){ sc=0; tc=(B>C?D/B:E/C); }
    else { sc=(B*E-C*D)/den; tc=(A*E-B*D)/den; }
    tc = tc<0?0:(tc>1?1:tc);
    outT = sc;
    V3 pRay = ray.o + u*sc;
    V3 pSeg = a + v*tc;
    return length(pRay - pSeg);
}

} // namespace ed
