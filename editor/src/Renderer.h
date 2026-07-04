#pragma once
// Minimal batched line/point renderer (OpenGL 3.3 core).
#include "Math.h"
#include <vector>
#include <cstdint>

namespace ed {

struct Renderer {
    bool init();                 // compile shaders, create buffers
    void shutdown();

    struct LightGPU { int type; V3 dir; V3 color; };  // color already scaled by intensity
    void setLights(const std::vector<LightGPU>& lights, float ambient);

    // offscreen render target (the 3D scene is drawn here, shown via ImGui::Image)
    void beginTarget(int w, int h);   // create/resize FBO, bind, set viewport, clear
    void endTarget();                 // unbind (back to default framebuffer)
    unsigned targetTexture() const { return mColorTex; }

    void begin(const M4& viewProj, const V3& eye);
    void line(const V3& a, const V3& b, const V3& color, float alpha = 1.0f);
    void point(const V3& p, const V3& color, float alpha = 1.0f);
    void wireBox(const M4& model, const V3& halfExtents, const V3& color, float alpha = 1.0f);
    void solidBox(const M4& model, const V3& halfExtents, const V3& color, float alpha = 1.0f);
    // add a single world-space triangle (face normal computed) to the batched lit pass
    void triWorld(const V3& a, const V3& b, const V3& c, const V3& color, float alpha = 1.0f);
    // world-space triangle with explicit per-vertex normals (smooth shading)
    void triSmooth(const V3& a, const V3& na, const V3& b, const V3& nb,
                   const V3& c, const V3& nc, const V3& color, float alpha = 1.0f);
    void solidSphere(const V3& center, float radius, const V3& color, int seg = 12);
    // smooth ellipsoid: unit sphere scaled by halfExtents, transformed by model (skin envelope)
    void solidEllipsoid(const M4& model, const V3& halfExtents, const V3& color, int seg = 14);
    // GPU mesh: upload once, draw many with a per-body model matrix (transform on GPU)
    unsigned uploadMesh(const std::vector<V3>& pos, const std::vector<V3>& nrm); // returns id (0 = fail)
    void drawMeshGPU(unsigned id, const M4& model, const V3& color);            // queued, drawn in flush
    void freeMeshes();

    // dedicated skin buffer (re-uploadable without leaking)
    void setSkinMesh(const std::vector<V3>& pos, const std::vector<V3>& nrm);
    void drawSkin(const M4& model, const V3& color);   // queued for this frame
    bool hasSkin() const { return mSkinCount > 0; }
    void axes(const M4& model, float scale);
    void grid(float size, int divs, const V3& color);
    void checkerGround(float size, int divs, const V3& a, const V3& b);
    void flush();                // upload + draw

private:
    struct Vtx { float x,y,z, r,g,b,a; };
    struct VtxN { float x,y,z, nx,ny,nz, r,g,b,a; };
    std::vector<Vtx> mLines;
    std::vector<Vtx> mPoints;
    std::vector<VtxN> mTris;
    unsigned mProg = 0, mVao = 0, mVbo = 0;         // flat (lines/points)
    unsigned mProgLit = 0, mVaoT = 0, mVboT = 0;    // lit (triangles)
    M4 mVP; V3 mEye;
    std::vector<LightGPU> mLightsGPU;
    float mAmbient = 0.25f;
    unsigned mFbo = 0, mColorTex = 0, mDepthRbo = 0;
    int mFbW = 0, mFbH = 0;
    struct GpuMesh { unsigned vao = 0, vbo = 0; int count = 0; };
    std::vector<GpuMesh> mGpuMeshes;
    struct MeshCmd { unsigned id; M4 model; V3 color; };
    std::vector<MeshCmd> mMeshCmds;
    unsigned mSkinVao = 0, mSkinVbo = 0; int mSkinCount = 0;
    struct { bool active = false; M4 model; V3 color; } mSkinCmd;
    void push(std::vector<Vtx>& buf, const V3& p, const V3& c, float a);
    void tri(const V3& a, const V3& b, const V3& c, const V3& n, const V3& col, float al);
    void draw(const std::vector<Vtx>& buf, unsigned mode);
};

} // namespace ed
