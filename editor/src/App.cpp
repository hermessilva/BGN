#include "App.h"
#include "Bootstrap.h"
#include "Dialog.h"
#include <glad/glad.h>
#include <GLFW/glfw3.h>
#include <imgui.h>
#include <ImGuizmo.h>
#include <cstdio>
#include <fstream>
#include <sstream>
#include <functional>
#include <cstring>
#include <algorithm>
#include <assimp/Importer.hpp>
#include <assimp/scene.h>
#include <assimp/postprocess.h>

namespace ed {

static std::string deriveRoot(const std::string& fileInData);

// ---- lifecycle ----
bool App::init(GLFWwindow* win) {
    mWin = win;
    if (!mRen.init()) { std::fprintf(stderr, "renderer init failed\n"); return false; }
    newModel();
    mTrain.start(8765);   // telemetry server
    return true;
}
void App::shutdown() { if (mFillThread.joinable()) mFillThread.join(); mTrain.stop(); mRen.shutdown(); }

void App::startTraining() {
    // export current model to the real training data dir, then launch training
    std::string err;
    if (!ExportToLegacy(mModel, mDataRoot + "/data", &err)) { mStatus = "export to data failed: " + err; return; }
    std::string cmd = "powershell -ExecutionPolicy Bypass -File \"" + mDataRoot + "\\scripts\\train.ps1\"";
    mTrain.launchTraining(cmd);
    mStatus = "Training started (model exported to data)";
}

// ---- OBJ loader (positions in world*100 -> *0.01; per-vertex normals) ----
static bool loadObj(const std::string& path, MeshData& out) {
    std::ifstream f(path);
    if (!f) return false;
    std::vector<V3> verts, norms;
    std::string line;
    const float S = 0.01f;
    while (std::getline(f, line)) {
        if (line.size() < 2) continue;
        if (line[0] == 'v' && line[1] == ' ') {
            float x,y,z; if (std::sscanf(line.c_str()+2, "%f %f %f", &x,&y,&z)==3) verts.push_back({x*S,y*S,z*S});
        } else if (line[0] == 'v' && line[1] == 'n') {
            float x,y,z; if (std::sscanf(line.c_str()+3, "%f %f %f", &x,&y,&z)==3) norms.push_back({x,y,z});
        } else if (line[0] == 'f' && line[1] == ' ') {
            // parse polygon vertices: tokens "v/vt/vn"
            std::vector<int> vi, ni;
            std::stringstream ss(line.substr(2));
            std::string tok;
            while (ss >> tok) {
                int a=0,b=0,c=0;
                // formats: a, a/b, a//c, a/b/c
                size_t p1 = tok.find('/');
                if (p1 == std::string::npos) { a = std::stoi(tok); }
                else {
                    a = std::stoi(tok.substr(0,p1));
                    size_t p2 = tok.find('/', p1+1);
                    if (p2 == std::string::npos) { /* a/b */ }
                    else if (p2 > p1+1) c = std::stoi(tok.substr(p2+1));
                    else c = std::stoi(tok.substr(p2+1)); // a//c
                }
                vi.push_back(a); ni.push_back(c);
            }
            // triangulate fan
            for (size_t k = 1; k + 1 < vi.size(); k++) {
                int idx[3] = {0, (int)k, (int)k+1};
                V3 tp[3], tn[3]; bool haveN = true;
                for (int t=0;t<3;t++) {
                    int v = vi[idx[t]]; if (v<0) v = (int)verts.size()+v+1;
                    if (v<1 || v>(int)verts.size()) { haveN=false; break; }
                    tp[t] = verts[v-1];
                    int nn = ni[idx[t]];
                    if (nn>0 && nn<=(int)norms.size()) tn[t] = norms[nn-1];
                    else haveN = false;
                }
                if (tp[0].x==0&&tp[0].y==0&&tp[0].z==0 && !haveN) {}
                // compute face normal if missing
                if (!haveN) {
                    V3 fn = normalize(cross(tp[1]-tp[0], tp[2]-tp[0]));
                    tn[0]=tn[1]=tn[2]=fn;
                }
                for (int t=0;t<3;t++){ out.pos.push_back(tp[t]); out.nrm.push_back(tn[t]); }
            }
        }
    }
    return !out.pos.empty();
}

void App::loadMeshes() {
    mRen.freeMeshes();
    mMeshes.clear();
    mMeshes.resize(mModel.skeleton.size());
    // capture the authoring rest pose (fixed reference for mesh/skin deformation)
    mRestBody.resize(mModel.skeleton.size());
    for (size_t i = 0; i < mModel.skeleton.size(); i++) mRestBody[i] = mModel.skeleton[i].body.t;
    for (size_t i = 0; i < mModel.skeleton.size(); i++) {
        const Node& n = mModel.skeleton[i];
        mMeshes[i].objName = n.body.obj;
        if (n.body.obj.empty()) continue;
        if (loadObj(mDataRoot + "/data/OBJ/" + n.body.obj, mMeshes[i]))
            mMeshes[i].gpuId = mRen.uploadMesh(mMeshes[i].pos, mMeshes[i].nrm);
    }
}

// Reload meshes whenever the model's obj files or node count change (live edit).
void App::syncMeshes() {
    bool dirty = (mMeshes.size() != mModel.skeleton.size());
    for (size_t i = 0; !dirty && i < mModel.skeleton.size(); i++)
        if (mMeshes[i].objName != mModel.skeleton[i].body.obj) dirty = true;
    if (dirty) loadMeshes();
}

void App::generateSkin() { regenerateSkin(); }
void App::startKinematicSim() { if (!mSimActive) toggleSim(); mSim.setMode(SimBridge::Kinematic); }
void App::beginFill(const std::string& name) { startFillGeneration(name); }

// Bind a skin mesh (world-space pos+nrm) to the skeleton: each vertex to the nearest
// body, stored in that body's local space, then upload the current live pose.
void App::bindSkin(const std::vector<V3>& pos, const std::vector<V3>& nrm) {
    size_t nv = pos.size();
    mSkinLocalPos.assign(nv, V3{});
    mSkinLocalNrm.assign(nv, V3{});
    mSkinBone.assign(nv, 0);
    int nb = (int)mModel.skeleton.size();
    if (nb == 0) return;
    std::vector<M4> restInv(nb);
    for (int b = 0; b < nb; b++) restInv[b] = rigidInverse(restBodyMatrix(b));
    float bs = (float)mSkinParams.bodyScale;
    for (size_t i = 0; i < nv; i++) {
        int best = 0; float bestK = 1e30f;
        for (int b = 0; b < nb; b++) {
            const Node& n = mModel.skeleton[b];
            V3 lp = mulPoint(restInv[b], pos[i]);
            V3 he = n.body.type == "Box"
                ? V3{(float)n.body.size[0]*0.5f*bs, (float)n.body.size[1]*0.5f*bs, (float)n.body.size[2]*0.5f*bs}
                : V3{(float)n.body.radius*bs, (float)n.body.radius*bs, (float)n.body.radius*bs};
            float k = length(V3{ lp.x/std::max(he.x,1e-4f), lp.y/std::max(he.y,1e-4f), lp.z/std::max(he.z,1e-4f) });
            if (k < bestK) { bestK = k; best = b; }
        }
        mSkinBone[i] = best;
        mSkinLocalPos[i] = mulPoint(restInv[best], pos[i]);
        mSkinLocalNrm[i] = mulDir(restInv[best], nrm[i]);
    }
    mShowSkin = nv > 0;
    updateSkinPose();
}

void App::regenerateSkin() {
    std::vector<V3> pos, nrm;
    GenerateSkin(mModel, mSkinParams, pos, nrm);
    bindSkin(pos, nrm);
    mStatus = "Skin generated: " + std::to_string(pos.size()/3) + " triangles";
}

// ---------- Fills (generated tissue envelopes) ----------
static std::string dirOf(const std::string& path) {
    size_t s = path.find_last_of("/\\");
    return s == std::string::npos ? std::string(".") : path.substr(0, s);
}
std::string App::fillsDir() const {
    std::string base = mProjectPath.empty() ? mDataRoot : dirOf(mProjectPath);
    return base + "/fills";
}
static bool saveFillBin(const std::string& path, const std::vector<V3>& pos, const std::vector<V3>& nrm) {
    std::ofstream f(path, std::ios::binary);
    if (!f) return false;
    uint32_t n = (uint32_t)pos.size();
    f.write((const char*)&n, sizeof(n));
    for (uint32_t i = 0; i < n; i++) {
        f.write((const char*)&pos[i], sizeof(V3));
        f.write((const char*)&nrm[i], sizeof(V3));
    }
    return (bool)f;
}
static bool loadFillBin(const std::string& path, std::vector<V3>& pos, std::vector<V3>& nrm) {
    std::ifstream f(path, std::ios::binary);
    if (!f) return false;
    uint32_t n = 0; f.read((char*)&n, sizeof(n));
    if (!f || n > 20000000u) return false;
    pos.resize(n); nrm.resize(n);
    for (uint32_t i = 0; i < n; i++) {
        f.read((char*)&pos[i], sizeof(V3));
        f.read((char*)&nrm[i], sizeof(V3));
    }
    return (bool)f;
}

void App::startFillGeneration(const std::string& name) {
    if (mFillRunning) return;
    if (mModel.skeleton.empty()) { mStatus = "Load a model first"; return; }
    std::strncpy(mFillNewName, name.c_str(), sizeof(mFillNewName)-1);
    mFillDone.store(false);
    mFillProgress.store(0.0f);
    mFillRunning = true;
    SkinParams params = mSkinParams;    // snapshot
    mFillThread = std::thread([this, params]() {
        GenerateSkin(mModel, params, mFillPos, mFillNrm, &mFillProgress);
        mFillDone.store(true);
    });
}

void App::finalizeFill() {
    if (mFillThread.joinable()) mFillThread.join();
    mFillRunning = false;
    if (mFillPos.empty()) { mStatus = "Fill generation produced no mesh"; return; }
    // save to project fills/ folder
#ifdef _WIN32
    std::system(("cmd /c mkdir \"" + fillsDir() + "\" 2>nul").c_str());
#endif
    std::string name = mFillNewName;
    std::string rel = "fills/" + name + ".fill";
    std::string base = mProjectPath.empty() ? mDataRoot : dirOf(mProjectPath);
    saveFillBin(base + "/" + rel, mFillPos, mFillNrm);
    // register in the model
    Fill fi;
    fi.name = name; fi.file = rel;
    fi.thickness = mSkinParams.inflate; fi.smooth = mSkinParams.smooth;
    fi.bodyScale = mSkinParams.bodyScale; fi.cell = mSkinParams.cell;
    fi.muscleScale = mSkinParams.muscleScale; fi.includeMuscles = mSkinParams.includeMuscles;
    if (mModel.fills.empty()) fi.isDefault = true;
    mModel.fills.push_back(fi);
    mActiveFill = (int)mModel.fills.size() - 1;
    bindSkin(mFillPos, mFillNrm);
    mStatus = "Fill '" + name + "' generated: " + std::to_string(mFillPos.size()/3) + " triangles";
}

void App::applyFill(int idx) {
    if (idx < 0 || idx >= (int)mModel.fills.size()) return;
    const Fill& f = mModel.fills[idx];
    std::string base = mProjectPath.empty() ? mDataRoot : dirOf(mProjectPath);
    std::vector<V3> pos, nrm;
    if (!loadFillBin(base + "/" + f.file, pos, nrm)) { mStatus = "Cannot load fill: " + f.file; return; }
    // adopt its params (so binding uses the same body scale)
    mSkinParams.inflate = (float)f.thickness; mSkinParams.smooth = (float)f.smooth;
    mSkinParams.bodyScale = (float)f.bodyScale; mSkinParams.cell = (float)f.cell;
    mSkinParams.muscleScale = (float)f.muscleScale; mSkinParams.includeMuscles = f.includeMuscles;
    bindSkin(pos, nrm);
    mActiveFill = idx;
    mStatus = "Fill '" + f.name + "' applied";
}
void App::setDefaultFill(int idx) {
    for (auto& f : mModel.fills) f.isDefault = false;
    if (idx >= 0 && idx < (int)mModel.fills.size()) mModel.fills[idx].isDefault = true;
}
void App::deleteFill(int idx) {
    if (idx < 0 || idx >= (int)mModel.fills.size()) return;
    mModel.fills.erase(mModel.fills.begin() + idx);
    if (mActiveFill == idx) { mActiveFill = -1; mShowSkin = false; mSkinLocalPos.clear(); }
    else if (mActiveFill > idx) mActiveFill--;
}
void App::applyDefaultFill() {
    for (int i = 0; i < (int)mModel.fills.size(); i++)
        if (mModel.fills[i].isDefault) { applyFill(i); return; }
}

void App::updateSkinPose() {
    size_t nv = mSkinLocalPos.size();
    if (nv == 0) return;
    std::vector<V3> pos(nv), nrm(nv);
    int nb = (int)mModel.skeleton.size();
    std::vector<M4> M(nb);
    for (int b = 0; b < nb; b++) M[b] = liveBodyMatrix(mModel.skeleton[b]);
    for (size_t i = 0; i < nv; i++) {
        int b = mSkinBone[i]; if (b < 0 || b >= nb) b = 0;
        pos[i] = mulPoint(M[b], mSkinLocalPos[i]);
        nrm[i] = normalize(mulDir(M[b], mSkinLocalNrm[i]));
    }
    mRen.setSkinMesh(pos, nrm);
}

void App::seedLightsIfEmpty() {
    if (!mModel.lights.empty()) return;
    Light key;  key.name = "Key"; key.type = 0; key.dir = {0.4, 0.9, 0.5};
    key.color = {1.0, 0.97, 0.9}; key.intensity = 1.1;
    Light fill; fill.name = "Fill"; fill.type = 0; fill.dir = {-0.5, 0.4, -0.4};
    fill.color = {0.6, 0.7, 1.0}; fill.intensity = 0.6;
    Light back; back.name = "Back"; back.type = 0; back.dir = {0.0, 0.3, -0.9};
    back.color = {0.8, 0.8, 0.9}; back.intensity = 0.4;
    mModel.lights = { key, fill, back };
    mModel.ambient = 0.4;
}

void App::newModel() {
    mModel = Model();
    seedLightsIfEmpty();
    mMeshes.clear();
    mSel.clear();
    mUndo.clear(); mRedo.clear();
    mProjectPath.clear();
    mStatus = "New empty model";
}

void App::loadProjectPath(const std::string& path) {
    std::string err;
    auto m = Model::LoadMass(path, &err);
    if (m) { mModel = *m; mProjectPath = path; mDataRoot = deriveRoot(path);
             seedLightsIfEmpty(); loadMeshes(); applyDefaultFill(); mSel.clear(); mUndo.clear(); mRedo.clear();
             mStatus = "Opened: " + path; }
    else mStatus = "Open error: " + err;
}

// ---- undo ----
void App::snapshot() {
    mUndo.push_back(mModel);
    if (mUndo.size() > 100) mUndo.pop_front();
    mRedo.clear();
}
void App::undo() {
    if (mUndo.empty()) return;
    mRedo.push_back(mModel);
    mModel = mUndo.back(); mUndo.pop_back();
    mSel.clear(); mStatus = "Undone";
}
void App::redo() {
    if (mRedo.empty()) return;
    mUndo.push_back(mModel);
    mModel = mRedo.back(); mRedo.pop_back();
    mSel.clear(); mStatus = "Redone";
}

// ---- helpers ----
Node* App::selNode() {
    if ((mSel.type == SelType::Body || mSel.type == SelType::Joint) &&
        mSel.index >= 0 && mSel.index < (int)mModel.skeleton.size())
        return &mModel.skeleton[mSel.index];
    return nullptr;
}
Muscle* App::selMuscle() {
    if ((mSel.type == SelType::Muscle || mSel.type == SelType::Waypoint) &&
        mSel.index >= 0 && mSel.index < (int)mModel.muscles.size())
        return &mModel.muscles[mSel.index];
    return nullptr;
}
M4 App::nodeBodyMatrix(const Node& n) const { return fromTransform(n.body.t); }
M4 App::restBodyMatrix(int i) const {
    if (i >= 0 && i < (int)mRestBody.size()) return fromTransform(mRestBody[i]);
    if (i >= 0 && i < (int)mModel.skeleton.size()) return fromTransform(mModel.skeleton[i].body.t);
    return M4{};
}
V3 App::worldOfWaypoint(const Waypoint& w) const { return V3(w.p); }
V3 App::lightHandle(const Light& L) const {
    if (L.type == 1) return V3(L.dir);                       // point: position itself
    V3 center{0, 1.2f, 0};
    return center + normalize(V3(L.dir)) * 1.8f;             // directional: handle offset along dir
}

static std::string deriveRoot(const std::string& fileInData) {
    std::string dir = fileInData;
    size_t s = dir.find_last_of("/\\"); if (s != std::string::npos) dir = dir.substr(0, s);
    // if dir ends with "data", root is its parent
    size_t s2 = dir.find_last_of("/\\");
    std::string base = (s2 != std::string::npos) ? dir.substr(s2 + 1) : dir;
    if (base == "data" && s2 != std::string::npos) return dir.substr(0, s2);
    return dir;
}

M4 App::liveBodyMatrix(const Node& n) const {
    if (mSimActive) {
        auto it = mLivePose.find(n.id);
        if (it != mLivePose.end()) return fromTransform(it->second);
    }
    return fromTransform(n.body.t);
}
V3 App::liveWaypoint(const Waypoint& w) const {
    if (mSimActive) {
        const Node* n = mModel.findNode(w.body);
        auto it = mLivePose.find(w.body);
        if (n && it != mLivePose.end()) {
            M4 delta = mul(fromTransform(it->second), rigidInverse(fromTransform(n->body.t)));
            return mulPoint(delta, V3(w.p));
        }
    }
    return V3(w.p);
}

static V3 catmullRom(const V3& p0, const V3& p1, const V3& p2, const V3& p3, float t) {
    float t2 = t*t, t3 = t2*t;
    return (p1*2.0f + (p2 - p0)*t
            + (p0*2.0f - p1*5.0f + p2*4.0f - p3)*t2
            + (p1*3.0f - p0 - p2*3.0f + p3)*t3) * 0.5f;
}

// Natural muscle volume: smooth Catmull-Rom tube through the (live) waypoints so it
// curves and follows the body as joints move. Parallel-transport frames (no twist),
// radial smooth normals, fusiform radius from PCSA (r=sqrt(PCSA/pi); PCSA=f0/sigma).
void App::drawMuscleTube(const Muscle& mu, const V3& color) {
    int nw = (int)mu.waypoints.size();
    if (nw < 2) return;
    std::vector<V3> cp(nw);
    for (int i = 0; i < nw; i++) cp[i] = liveWaypoint(mu.waypoints[i]);
    auto CP = [&](int i){ return cp[i < 0 ? 0 : (i >= nw ? nw-1 : i)]; };

    // sample a smooth centerline
    const int SUB = 6;
    std::vector<V3> S;
    S.reserve((nw-1)*SUB + 1);
    for (int seg = 0; seg < nw-1; seg++) {
        V3 p0=CP(seg-1), p1=CP(seg), p2=CP(seg+1), p3=CP(seg+2);
        for (int j = 0; j < SUB; j++) S.push_back(catmullRom(p0,p1,p2,p3, (float)j/SUB));
    }
    S.push_back(cp[nw-1]);
    int ns = (int)S.size();
    if (ns < 2) return;

    double pcsa = mu.pcsa_cm2 > 0.0 ? mu.pcsa_cm2 : mu.f0 / mModel.meta.specific_tension_N_cm2;
    float rBase = (float)std::sqrt(std::max(pcsa, 0.01) / 3.14159265) * 0.01f;
    if (rBase < 0.003f) rBase = 0.003f;
    if (rBase > 0.035f) rBase = 0.035f;

    const int RING = 9;
    // initial frame
    V3 tan0 = normalize(S[1] - S[0]);
    V3 up = (std::fabs(tan0.y) > 0.9f) ? V3{1,0,0} : V3{0,1,0};
    V3 u = normalize(cross(tan0, up));
    V3 v = normalize(cross(tan0, u));
    std::vector<V3> prevRing, prevN;
    for (int i = 0; i < ns; i++) {
        V3 tan = normalize(i==0 ? S[1]-S[0] : (i==ns-1 ? S[ns-1]-S[ns-2] : S[i+1]-S[i-1]));
        // parallel transport: keep u perpendicular to the new tangent
        u = normalize(u - tan * dot(u, tan));
        if (length(u) < 1e-4f) { up = (std::fabs(tan.y) > 0.9f) ? V3{1,0,0} : V3{0,1,0}; u = normalize(cross(tan, up)); }
        v = normalize(cross(tan, u));
        float s = (float)i / (ns - 1);
        float profile = 0.30f + 0.70f * std::sin(3.14159265f * s);
        float r = rBase * profile;
        std::vector<V3> ring(RING), rn(RING);
        for (int k = 0; k < RING; k++) {
            float a = 6.2831853f * k / RING;
            V3 dir = u * std::cos(a) + v * std::sin(a);
            ring[k] = S[i] + dir * r;
            rn[k] = dir;                    // radial smooth normal
        }
        if (i > 0) {
            for (int k = 0; k < RING; k++) {
                int k2 = (k + 1) % RING;
                mRen.triSmooth(prevRing[k], prevN[k], prevRing[k2], prevN[k2], ring[k2], rn[k2], color);
                mRen.triSmooth(prevRing[k], prevN[k], ring[k2], rn[k2], ring[k], rn[k], color);
            }
        }
        prevRing = ring; prevN = rn;
    }
}

V3 App::selectionCenter() const {
    switch (mSel.type) {
        case SelType::Body:
        case SelType::Joint: {
            if (mSel.index >= 0 && mSel.index < (int)mModel.skeleton.size()) {
                const Node& n = mModel.skeleton[mSel.index];
                if (mSel.type == SelType::Joint) return mulPoint(fromTransform(n.joint.t), {0,0,0});
                return mulPoint(liveBodyMatrix(n), {0,0,0});
            }
        } break;
        case SelType::Muscle: {
            if (mSel.index >= 0 && mSel.index < (int)mModel.muscles.size()) {
                const Muscle& mu = mModel.muscles[mSel.index];
                if (!mu.waypoints.empty()) {
                    V3 s{0,0,0};
                    for (auto& w : mu.waypoints) s = s + liveWaypoint(w);
                    return s * (1.0f / (float)mu.waypoints.size());
                }
            }
        } break;
        case SelType::Waypoint: {
            const Muscle* mu = (mSel.index>=0 && mSel.index<(int)mModel.muscles.size()) ? &mModel.muscles[mSel.index] : nullptr;
            if (mu && mSel.sub >= 0 && mSel.sub < (int)mu->waypoints.size())
                return liveWaypoint(mu->waypoints[mSel.sub]);
        } break;
        case SelType::Light: {
            if (mSel.index >= 0 && mSel.index < (int)mModel.lights.size())
                return lightHandle(mModel.lights[mSel.index]);
        } break;
        default: break;
    }
    return mCam.target;
}

// Move the selected element in the camera's view plane (screen horizontal/vertical),
// keeping its depth — no change along the view direction.
void App::moveSelectionScreen(float dx, float dy) {
    if (mSel.type == SelType::None) return;
    V3 eye = mCam.eye();
    V3 fwd = normalize(mCam.target - eye);
    V3 right = normalize(cross(fwd, V3{0,1,0}));
    if (length(right) < 1e-4f) right = V3{1,0,0};
    V3 up = cross(right, fwd);
    V3 center = selectionCenter();
    float d = dot(center - eye, fwd); if (d < 0.05f) d = 0.05f;
    float wpp = 2.0f * d * std::tan(mCam.fovy * 0.5f) / std::max(mVpH, 1.0f);
    V3 mv = right * (dx * wpp) + up * (-dy * wpp);   // screen -> world, depth preserved
    auto add = [](Vec3& t, const V3& m){ t = { t[0]+m.x, t[1]+m.y, t[2]+m.z }; };

    if (mSel.type == SelType::Body && mSel.index < (int)mModel.skeleton.size())
        add(mModel.skeleton[mSel.index].body.t.translation, mv);
    else if (mSel.type == SelType::Joint && mSel.index < (int)mModel.skeleton.size())
        add(mModel.skeleton[mSel.index].joint.t.translation, mv);
    else if (mSel.type == SelType::Waypoint) {
        if (Muscle* mu = selMuscle())
            if (mSel.sub >= 0 && mSel.sub < (int)mu->waypoints.size())
                add(mu->waypoints[mSel.sub].p, mv);
    } else if (mSel.type == SelType::Muscle) {
        if (Muscle* mu = selMuscle())
            for (auto& w : mu->waypoints) add(w.p, mv);   // move the whole muscle
    } else if (mSel.type == SelType::Light && mSel.index < (int)mModel.lights.size()) {
        Light& L = mModel.lights[mSel.index];
        if (L.type == 1) add(L.dir, mv);
        else { V3 h = lightHandle(L) + mv; V3 dir = normalize(h - V3{0,1.2f,0}); L.dir = {dir.x,dir.y,dir.z}; }
    }
}

void App::resetView() {
    // frame the character: default camera + target at model center
    V3 center{0, 1.0f, 0};
    if (!mModel.skeleton.empty()) {
        V3 s{0,0,0};
        for (const auto& n : mModel.skeleton) s = s + V3(n.body.t.translation);
        center = s * (1.0f / (float)mModel.skeleton.size());
    }
    Camera def;                 // defaults (yaw/pitch/dist/fov)
    def.target = center;
    mCam = def;
    if (mSimActive) mSim.requestReset();   // reposition the character too
    mStatus = "View reset";
}

void App::toggleSim() {
    if (mSimActive) {
        mSim.stop();
        mSimActive = false;
        mLivePose.clear();
        mStatus = "Simulation stopped (edit mode)";
    } else {
        if (mModel.skeleton.empty()) { mStatus = "Load a model first"; return; }
        std::string tmp = mDataRoot + "/build/editor_tmp";
#ifdef _WIN32
        std::string mk = "cmd /c mkdir \"" + tmp + "\" 2>nul";
        std::system(mk.c_str());
#endif
        mSim.configure(mDataRoot, tmp);
        mSim.setModel(mModel);
        mSim.setMode(SimBridge::Kinematic);
        mSim.setActivation(mActivation);
        mSim.setPaused(false);
        mSim.start();
        mSimActive = true;
        mSimSigApplied = mSimSigPending = simSignature();
        mStatus = "Simulation started";
    }
}

// hash the model fields that change what the live sim builds (skeleton geometry,
// muscle Hill params + routing, joint PD/limits, env timing/actuation).
static inline void hashD(size_t& h, double d) {
    size_t b = 0; std::memcpy(&b, &d, sizeof(double));
    h ^= b + 0x9e3779b97f4a7c15ULL + (h << 6) + (h >> 2);
}
static inline void hashV3(size_t& h, const Vec3& v) { hashD(h, v[0]); hashD(h, v[1]); hashD(h, v[2]); }
static inline void hashS(size_t& h, const std::string& s) {
    h ^= std::hash<std::string>{}(s) + 0x9e3779b97f4a7c15ULL + (h << 6) + (h >> 2);
}
size_t App::simSignature() const {
    size_t h = 1469598103934665603ULL;
    for (const auto& n : mModel.skeleton) {
        hashS(h, n.id); hashS(h, n.parent); hashD(h, n.endeffector ? 1 : 0);
        hashS(h, n.body.type); hashD(h, n.body.mass); hashV3(h, n.body.size);
        hashD(h, n.body.radius); hashD(h, n.body.height); hashD(h, n.body.contact ? 1 : 0);
        hashV3(h, n.body.t.translation); for (double x : n.body.t.linear) hashD(h, x);
        hashS(h, n.joint.type); hashS(h, n.joint.bvh);
        hashV3(h, n.joint.lower); hashV3(h, n.joint.upper); hashV3(h, n.joint.axis);
        hashV3(h, n.joint.kp); hashV3(h, n.joint.kv);
        hashV3(h, n.joint.t.translation); for (double x : n.joint.t.linear) hashD(h, x);
    }
    for (const auto& m : mModel.muscles) {
        hashS(h, m.name); hashD(h, m.f0); hashD(h, m.lm); hashD(h, m.lt);
        hashD(h, m.pen_angle); hashD(h, m.lmax);
        for (const auto& w : m.waypoints) { hashS(h, w.body); hashV3(h, w.p); }
    }
    const auto& e = mModel.env;
    hashS(h, e.actuator); hashD(h, e.defaultKp); hashD(h, e.defaultKv); hashD(h, e.damping);
    hashD(h, e.simHz); hashD(h, e.controlHz); hashD(h, e.actionScale);
    hashS(h, e.bvh_file); hashD(h, e.residual ? 1 : 0); hashD(h, e.enforceSymmetry ? 1 : 0);
    return h;
}

void App::applySimNow() {
    if (!mSimActive) return;
    mSim.setModel(mModel);
    mSimSigApplied = mSimSigPending = simSignature();
    mStatus = "Applied edits to sim";
}

void App::maybeAutoApplySim() {
    if (!mSimActive) return;
    size_t cur = simSignature();
    double now = glfwGetTime();
    if (cur != mSimSigPending) { mSimSigPending = cur; mSimDirtyAt = now; }  // an edit landed
    if (mLiveSimApply && mSimSigApplied != mSimSigPending && (now - mSimDirtyAt) > 0.4) {
        mSim.setModel(mModel);            // rebuild Environment from the edited model
        mSimSigApplied = mSimSigPending;
    }
}

// ---- scene ----
void App::drawScene() {
    if (mVpW < 1 || mVpH < 1) return;
    syncMeshes();  // keep visuals in sync with the model's obj/type edits
    // render into the offscreen FBO at 2x (supersampling AA), shown via ImGui::Image
    const float ss = 2.0f;
    mRen.beginTarget((int)(mVpW * ss), (int)(mVpH * ss));

    float aspect = mVpW / mVpH;
    M4 vp = mul(mCam.proj(aspect), mCam.view());
    mRen.begin(vp, mCam.eye());

    // scene lights -> renderer
    std::vector<Renderer::LightGPU> gpu;
    for (const auto& L : mModel.lights) {
        if (!L.enabled) continue;
        gpu.push_back({ L.type, V3(L.dir),
            V3{(float)(L.color[0]*L.intensity), (float)(L.color[1]*L.intensity), (float)(L.color[2]*L.intensity)} });
    }
    mRen.setLights(gpu, (float)mModel.ambient);

    if (mSimActive && mSim.running()) mLivePose = mSim.pose();

    // checkerboard floor: 0.5 m tiles, dark, to scale (10 m -> 20 divs = 0.5 m)
    if (mDrawGrid) mRen.checkerGround(10.0f, 20, {0.16f,0.16f,0.19f}, {0.09f,0.09f,0.11f});

    auto clamp01 = [](double v){ return v<0?0.0f:(v>1?1.0f:(float)v); };

    // skeleton (solid bodies + edge definition)
    for (int i = 0; i < (int)mModel.skeleton.size(); i++) {
        const Node& n = mModel.skeleton[i];
        M4 bm = liveBodyMatrix(n);
        bool selBody = (mSel.type == SelType::Body && mSel.index == i);
        bool selJoint = (mSel.type == SelType::Joint && mSel.index == i);
        V3 base = { clamp01(n.body.color[0]), clamp01(n.body.color[1]), clamp01(n.body.color[2]) };
        if (base.x + base.y + base.z < 0.05f) base = {0.30f, 0.40f, 0.95f}; // default blue
        base = { base.x*0.6f+0.15f, base.y*0.6f+0.18f, base.z*0.6f+0.35f }; // brighten toward reference blue
        V3 col = selBody ? V3{1.0f, 0.75f, 0.15f} : base;
        V3 he = n.body.type == "Box"
            ? V3{(float)n.body.size[0]*0.5f, (float)n.body.size[1]*0.5f, (float)n.body.size[2]*0.5f}
            : V3{(float)n.body.radius, (float)n.body.radius, (float)n.body.radius};
        if (mShowBones) {
            const MeshData* md = (i < (int)mMeshes.size()) ? &mMeshes[i] : nullptr;
            if (mShowMesh && md && md->gpuId != 0) {
                // mesh authored in rest world space -> follow current body via delta (on GPU)
                M4 delta = mul(bm, rigidInverse(restBodyMatrix(i)));
                V3 mcol = selBody ? V3{1.0f, 0.75f, 0.2f} : V3{0.86f, 0.80f, 0.72f}; // bone/ivory
                mRen.drawMeshGPU(md->gpuId, delta, mcol);
            } else {
                if (n.body.type == "Sphere") mRen.solidSphere(mulPoint(bm,{0,0,0}), (float)n.body.radius, col);
                else mRen.solidBox(bm, he, col);
                if (selBody) mRen.wireBox(bm, he, {1.0f,0.9f,0.3f}, 1.0f);
            }
        }
        // joint marker
        if (mShowJoints) {
            M4 jm = fromTransform(n.joint.t);
            mRen.axes(jm, selJoint ? 0.12f : 0.05f);
            if (selJoint) mRen.point(mulPoint(jm, {0,0,0}), {1.0f,0.8f,0.2f});
        }
    }

    // muscles
    if (mShowMuscles) {
        for (int i = 0; i < (int)mModel.muscles.size(); i++) {
            const Muscle& mu = mModel.muscles[i];
            bool selM = ((mSel.type == SelType::Muscle || mSel.type == SelType::Waypoint) && mSel.index == i);
            V3 mcol = selM ? V3{1.0f, 0.85f, 0.3f} : V3{0.72f, 0.12f, 0.13f};
            if (mMuscleVolume)
                drawMuscleTube(mu, mcol);
            else
                for (size_t k = 1; k < mu.waypoints.size(); k++)
                    mRen.line(liveWaypoint(mu.waypoints[k-1]), liveWaypoint(mu.waypoints[k]), mcol, 0.9f);
            if (mShowWaypoints)
                for (size_t k = 0; k < mu.waypoints.size(); k++) {
                    bool selW = (mSel.type == SelType::Waypoint && mSel.index == i && mSel.sub == (int)k);
                    mRen.point(liveWaypoint(mu.waypoints[k]), selW ? V3{0.2f,1.0f,1.0f} : V3{1.0f,0.6f,0.2f});
                }
        }
    }

    // generated tissue fill with rigid skinning so it follows the skeleton
    if (mShowSkin && !mSkinLocalPos.empty()) {
        updateSkinPose();                           // deform to the live pose each frame
        M4 ident;
        mRen.drawSkin(ident, V3{0.86f, 0.66f, 0.56f});
    }

    // light markers (visible + clickable icons)
    for (int i = 0; mShowLightMarkers && i < (int)mModel.lights.size(); i++) {
        const Light& L = mModel.lights[i];
        bool sel = (mSel.type == SelType::Light && mSel.index == i);
        V3 c = sel ? V3{1.0f, 1.0f, 0.2f}
                   : V3{ (float)L.color[0], (float)L.color[1], (float)L.color[2] };
        if (!L.enabled) c = c * 0.35f;
        V3 h = lightHandle(L);
        mRen.solidSphere(h, sel ? 0.10f : 0.07f, c);       // emissive bulb
        // radiating rays
        const float r = 0.16f;
        mRen.line(h, h + V3{r,0,0}, c);  mRen.line(h, h + V3{-r,0,0}, c);
        mRen.line(h, h + V3{0,r,0}, c);  mRen.line(h, h + V3{0,-r,0}, c);
        mRen.line(h, h + V3{0,0,r}, c);  mRen.line(h, h + V3{0,0,-r}, c);
        if (L.type == 0) { // directional: line from handle toward the model center (aim)
            V3 center{0, 1.2f, 0};
            mRen.line(h, center, c, 0.5f);
        }
    }

    mRen.flush();
    mRen.endTarget();
}

// ---- picking ----
void App::pickAt(double mx, double my) {
    float aspect = mVpW / mVpH;
    float nx = ((float)(mx - mVpX) / mVpW) * 2.0f - 1.0f;
    float ny = 1.0f - ((float)(my - mVpY) / mVpH) * 2.0f;
    Ray ray = mCam.rayFromNDC(nx, ny, aspect);

    float best = 1e30f;
    Selection hit;

    // lights (handles) — only if their markers are visible
    for (int i = 0; mShowLightMarkers && i < (int)mModel.lights.size(); i++) {
        float t = raySphere(ray, lightHandle(mModel.lights[i]), 0.14f);
        if (t > 0 && t < best) { best = t; hit = {SelType::Light, i, -1}; }
    }

    // waypoints (small spheres) — only if visible
    for (int i = 0; mShowWaypoints && i < (int)mModel.muscles.size(); i++) {
        const Muscle& mu = mModel.muscles[i];
        for (int k = 0; k < (int)mu.waypoints.size(); k++) {
            float t = raySphere(ray, liveWaypoint(mu.waypoints[k]), 0.02f);
            if (t > 0 && t < best) { best = t; hit = {SelType::Waypoint, i, k}; }
        }
    }
    // bodies — ray-cast against the ACTUAL mesh triangles (what you see); box fallback
    for (int i = 0; mShowBones && i < (int)mModel.skeleton.size(); i++) {
        const Node& n = mModel.skeleton[i];
        const MeshData* md = (i < (int)mMeshes.size()) ? &mMeshes[i] : nullptr;
        if (mShowMesh && md && md->gpuId != 0) {
            M4 delta = mul(liveBodyMatrix(n), rigidInverse(restBodyMatrix(i)));
            const auto& P = md->pos;
            for (size_t k = 0; k + 2 < P.size(); k += 3) {
                float t = rayTriangle(ray, mulPoint(delta,P[k]), mulPoint(delta,P[k+1]), mulPoint(delta,P[k+2]));
                if (t > 0 && t < best) { best = t; hit = {SelType::Body, i, -1}; }
            }
        } else {
            V3 he = n.body.type == "Box"
                ? V3{(float)n.body.size[0]*0.5f, (float)n.body.size[1]*0.5f, (float)n.body.size[2]*0.5f}
                : V3{(float)n.body.radius,(float)n.body.radius,(float)n.body.radius};
            float t = rayOBB(ray, nodeBodyMatrix(n), he);
            if (t > 0 && t < best) { best = t; hit = {SelType::Body, i, -1}; }
        }
    }
    // muscle segments — compete by distance-along-ray so a muscle in front wins over the bone
    for (int i = 0; mShowMuscles && i < (int)mModel.muscles.size(); i++) {
        const Muscle& mu = mModel.muscles[i];
        for (size_t k = 1; k < mu.waypoints.size(); k++) {
            float tt;
            float d = raySegmentDist(ray, liveWaypoint(mu.waypoints[k-1]), liveWaypoint(mu.waypoints[k]), tt);
            if (d < 0.012f && tt > 0 && tt < best) { best = tt; hit = {SelType::Muscle, i, -1}; }
        }
    }
    mSel = hit;
    if (hit.type == SelType::Light) mSelLight = hit.index;
}

// ---- gizmo ----
void App::drawGizmo() {
    if (mSel.type == SelType::None) return;
    if (mSimActive) return; // no structural editing while simulating
    ImGuizmo::SetOrthographic(false);
    ImGuizmo::SetDrawlist(ImGui::GetForegroundDrawList());
    ImGuizmo::SetRect(mVpX, mVpY, mVpW, mVpH);
    float aspect = mVpW / mVpH;
    M4 view = mCam.view();
    M4 proj = mCam.proj(aspect);

    if (Node* n = selNode()) {
        bool isJoint = (mSel.type == SelType::Joint);
        Transform& t = isJoint ? n->joint.t : n->body.t;
        M4 model = fromTransform(t);
        static M4 before; static bool wasUsing = false;
        if (ImGuizmo::Manipulate(view.data(), proj.data(),
                (ImGuizmo::OPERATION)mGizmoOp, (ImGuizmo::MODE)mGizmoMode, model.data())) {
            // write back column-major model -> Transform (row-major Mat3 + Vec3)
            t.linear = { model.m[0],model.m[4],model.m[8],
                         model.m[1],model.m[5],model.m[9],
                         model.m[2],model.m[6],model.m[10] };
            t.translation = { model.m[12], model.m[13], model.m[14] };
        }
        if (ImGuizmo::IsUsing() && !wasUsing) { snapshot(); }
        wasUsing = ImGuizmo::IsUsing();
    } else if (mSel.type == SelType::Light) {
        if (mSel.index >= 0 && mSel.index < (int)mModel.lights.size()) {
            Light& L = mModel.lights[mSel.index];
            V3 h = lightHandle(L);
            M4 model; model.m[12]=h.x; model.m[13]=h.y; model.m[14]=h.z;
            static bool wasUsingL = false;
            if (ImGuizmo::Manipulate(view.data(), proj.data(),
                    ImGuizmo::TRANSLATE, ImGuizmo::WORLD, model.data())) {
                V3 np{model.m[12], model.m[13], model.m[14]};
                if (L.type == 1) { L.dir = {np.x, np.y, np.z}; }
                else { V3 d = normalize(np - V3{0,1.2f,0}); L.dir = {d.x, d.y, d.z}; }
            }
            if (ImGuizmo::IsUsing() && !wasUsingL) snapshot();
            wasUsingL = ImGuizmo::IsUsing();
        }
    } else if (mSel.type == SelType::Waypoint) {
        if (Muscle* mu = selMuscle()) {
            if (mSel.sub >= 0 && mSel.sub < (int)mu->waypoints.size()) {
                Waypoint& w = mu->waypoints[mSel.sub];
                M4 model; model.m[12]=(float)w.p[0]; model.m[13]=(float)w.p[1]; model.m[14]=(float)w.p[2];
                static bool wasUsing = false;
                if (ImGuizmo::Manipulate(view.data(), proj.data(),
                        ImGuizmo::TRANSLATE, ImGuizmo::WORLD, model.data())) {
                    w.p = { model.m[12], model.m[13], model.m[14] };
                }
                if (ImGuizmo::IsUsing() && !wasUsing) snapshot();
                wasUsing = ImGuizmo::IsUsing();
            }
        }
    }
}

// ---- actions ----
void App::openMass() {
    std::string p = openFileDialog();
    if (p.empty()) return;
    std::string err;
    auto m = Model::LoadMass(p, &err);
    if (m) { mModel = *m; mProjectPath = p; mDataRoot = deriveRoot(p);
             seedLightsIfEmpty(); loadMeshes(); applyDefaultFill(); mSel.clear(); mUndo.clear(); mRedo.clear();
             mStatus = "Opened: " + p; }
    else mStatus = "Open error: " + err;
}
void App::saveMass(bool as) {
    std::string p = mProjectPath;
    if (as || p.empty()) p = saveFileDialog();
    if (p.empty()) return;
    std::string err;
    if (mModel.SaveMass(p, &err)) { mProjectPath = p; mStatus = "Saved: " + p; }
    else mStatus = "Save error: " + err;
}
void App::bootstrap() {
    std::string meta = openFileDialog("GaitNet env (*.xml)\0*.xml\0All\0*.*\0");
    if (meta.empty()) return;
    // data_root = parent of the folder containing env.xml (assets live under data/)
    std::string dir = meta;
    size_t s = dir.find_last_of("/\\"); if (s != std::string::npos) dir = dir.substr(0, s); // .../data
    size_t s2 = dir.find_last_of("/\\"); std::string root = (s2 != std::string::npos) ? dir.substr(0, s2) : ".";
    std::string err;
    auto m = BootstrapFromLegacy(meta, root, &err);
    if (m) { mModel = *m; mDataRoot = root; seedLightsIfEmpty(); loadMeshes(); mSel.clear(); mUndo.clear(); mRedo.clear();
             mStatus = "Bootstrap OK: " + std::to_string(mModel.skeleton.size()) + " nodes, "
                       + std::to_string(mModel.muscles.size()) + " muscles"; }
    else mStatus = "Bootstrap error: " + err;
}
void App::exportLegacy() {
    std::string dir = pickFolderDialog();
    if (dir.empty()) return;
    std::string err;
    if (ExportToLegacy(mModel, dir, &err)) mStatus = "Exported to: " + dir;
    else mStatus = "Export error: " + err;
}
void App::addBody() {
    snapshot();
    Node n;
    n.id = "NewBody" + std::to_string(mModel.skeleton.size());
    if (!mModel.skeleton.empty()) n.parent = mModel.skeleton.front().id;
    n.joint.type = "Ball";
    mModel.skeleton.push_back(n);
    mSel = {SelType::Body, (int)mModel.skeleton.size()-1, -1};
    mStatus = "Body added";
}
void App::removeSelected() {
    if (mSel.type == SelType::Body || mSel.type == SelType::Joint) {
        if (mSel.index < 0 || mSel.index >= (int)mModel.skeleton.size()) return;
        snapshot();
        mModel.skeleton.erase(mModel.skeleton.begin() + mSel.index);
        mSel.clear(); mStatus = "Node removed";
    } else if (mSel.type == SelType::Muscle) {
        if (mSel.index < 0 || mSel.index >= (int)mModel.muscles.size()) return;
        snapshot();
        mModel.muscles.erase(mModel.muscles.begin() + mSel.index);
        mSel.clear(); mStatus = "Muscle removed";
    } else if (mSel.type == SelType::Waypoint) {
        if (Muscle* mu = selMuscle()) {
            if (mu->waypoints.size() > 2 && mSel.sub >= 0) {
                snapshot();
                mu->waypoints.erase(mu->waypoints.begin() + mSel.sub);
                mSel = {SelType::Muscle, mSel.index, -1}; mStatus = "Waypoint removed";
            }
        }
    }
}
void App::importOsim() {
    std::string p = openFileDialog("OpenSim model (*.osim)\0*.osim\0All\0*.*\0");
    if (p.empty()) return;
    std::string err;
    mAtlas = ImportOsimMuscles(p, &err);
    mStatus = mAtlas.empty() ? ("Atlas: " + err)
                             : ("Atlas loaded: " + std::to_string(mAtlas.size()) + " reference muscles");
}

// Import a rigless mesh (obj/glb/fbx/stl/3mf/dxf via assimp) as the model's skin:
// deindex to a triangle soup, auto-scale/align to the skeleton's rest bbox, then
// bind each vertex to the nearest body so it deforms with the live sim.
void App::importSkinMesh() {
    if (mModel.skeleton.empty()) { mStatus = "Load a model first"; return; }
    std::string path = openFileDialog(
        "Mesh (obj/glb/fbx/stl/3mf/dxf)\0*.obj;*.glb;*.gltf;*.fbx;*.stl;*.3mf;*.dxf;*.ply\0All\0*.*\0");
    if (path.empty()) return;

    Assimp::Importer imp;
    const aiScene* sc = imp.ReadFile(path,
        aiProcess_Triangulate | aiProcess_GenSmoothNormals |
        aiProcess_PreTransformVertices | aiProcess_JoinIdenticalVertices);
    if (!sc || !sc->mRootNode || sc->mNumMeshes == 0) {
        mStatus = std::string("Mesh import failed: ") + imp.GetErrorString(); return;
    }

    std::vector<V3> pos, nrm;
    for (unsigned mi = 0; mi < sc->mNumMeshes; mi++) {
        const aiMesh* m = sc->mMeshes[mi];
        if (!m->HasNormals()) continue;
        for (unsigned fi = 0; fi < m->mNumFaces; fi++) {
            const aiFace& f = m->mFaces[fi];
            if (f.mNumIndices != 3) continue;
            for (int k = 0; k < 3; k++) {
                unsigned idx = f.mIndices[k];
                const aiVector3D& v = m->mVertices[idx];
                const aiVector3D& n = m->mNormals[idx];
                pos.push_back(V3{ v.x, v.y, v.z });
                nrm.push_back(V3{ n.x, n.y, n.z });
            }
        }
    }
    if (pos.empty()) { mStatus = "Mesh has no triangles/normals"; return; }

    mSkinOrigPos = std::move(pos);
    mSkinOrigNrm = std::move(nrm);
    mSkinRotDeg = V3{ 0,0,0 };
    mSkinUserScale = 1.0f; mSkinUserOff = V3{ 0,0,0 };
    rebuildSkinFromOrig();
    mShowSkin = true; mShowMesh = false;
    mStatus = "Skin imported: " + std::to_string(mSkinOrigPos.size() / 3) + " tris (use Rotate if lying down)";
}

// apply the user orientation fix to the original mesh, recompute the height-match
// auto-fit against the skeleton, then place + bind.
void App::rebuildSkinFromOrig() {
    if (mSkinOrigPos.empty()) return;
    // rotation matrix from Euler XYZ (degrees): R = Rz * Ry * Rx
    const float d2r = 3.14159265358979323846f / 180.0f;
    float cx = std::cos(mSkinRotDeg.x*d2r), sx = std::sin(mSkinRotDeg.x*d2r);
    float cy = std::cos(mSkinRotDeg.y*d2r), sy = std::sin(mSkinRotDeg.y*d2r);
    float cz = std::cos(mSkinRotDeg.z*d2r), sz = std::sin(mSkinRotDeg.z*d2r);
    auto rot = [&](const V3& v) {
        // Rx
        V3 a{ v.x, cx*v.y - sx*v.z, sx*v.y + cx*v.z };
        // Ry
        V3 b{ cy*a.x + sy*a.z, a.y, -sy*a.x + cy*a.z };
        // Rz
        return V3{ cz*b.x - sz*b.y, sz*b.x + cz*b.y, b.z };
    };
    mSkinRawPos.resize(mSkinOrigPos.size());
    mSkinRawNrm.resize(mSkinOrigNrm.size());
    for (size_t i = 0; i < mSkinOrigPos.size(); i++) { mSkinRawPos[i] = rot(mSkinOrigPos[i]); mSkinRawNrm[i] = rot(mSkinOrigNrm[i]); }

    // mesh bbox (after rotation)
    V3 mn = mSkinRawPos[0], mx = mSkinRawPos[0];
    for (const auto& p : mSkinRawPos) {
        mn.x = std::min(mn.x, p.x); mn.y = std::min(mn.y, p.y); mn.z = std::min(mn.z, p.z);
        mx.x = std::max(mx.x, p.x); mx.y = std::max(mx.y, p.y); mx.z = std::max(mx.z, p.z);
    }
    // skeleton rest bbox
    V3 smn{ 1e30f,1e30f,1e30f }, smx{ -1e30f,-1e30f,-1e30f };
    for (int b = 0; b < (int)mModel.skeleton.size(); b++) {
        const Node& n = mModel.skeleton[b];
        V3 c{ (float)n.body.t.translation[0], (float)n.body.t.translation[1], (float)n.body.t.translation[2] };
        float he = n.body.type == "Box"
            ? 0.5f * (float)std::max({ n.body.size[0], n.body.size[1], n.body.size[2] })
            : (float)n.body.radius;
        smn.x = std::min(smn.x, c.x - he); smn.y = std::min(smn.y, c.y - he); smn.z = std::min(smn.z, c.z - he);
        smx.x = std::max(smx.x, c.x + he); smx.y = std::max(smx.y, c.y + he); smx.z = std::max(smx.z, c.z + he);
    }
    float meshH = std::max(1e-4f, mx.y - mn.y);
    float skelH = std::max(1e-4f, smx.y - smn.y);
    mSkinAutoScale = skelH / meshH;
    mSkinAutoMeshCtr = V3{ (mn.x + mx.x) * 0.5f, 0, (mn.z + mx.z) * 0.5f };
    mSkinAutoMeshMinY = mn.y;
    mSkinAutoSkelCtr = V3{ (smn.x + smx.x) * 0.5f, 0, (smn.z + smx.z) * 0.5f };
    mSkinAutoSkelMinY = smn.y;
    applySkinPlacement();
}

// re-place the raw imported mesh using auto-fit * user scale + user offset, then bind
void App::applySkinPlacement() {
    if (mSkinRawPos.empty()) return;
    float s = mSkinAutoScale * std::max(0.01f, mSkinUserScale);
    std::vector<V3> pos(mSkinRawPos.size());
    for (size_t i = 0; i < mSkinRawPos.size(); i++) {
        const V3& r = mSkinRawPos[i];
        pos[i].x = (r.x - mSkinAutoMeshCtr.x) * s + mSkinAutoSkelCtr.x + mSkinUserOff.x;
        pos[i].z = (r.z - mSkinAutoMeshCtr.z) * s + mSkinAutoSkelCtr.z + mSkinUserOff.z;
        pos[i].y = (r.y - mSkinAutoMeshMinY) * s + mSkinAutoSkelMinY + mSkinUserOff.y;
    }
    bindSkin(pos, mSkinRawNrm);   // normals re-normalized inside the bind
}

void App::clearSkin() {
    mSkinOrigPos.clear(); mSkinOrigNrm.clear();
    mSkinRawPos.clear(); mSkinRawNrm.clear();
    mSkinLocalPos.clear(); mSkinLocalNrm.clear(); mSkinBone.clear();
    mShowSkin = false;
    mStatus = "Skin cleared";
}

// Morph the skeleton's limb chains so the bones fit inside the imported character
// mesh (the mesh defines the look; bones adapt to it). Heuristic: for each proximal
// joint, scale its whole subtree about that joint so the chain end reaches the mesh
// silhouette extent along the chain's dominant axis. No landmarks/ML — approximate.
void App::fitSkeletonToSkin() {
    if (mSkinRawPos.empty() || mModel.skeleton.empty()) { mStatus = "Import a skin first"; return; }
    snapshot();

    // placed skin verts (same transform as applySkinPlacement, skeleton space)
    float s = mSkinAutoScale * std::max(0.01f, mSkinUserScale);
    std::vector<V3> P(mSkinRawPos.size());
    for (size_t i = 0; i < P.size(); i++) {
        const V3& r = mSkinRawPos[i];
        P[i].x = (r.x - mSkinAutoMeshCtr.x) * s + mSkinAutoSkelCtr.x + mSkinUserOff.x;
        P[i].z = (r.z - mSkinAutoMeshCtr.z) * s + mSkinAutoSkelCtr.z + mSkinUserOff.z;
        P[i].y = (r.y - mSkinAutoMeshMinY) * s + mSkinAutoSkelMinY + mSkinUserOff.y;
    }

    auto idOf = [&](const std::string& id) -> int {
        for (int i = 0; i < (int)mModel.skeleton.size(); i++) if (mModel.skeleton[i].id == id) return i;
        return -1;
    };
    // subtree (indices) rooted at a node, via parent links
    auto subtree = [&](const std::string& root) {
        std::vector<int> out; std::vector<std::string> frontier{root};
        while (!frontier.empty()) {
            std::string cur = frontier.back(); frontier.pop_back();
            int ci = idOf(cur); if (ci >= 0) out.push_back(ci);
            for (auto& n : mModel.skeleton) if (n.parent == cur) frontier.push_back(n.id);
        }
        return out;
    };
    // scale every node in `nodes` about pivot by factor k (uniform); shrink body boxes too
    auto scaleAbout = [&](const std::vector<int>& nodes, const Vec3& piv, double k) {
        for (int i : nodes) {
            Node& n = mModel.skeleton[i];
            for (Transform* t : { &n.body.t, &n.joint.t }) {
                t->translation[0] = piv[0] + (t->translation[0] - piv[0]) * k;
                t->translation[1] = piv[1] + (t->translation[1] - piv[1]) * k;
                t->translation[2] = piv[2] + (t->translation[2] - piv[2]) * k;
            }
            for (int a = 0; a < 3; a++) n.body.size[a] *= k;
        }
    };
    auto meshReachAxis = [&](const Vec3& piv, int axis, double sgn, double band) {
        double best = 0;
        for (const auto& v : P) {
            double dy = std::fabs((axis == 1 ? v.z : v.y) - (axis == 1 ? piv[2] : piv[1]));
            double dperp = std::fabs((axis == 0 ? v.z : v.x) - (axis == 0 ? piv[2] : piv[0]));
            if (std::max(dy, dperp) > band) continue;
            double comp = (axis == 0 ? v.x : v.y) - piv[axis];
            double d = comp * sgn;
            if (d > best) best = d;
        }
        return best;
    };

    int fitted = 0;
    // arms: scale ShoulderX subtree about the shoulder joint so the hand reaches the mesh span (X axis)
    for (const char* side : { "R", "L" }) {
        int sh = idOf(std::string("Shoulder") + side);
        int hand = idOf(std::string("Hand") + side);
        if (sh < 0 || hand < 0) continue;
        Vec3 piv = mModel.skeleton[sh].joint.t.translation;
        double handX = mModel.skeleton[hand].body.t.translation[0];
        double sgn = (handX - piv[0]) >= 0 ? 1.0 : -1.0;
        double reach = std::fabs(handX - piv[0]);
        double meshReach = meshReachAxis(piv, 0, sgn, 0.28);
        if (reach > 1e-3 && meshReach > 1e-3) {
            double k = std::clamp(meshReach / reach, 0.3, 1.3);
            scaleAbout(subtree(std::string("Shoulder") + side), piv, k);
            fitted++;
        }
    }
    // legs: scale FemurX subtree about the hip joint so the foot reaches the mesh bottom (Y axis)
    for (const char* side : { "R", "L" }) {
        int fem = idOf(std::string("Femur") + side);
        int foot = idOf(std::string("Talus") + side);
        if (fem < 0 || foot < 0) continue;
        Vec3 piv = mModel.skeleton[fem].joint.t.translation;
        double footY = mModel.skeleton[foot].body.t.translation[1];
        double reach = std::fabs(footY - piv[1]);       // hip -> ankle
        double meshReach = meshReachAxis(piv, 1, -1.0, 0.20); // downward
        if (reach > 1e-3 && meshReach > 1e-3) {
            double k = std::clamp(meshReach / reach, 0.5, 1.4);
            scaleAbout(subtree(std::string("Femur") + side), piv, k);
            fitted++;
        }
    }

    syncMeshes();            // reload bone OBJ meshes at the new transforms
    applySkinPlacement();    // rebind skin to the adapted skeleton
    if (mSimActive) mSim.setModel(mModel);
    mStatus = "Fitted skeleton to skin (" + std::to_string(fitted) + " limb chains)";
}

// import button + scale/offset sliders for the imported skin (Tools panel)
void App::drawSkinControls() {
    ImGui::SeparatorText("Skin (imported mesh)");
    if (ImGui::Button("Import mesh as skin...")) importSkinMesh();
    if (mSkinRawPos.empty()) {
        ImGui::TextDisabled("obj / glb / fbx / stl / 3mf");
        return;
    }
    ImGui::SameLine();
    if (ImGui::Checkbox("Show", &mShowSkin)) {}
    // orientation fix (GLB/FBX from generators are often Z-up or face a different way)
    bool rot = false;
    ImGui::TextUnformatted("Orient:"); ImGui::SameLine();
    if (ImGui::Button("X+90")) { mSkinRotDeg.x += 90; rot = true; } ImGui::SameLine();
    if (ImGui::Button("Y+90")) { mSkinRotDeg.y += 90; rot = true; } ImGui::SameLine();
    if (ImGui::Button("Z+90")) { mSkinRotDeg.z += 90; rot = true; } ImGui::SameLine();
    if (ImGui::Button("Rot 0")) { mSkinRotDeg = V3{0,0,0}; rot = true; }
    if (ImGui::SliderFloat3("Rot (deg)", &mSkinRotDeg.x, -180.0f, 180.0f, "%.0f")) rot = true;
    if (rot) rebuildSkinFromOrig();
    bool changed = false;
    changed |= ImGui::SliderFloat("Skin scale", &mSkinUserScale, 0.2f, 3.0f, "%.2fx");
    changed |= ImGui::SliderFloat("Offset X", &mSkinUserOff.x, -1.0f, 1.0f, "%.3f");
    changed |= ImGui::SliderFloat("Offset Y", &mSkinUserOff.y, -1.0f, 1.0f, "%.3f");
    changed |= ImGui::SliderFloat("Offset Z", &mSkinUserOff.z, -1.0f, 1.0f, "%.3f");
    if (ImGui::Button("Reset fit")) { mSkinUserScale = 1.0f; mSkinUserOff = V3{0,0,0}; changed = true; }
    ImGui::SameLine();
    if (ImGui::Button("Clear skin")) { clearSkin(); return; }
    if (ImGui::Button("Fit skeleton to skin")) fitSkeletonToSkin();
    ImGui::SameLine(); ImGui::TextDisabled("(adapt bones)");
    if (changed) applySkinPlacement();
    ImGui::TextDisabled("%zu tris  |  auto x%.2f", mSkinRawPos.size()/3, mSkinAutoScale);
}
void App::applyAtlas(const AtlasEntry& a) {
    if (Muscle* mu = selMuscle()) {
        snapshot();
        mu->f0 = a.f0;                 // physiological max isometric force
        mu->pen_angle = a.pen_angle;   // pennation angle (rad)
        if (mu->latin.empty()) mu->latin = a.name;
        mStatus = "Atlas applied to " + mu->name + " (f0=" + std::to_string((int)a.f0) + "N)";
    } else mStatus = "Select a muscle first";
}
// mirror a body/muscle name across sides: FemurR<->FemurL, R_x<->L_x
static std::string mirrorName(const std::string& s) {
    if (s.size() >= 2 && s.rfind("R_", 0) == 0) return "L_" + s.substr(2);
    if (s.size() >= 2 && s.rfind("L_", 0) == 0) return "R_" + s.substr(2);
    if (!s.empty() && s.back() == 'R') return s.substr(0, s.size()-1) + "L";
    if (!s.empty() && s.back() == 'L') return s.substr(0, s.size()-1) + "R";
    return s;
}
void App::mirrorSelectedMuscle() {
    Muscle* mu = selMuscle();
    if (!mu) { mStatus = "Select a muscle to mirror"; return; }
    snapshot();
    Muscle mir = *mu;
    mir.name = mirrorName(mu->name);
    mir.side = (mu->side == "R") ? "L" : (mu->side == "L" ? "R" : mu->side);
    if (!mu->antagonist.empty()) mir.antagonist = mirrorName(mu->antagonist);
    for (auto& w : mir.waypoints) { w.p[0] = -w.p[0]; w.body = mirrorName(w.body); }
    // replace existing mirror if present, else append
    if (Muscle* existing = mModel.findMuscle(mir.name)) *existing = mir;
    else mModel.muscles.push_back(mir);
    mStatus = "Mirrored -> " + mir.name;
}
void App::duplicateSelectedMuscle() {
    if (Muscle* mu = selMuscle()) {
        snapshot();
        Muscle copy = *mu; copy.name += "_copy";
        mModel.muscles.push_back(copy);
        mSel = {SelType::Muscle, (int)mModel.muscles.size()-1, -1};
        mStatus = "Muscle duplicated";
    }
}

} // namespace ed
