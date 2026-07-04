#pragma once
#include "MassModel.h"
#include "Camera.h"
#include "Renderer.h"
#include "SimBridge.h"
#include "TrainBridge.h"
#include "SkinGen.h"
#include <vector>
#include <string>
#include <deque>
#include <map>
#include <atomic>
#include <thread>

struct GLFWwindow;

namespace ed {

// per-node visual mesh (OBJ triangle soup) in rest world space
struct MeshData { std::vector<V3> pos, nrm; unsigned gpuId = 0; std::string objName; };

enum class SelType { None, Body, Joint, Muscle, Waypoint, Light };
struct Selection {
    SelType type = SelType::None;
    int index = -1;   // node or muscle index
    int sub   = -1;   // waypoint index (when Waypoint)
    void clear(){ type = SelType::None; index = -1; sub = -1; }
};

class App {
public:
    bool init(GLFWwindow* win);
    void frame();           // one UI+render frame
    void shutdown();
    void loadProjectPath(const std::string& path); // load a specific .mass
    void generateSkin();                            // generate the skin now (public trigger)
    void startKinematicSim();                        // start live BVH playback (public trigger)
    void beginFill(const std::string& name);         // start a fill generation (public trigger)

private:
    GLFWwindow* mWin = nullptr;
    Model mModel;
    Camera mCam;
    Renderer mRen;
    Selection mSel;
    std::string mProjectPath;      // current .mass path
    std::string mStatus = "Ready";
    bool mDrawGrid = false;
    bool mShowMuscles = true;
    bool mShowBones = true;
    bool mShowJoints = false;
    bool mShowWaypoints = false;
    bool mShowLightMarkers = false;
    bool mShowMesh = true;      // render OBJ body meshes instead of boxes
    bool mMuscleVolume = true;  // render muscles as fusiform volume tubes (not lines)
    bool mShowSkin = false;     // continuous skin (marching-cubes)
    SkinParams mSkinParams;
    void drawMuscleTube(const Muscle& mu, const V3& color);
    void regenerateSkin();
    void drawFillsPanel();
    // skin skinning: each vertex bound to nearest body, stored in that body's local space
    std::vector<V3> mSkinLocalPos, mSkinLocalNrm;
    std::vector<int> mSkinBone;
    void updateSkinPose();   // rebuild live skin verts from current body transforms
    void bindSkin(const std::vector<V3>& pos, const std::vector<V3>& nrm); // bind + upload

    // imported skin mesh (raw, pre-placement) + adjustable fit so scale/offset can
    // be tweaked live without reloading the file
    std::vector<V3> mSkinOrigPos, mSkinOrigNrm;  // exactly as loaded (never mutated)
    V3 mSkinRotDeg{0,0,0};                        // user orientation fix (Euler XYZ, degrees)
    void rebuildSkinFromOrig();                   // apply rotation -> raw, recompute auto-fit, place
    std::vector<V3> mSkinRawPos, mSkinRawNrm;
    float mSkinAutoScale = 1.0f;           // height-match scale computed at import
    V3 mSkinAutoMeshCtr{0,0,0}; float mSkinAutoMeshMinY = 0.0f;
    V3 mSkinAutoSkelCtr{0,0,0}; float mSkinAutoSkelMinY = 0.0f;
    float mSkinUserScale = 1.0f;           // user multiplier on the auto scale
    V3 mSkinUserOff{0,0,0};                // user world offset (m)
    void applySkinPlacement();             // re-place raw mesh -> bindSkin
    void clearSkin();
    void drawSkinControls();               // scale/offset sliders
    void fitSkeletonToSkin();              // morph bone chains to the imported mesh silhouette

    // fills (generated skin/tissue envelopes), async generation with progress
    std::atomic<float> mFillProgress{0.0f};
    std::atomic<bool>  mFillDone{false};
    bool mFillRunning = false;
    std::thread mFillThread;
    std::vector<V3> mFillPos, mFillNrm;         // worker output
    char mFillNewName[64] = "fill";
    int mActiveFill = -1;
    std::string fillsDir() const;               // <project>/fills
    void startFillGeneration(const std::string& name);
    void finalizeFill();                        // called from frame when worker done
    void applyFill(int idx);
    void setDefaultFill(int idx);
    void deleteFill(int idx);
    void applyDefaultFill();

    std::vector<MeshData> mMeshes;  // per-node visual meshes (OBJ), rest world space
    void loadMeshes();
    void syncMeshes();   // reload meshes when the model's obj/type/count changes

    // live simulation
    SimBridge mSim;
    bool mSimActive = false;
    float mActivation = 0.0f;
    std::string mDataRoot = ".";
    std::map<std::string, Transform> mLivePose;
    void toggleSim();
    // auto-apply model edits to the running sim (debounced rebuild)
    bool mLiveSimApply = true;
    size_t mSimSigPending = 0, mSimSigApplied = 0;
    double mSimDirtyAt = -1.0;
    size_t simSignature() const;   // hash of the sim-relevant model fields
    void maybeAutoApplySim();      // push model to SimBridge when edits settle
    void applySimNow();            // force a rebuild from the current model
    M4 liveBodyMatrix(const Node& n) const;   // current world (sim pose if simulating, else body.t)
    M4 restBodyMatrix(int i) const;           // fixed authoring rest pose of body i
    std::vector<Transform> mRestBody;         // captured rest pose (mesh/skin authoring reference)
    V3 liveWaypoint(const Waypoint& w) const;

    // viewport rect (for picking + gizmo)
    float mVpX=0, mVpY=0, mVpW=1, mVpH=1;
    int mGizmoOp = 7;              // ImGuizmo TRANSLATE default
    int mGizmoMode = 1;           // WORLD

    // undo/redo (full-model snapshots)
    std::deque<Model> mUndo, mRedo;
    void snapshot();              // push current state to undo
    void undo(); void redo();

    // scene
    void drawScene();
    void drawGizmo();
    V3 worldOfWaypoint(const Waypoint& w) const;
    M4 nodeBodyMatrix(const Node& n) const;
    V3 lightHandle(const Light& L) const;   // 3D position of a light's handle
    V3 selectionCenter() const;             // world center of the current selection
    void resetView();                       // reset camera + pose (Ctrl+1)
    void moveSelectionScreen(float dx, float dy); // drag selection in the camera view plane

    // picking
    void pickAt(double mx, double my);

    // anatomy atlas (.osim)
    std::vector<AtlasEntry> mAtlas;
    char mAtlasFilter[64] = "";

    // panels
    void drawMenuBar();
    void drawTopToolbar();   // horizontal action button bar under the menu
    void drawToolbar();
    void drawTree();
    void drawProperties();
    void drawValidation();
    void drawAtlas();
    void drawTrain();
    void drawGaitNet();      // env.xml settings + <parameter> block editor
    void drawLights();

    // scene lighting
    int mSelLight = -1;
    void seedLightsIfEmpty();

    // training telemetry bridge (asio)
    TrainBridge mTrain;
    void startTraining();

    // import an external (rigless) mesh as the skin, auto-scaled to the skeleton
    void importSkinMesh();
    float mSkinImportScale = 1.0f;

    // anatomy actions
    void importOsim();
    void applyAtlas(const AtlasEntry& a);
    void mirrorSelectedMuscle();

    // actions
    void newModel();
    void openMass();
    void saveMass(bool as);
    void bootstrap();
    void exportLegacy();
    void addBody();
    void removeSelected();
    void duplicateSelectedMuscle();

    // helpers
    Node* selNode();
    Muscle* selMuscle();
};

} // namespace ed
