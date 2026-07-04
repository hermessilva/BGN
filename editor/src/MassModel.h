#pragma once
// In-memory representation of a unified MASS musculoskeletal project (.mass).
// Holds skeleton + muscles + motions + training config + anatomical metadata.
// Load/Save as JSON; Bootstrap/Export bridge the legacy human.xml/muscle284.xml.
#include <string>
#include <vector>
#include <array>
#include <optional>

namespace ed {

using Vec3 = std::array<double, 3>;
using Vec4 = std::array<double, 4>;
using Mat3 = std::array<double, 9>; // row-major

struct Transform {
    Mat3 linear{1,0,0, 0,1,0, 0,0,1};
    Vec3 translation{0,0,0};
};

// ---- Skeleton ----
struct Body {
    std::string type = "Box";   // Box | Sphere | Capsule
    double mass = 1.0;
    Vec3 size{0.1,0.1,0.1};      // Box
    double radius = 0.05;        // Sphere/Capsule
    double height = 0.1;         // Capsule
    std::string obj;             // visual mesh file (optional, "" = none)
    bool contact = false;
    Vec4 color{0.2,0.2,0.2,1.0};
    Transform t;                 // world transform of the body
};

struct Joint {
    std::string type = "Ball";   // Free | Ball | Revolute | Weld | Planar
    std::string bvh;             // mapped BVH channel name (optional)
    Vec3 lower{-2,-2,-2};        // Ball/Revolute limits (Revolute uses [0])
    Vec3 upper{ 2, 2, 2};
    Vec3 axis{1,0,0};            // Revolute
    Vec3 kp{0,0,0};              // per-joint PD stiffness (empty => env default); Revolute uses [0]
    Vec3 kv{0,0,0};              // per-joint PD damping   (empty => env default)
    bool hasKp = false, hasKv = false;
    Transform t;                 // world transform of the joint
};

struct Node {
    std::string id;              // unique name
    std::string parent;          // parent id, "" = root (None)
    bool endeffector = false;    // reward end-effector flag (human.xml endeffector="True")
    Body body;
    Joint joint;
};

// ---- Muscles ----
struct Waypoint {
    std::string body;            // anchoring body id
    Vec3 p{0,0,0};               // world position (as stored in muscle284.xml)
};

struct Muscle {
    std::string name;
    // runtime Hill parameters (exported to muscle284.xml)
    double f0 = 1000.0;          // max isometric force (N)
    double lm = 1.0;             // optimal fiber length (normalized in MASS)
    double lt = 0.2;             // tendon slack length
    double pen_angle = 0.0;      // pennation angle (rad)
    double lmax = -0.1;          // MASS lmax field
    std::vector<Waypoint> waypoints;
    // ---- anatomical metadata (editor-only, not exported to training) ----
    std::string latin;
    std::string pt;              // Portuguese name
    std::string group;           // functional group (e.g. knee_extensor)
    std::string side;            // "L" | "R" | "" (midline)
    std::string antagonist;      // name of antagonist muscle
    double pcsa_cm2 = 0.0;       // physiological cross-sectional area
};

// ---- Motion ----
struct Motion {
    std::string name;
    std::string bvh;             // relative path under data/
    bool cyclic = true;
};

// ---- Training ----
struct Training {
    bool use_muscle = true;
    int con_hz = 30;
    int sim_hz = 600;
    Vec4 reward_param{0.75, 0.1, 0.0, 0.15};
    std::string default_motion;  // motion name used for metadata bvh_file
};

struct Meta {
    std::string name = "human";
    std::string unit = "m";
    double specific_tension_N_cm2 = 60.0;
};

// Scene light (editor-only, persisted in .mass, not exported to training).
struct Light {
    std::string name = "Luz";
    int type = 0;                    // 0 = directional, 1 = point
    Vec3 dir{0.4, 0.9, 0.5};         // direction (directional) or position (point)
    Vec3 color{1, 1, 1};
    double intensity = 1.0;
    bool enabled = true;
};

// ---- GaitNet env.xml configuration (the master file of this project) ----
struct EnvConfig {
    // actuation / files
    std::string actuator = "mass";       // pd | torque | muscle | mass
    double defaultKp = 200.0;
    double defaultKv = 40.0;
    double damping = 0.1;
    std::string skel_file = "../data/skeleton_gaitnet_narrow_model.xml";
    std::string ground_file = "../data/ground.xml";
    std::string muscle_file = "../data/muscle_gaitnet.xml";
    // reference motion
    std::string bvh_file = "../data/motion/walk.bvh";
    bool bvh_symmetry = true;
    bool bvh_heightCalibration = true;
    bool cyclicbvh = true;
    // controller / simulation
    bool residual = true;
    int simHz = 480;
    int controlHz = 30;
    int inferencePerSim = 1;
    bool heightCalibration = true, heightCalibrationStrict = true;
    bool musclePoseOptimization = true; std::string musclePoseOptRot = "one_foot";
    bool enforceSymmetry = true;
    double actionScale = 0.04;
    double timeWarping = 0.001;
    bool stanceLearning = false;
    bool metabolicReward = true;
    // muscle advanced
    bool meshLbsWeight = false, useVelocityForce = false, useJointState = true;
    bool learningStd = true, hardPhaseClipping = true, softPhaseClipping = false;
    bool torqueClipping = false, includeJtPinSPD = false, useNormalizedParamState = false;
    // rewards
    std::string eoeType = "tuple";       // time | tuple
    std::string rewardType = "gaitnet";  // deepmimic | gaitnet
    double headLinearAccWeight = 4.0, headRotWeight = 4.0, stepWeight = 2.0;
    double metabolicWeight = 8.0, avgVelWeight = 6.0;
};

// One entry of the <parameter> block (anatomy/gait parameterization ranges).
struct ParamRange {
    std::string name;                    // e.g. "stride", "FemurL", "global"
    double min = 1.0, max = 1.0, def = 1.0;
    std::string sampling;                // "uniform" | "" (unset)
    bool hasDefault = false, hasSampling = false;
};
struct Parameters {
    std::vector<ParamRange> gait;         // stride, cadence
    std::vector<ParamRange> skeleton;     // global + per-body scaling
    std::vector<ParamRange> torsion;      // per-lower-body torsion
    std::vector<ParamRange> muscle_length;
    std::vector<ParamRange> muscle_force;
};

// Reference muscle parameters imported from an OpenSim .osim (atlas).
struct AtlasEntry {
    std::string name;
    double f0 = 0, lm = 0, lt = 0, pen_angle = 0;
    std::string origin_body, insertion_body;
};

// A generated body "fill" (skin/tissue envelope): generation parameters + cached mesh
// file (saved under <project>/fills/). One can be marked as the default.
struct Fill {
    std::string name = "fill";
    std::string file;               // relative mesh path, e.g. "fills/fill.obj"
    bool isDefault = false;
    // generation parameters (mirror of SkinParams)
    double thickness = 0.015, smooth = 0.06, bodyScale = 1.15, cell = 0.02, muscleScale = 1.0;
    bool includeMuscles = true;
};

struct Model {
    int version = 1;
    Meta meta;
    std::vector<Node> skeleton;
    std::vector<Muscle> muscles;
    std::vector<Motion> motions;
    Training training;
    EnvConfig env;                   // GaitNet env.xml master config
    Parameters params;               // <parameter> block (gait/skeleton/torsion/muscle ranges)
    std::vector<Light> lights;       // scene lighting (editor)
    double ambient = 0.25;           // global ambient term
    std::vector<Fill> fills;         // generated skin/tissue fills

    // lookups
    Node* findNode(const std::string& id);
    const Node* findNode(const std::string& id) const;
    Muscle* findMuscle(const std::string& name);

    // JSON project (.mass)
    static std::optional<Model> LoadMass(const std::string& path, std::string* err = nullptr);
    bool SaveMass(const std::string& path, std::string* err = nullptr) const;
};

} // namespace ed
