#include "MassModel.h"
#include <nlohmann/json.hpp>
#include <fstream>
#include <sstream>
#include <unordered_set>
#include <cstdio>

using json = nlohmann::json;

namespace mass {

// ---- lookups ----
Node* Model::findNode(const std::string& id) {
    for (auto& n : skeleton) if (n.id == id) return &n;
    return nullptr;
}
const Node* Model::findNode(const std::string& id) const {
    for (auto& n : skeleton) if (n.id == id) return &n;
    return nullptr;
}
Muscle* Model::findMuscle(const std::string& name) {
    for (auto& m : muscles) if (m.name == name) return &m;
    return nullptr;
}

int Model::assignUids() {
    std::unordered_set<std::string> used;
    for (const auto& n : skeleton) if (!n.uid.empty()) used.insert(n.uid);
    for (const auto& m : muscles)  if (!m.uid.empty()) used.insert(m.uid);
    int assigned = 0;
    int nc = 0, mc = 0;
    char buf[16];
    auto gen = [&](const char* pfx, int& ctr) {
        std::string u;
        do { std::snprintf(buf, sizeof(buf), "%s%04d", pfx, ++ctr); u = buf; } while (used.count(u));
        used.insert(u); return u;
    };
    for (auto& n : skeleton) if (n.uid.empty()) { n.uid = gen("N", nc); ++assigned; }
    for (auto& m : muscles)  if (m.uid.empty()) { m.uid = gen("M", mc); ++assigned; }
    return assigned;
}

// ---- json helpers ----
static json j(const Vec3& v) { return json::array({v[0], v[1], v[2]}); }
static json j(const Vec4& v) { return json::array({v[0], v[1], v[2], v[3]}); }
static json j(const Mat3& m) { json a = json::array(); for (double x : m) a.push_back(x); return a; }
static Vec3 v3(const json& a) { return {a[0], a[1], a[2]}; }
static Vec4 v4(const json& a) { return {a[0], a[1], a[2], a[3]}; }
static Mat3 m3(const json& a) { Mat3 m{}; for (int i=0;i<9;i++) m[i]=a[i]; return m; }
static json j(const Transform& t) { return json{{"linear", j(t.linear)}, {"translation", j(t.translation)}}; }
static Transform tf(const json& o) { Transform t; t.linear = m3(o["linear"]); t.translation = v3(o["translation"]); return t; }

// safe getters
template <class T> static T get(const json& o, const char* k, T def) {
    auto it = o.find(k); return it != o.end() ? it->get<T>() : def;
}

// ---- serialize ----
static json toJson(const Model& m) {
    json root;
    root["massVersion"] = m.version;
    root["meta"] = { {"name", m.meta.name}, {"unit", m.meta.unit},
                     {"specificTension_N_cm2", m.meta.specific_tension_N_cm2} };

    json sk = json::array();
    for (const auto& n : m.skeleton) {
        json body = { {"type", n.body.type}, {"mass", n.body.mass},
                      {"obj", n.body.obj}, {"contact", n.body.contact},
                      {"color", j(n.body.color)}, {"transform", j(n.body.t)} };
        if (n.body.type == "Box") body["size"] = j(n.body.size);
        if (n.body.type == "Sphere") body["radius"] = n.body.radius;
        if (n.body.type == "Capsule") { body["radius"] = n.body.radius; body["height"] = n.body.height; }
        json joint = { {"type", n.joint.type}, {"bvh", n.joint.bvh},
                       {"transform", j(n.joint.t)} };
        if (n.joint.type == "Ball") { joint["lower"] = j(n.joint.lower); joint["upper"] = j(n.joint.upper); }
        else if (n.joint.type == "Revolute") {
            joint["axis"] = j(n.joint.axis);
            joint["lower"] = n.joint.lower[0]; joint["upper"] = n.joint.upper[0];
        }
        if (n.joint.hasKp) joint["kp"] = j(n.joint.kp);
        if (n.joint.hasKv) joint["kv"] = j(n.joint.kv);
        json jn = { {"id", n.id}, {"parent", n.parent}, {"body", body}, {"joint", joint} };
        if (!n.uid.empty()) jn["uid"] = n.uid;
        if (n.endeffector) jn["endeffector"] = true;
        sk.push_back(jn);
    }
    root["skeleton"] = sk;

    json ms = json::array();
    for (const auto& mu : m.muscles) {
        json wps = json::array();
        for (const auto& w : mu.waypoints) wps.push_back({ {"body", w.body}, {"p", j(w.p)} });
        json muj = {
            {"name", mu.name},
            {"hill", { {"f0", mu.f0}, {"lm", mu.lm}, {"lt", mu.lt},
                       {"pen_angle", mu.pen_angle}, {"lmax", mu.lmax} }},
            {"waypoints", wps},
            {"anatomy", { {"latin", mu.latin}, {"pt", mu.pt}, {"group", mu.group},
                          {"side", mu.side}, {"antagonist", mu.antagonist}, {"pcsa_cm2", mu.pcsa_cm2} }}
        };
        if (!mu.uid.empty()) muj["uid"] = mu.uid;
        ms.push_back(muj);
    }
    root["muscles"] = ms;

    json mo = json::array();
    for (const auto& x : m.motions) mo.push_back({ {"name", x.name}, {"bvh", x.bvh}, {"cyclic", x.cyclic} });
    root["motions"] = mo;

    root["training"] = {
        {"use_muscle", m.training.use_muscle}, {"con_hz", m.training.con_hz},
        {"sim_hz", m.training.sim_hz}, {"reward_param", j(m.training.reward_param)},
        {"default_motion", m.training.default_motion}
    };

    json lg = json::array();
    for (const auto& L : m.lights)
        lg.push_back({ {"name", L.name}, {"type", L.type}, {"dir", j(L.dir)},
                       {"color", j(L.color)}, {"intensity", L.intensity}, {"enabled", L.enabled} });
    root["scene"] = { {"ambient", m.ambient}, {"lights", lg} };

    json fl = json::array();
    for (const auto& f : m.fills)
        fl.push_back({ {"name", f.name}, {"file", f.file}, {"default", f.isDefault},
                       {"thickness", f.thickness}, {"smooth", f.smooth}, {"bodyScale", f.bodyScale},
                       {"cell", f.cell}, {"muscleScale", f.muscleScale}, {"includeMuscles", f.includeMuscles} });
    root["fills"] = fl;

    json gr = json::array();
    for (const auto& g : m.grooms)
        gr.push_back({ {"name", g.name}, {"region", g.region}, {"length", g.length},
                       {"stiffness", g.stiffness}, {"damping", g.damping}, {"density", g.density},
                       {"guides", g.guides}, {"segments", g.segments}, {"dynamic", g.dynamic} });
    root["grooms"] = gr;

    const auto& e = m.env;
    root["env"] = {
        {"actuator", e.actuator}, {"defaultKp", e.defaultKp}, {"defaultKv", e.defaultKv},
        {"damping", e.damping}, {"skel_file", e.skel_file}, {"ground_file", e.ground_file},
        {"muscle_file", e.muscle_file}, {"bvh_file", e.bvh_file}, {"bvh_symmetry", e.bvh_symmetry},
        {"bvh_heightCalibration", e.bvh_heightCalibration}, {"cyclicbvh", e.cyclicbvh},
        {"residual", e.residual}, {"simHz", e.simHz}, {"controlHz", e.controlHz},
        {"inferencePerSim", e.inferencePerSim}, {"heightCalibration", e.heightCalibration},
        {"heightCalibrationStrict", e.heightCalibrationStrict},
        {"musclePoseOptimization", e.musclePoseOptimization}, {"musclePoseOptRot", e.musclePoseOptRot},
        {"enforceSymmetry", e.enforceSymmetry}, {"actionScale", e.actionScale},
        {"timeWarping", e.timeWarping}, {"stanceLearning", e.stanceLearning},
        {"metabolicReward", e.metabolicReward}, {"meshLbsWeight", e.meshLbsWeight},
        {"useVelocityForce", e.useVelocityForce}, {"useJointState", e.useJointState},
        {"learningStd", e.learningStd}, {"hardPhaseClipping", e.hardPhaseClipping},
        {"softPhaseClipping", e.softPhaseClipping}, {"torqueClipping", e.torqueClipping},
        {"includeJtPinSPD", e.includeJtPinSPD}, {"useNormalizedParamState", e.useNormalizedParamState},
        {"eoeType", e.eoeType}, {"rewardType", e.rewardType},
        {"headLinearAccWeight", e.headLinearAccWeight}, {"headRotWeight", e.headRotWeight},
        {"stepWeight", e.stepWeight}, {"metabolicWeight", e.metabolicWeight},
        {"avgVelWeight", e.avgVelWeight}
    };
    auto prlist = [](const std::vector<ParamRange>& v) {
        json a = json::array();
        for (const auto& p : v) {
            json o = { {"name", p.name}, {"min", p.min}, {"max", p.max} };
            if (p.hasDefault)  o["default"]  = p.def;
            if (p.hasSampling) o["sampling"] = p.sampling;
            a.push_back(o);
        }
        return a;
    };
    root["params"] = {
        {"gait", prlist(m.params.gait)}, {"skeleton", prlist(m.params.skeleton)},
        {"torsion", prlist(m.params.torsion)}, {"muscle_length", prlist(m.params.muscle_length)},
        {"muscle_force", prlist(m.params.muscle_force)}
    };
    return root;
}

static Model fromJson(const json& root) {
    Model m;
    m.version = get<int>(root, "massVersion", 1);
    if (root.contains("meta")) {
        const auto& me = root["meta"];
        m.meta.name = get<std::string>(me, "name", "human");
        m.meta.unit = get<std::string>(me, "unit", "m");
        m.meta.specific_tension_N_cm2 = get<double>(me, "specificTension_N_cm2", 60.0);
    }
    for (const auto& n : root.value("skeleton", json::array())) {
        Node nd;
        nd.id = n.value("id", "");
        nd.uid = n.value("uid", "");
        nd.parent = n.value("parent", "");
        nd.endeffector = n.value("endeffector", false);
        const auto& b = n["body"];
        nd.body.type = b.value("type", "Box");
        nd.body.mass = b.value("mass", 1.0);
        nd.body.obj = b.value("obj", "");
        nd.body.contact = b.value("contact", false);
        if (b.contains("color")) nd.body.color = v4(b["color"]);
        if (b.contains("size")) nd.body.size = v3(b["size"]);
        nd.body.radius = b.value("radius", 0.05);
        nd.body.height = b.value("height", 0.1);
        nd.body.t = tf(b["transform"]);
        const auto& jo = n["joint"];
        nd.joint.type = jo.value("type", "Ball");
        nd.joint.bvh = jo.value("bvh", "");
        nd.joint.t = tf(jo["transform"]);
        if (jo.contains("axis")) nd.joint.axis = v3(jo["axis"]);
        if (nd.joint.type == "Ball") {
            if (jo.contains("lower")) nd.joint.lower = v3(jo["lower"]);
            if (jo.contains("upper")) nd.joint.upper = v3(jo["upper"]);
        } else if (nd.joint.type == "Revolute") {
            nd.joint.lower[0] = jo.value("lower", -2.0);
            nd.joint.upper[0] = jo.value("upper", 2.0);
        }
        if (jo.contains("kp")) { nd.joint.kp = v3(jo["kp"]); nd.joint.hasKp = true; }
        if (jo.contains("kv")) { nd.joint.kv = v3(jo["kv"]); nd.joint.hasKv = true; }
        m.skeleton.push_back(std::move(nd));
    }
    for (const auto& mu : root.value("muscles", json::array())) {
        Muscle x;
        x.name = mu.value("name", "");
        x.uid = mu.value("uid", "");
        if (mu.contains("hill")) {
            const auto& h = mu["hill"];
            x.f0 = h.value("f0", 1000.0); x.lm = h.value("lm", 1.0);
            x.lt = h.value("lt", 0.2); x.pen_angle = h.value("pen_angle", 0.0);
            x.lmax = h.value("lmax", -0.1);
        }
        for (const auto& w : mu.value("waypoints", json::array()))
            x.waypoints.push_back({ w.value("body", ""), v3(w["p"]) });
        if (mu.contains("anatomy")) {
            const auto& a = mu["anatomy"];
            x.latin = a.value("latin", ""); x.pt = a.value("pt", "");
            x.group = a.value("group", ""); x.side = a.value("side", "");
            x.antagonist = a.value("antagonist", ""); x.pcsa_cm2 = a.value("pcsa_cm2", 0.0);
        }
        m.muscles.push_back(std::move(x));
    }
    for (const auto& x : root.value("motions", json::array()))
        m.motions.push_back({ x.value("name", ""), x.value("bvh", ""), x.value("cyclic", true) });
    if (root.contains("training")) {
        const auto& t = root["training"];
        m.training.use_muscle = t.value("use_muscle", true);
        m.training.con_hz = t.value("con_hz", 30);
        m.training.sim_hz = t.value("sim_hz", 600);
        if (t.contains("reward_param")) m.training.reward_param = v4(t["reward_param"]);
        m.training.default_motion = t.value("default_motion", "");
    }
    if (root.contains("scene")) {
        const auto& sc = root["scene"];
        m.ambient = sc.value("ambient", 0.25);
        for (const auto& L : sc.value("lights", json::array())) {
            Light li;
            li.name = L.value("name", "Luz");
            li.type = L.value("type", 0);
            if (L.contains("dir")) li.dir = v3(L["dir"]);
            if (L.contains("color")) li.color = v3(L["color"]);
            li.intensity = L.value("intensity", 1.0);
            li.enabled = L.value("enabled", true);
            m.lights.push_back(li);
        }
    }
    for (const auto& f : root.value("fills", json::array())) {
        Fill fi;
        fi.name = f.value("name", "fill");
        fi.file = f.value("file", "");
        fi.isDefault = f.value("default", false);
        fi.thickness = f.value("thickness", 0.015);
        fi.smooth = f.value("smooth", 0.06);
        fi.bodyScale = f.value("bodyScale", 1.15);
        fi.cell = f.value("cell", 0.02);
        fi.muscleScale = f.value("muscleScale", 1.0);
        fi.includeMuscles = f.value("includeMuscles", true);
        m.fills.push_back(fi);
    }
    for (const auto& g : root.value("grooms", json::array())) {
        GroomParams gp;
        gp.name = g.value("name", "groom");
        gp.region = g.value("region", "");
        gp.length = g.value("length", 0.3);
        gp.stiffness = g.value("stiffness", 0.5);
        gp.damping = g.value("damping", 0.1);
        gp.density = g.value("density", 1.0);
        gp.guides = g.value("guides", 128);
        gp.segments = g.value("segments", 24);
        gp.dynamic = g.value("dynamic", true);
        m.grooms.push_back(gp);
    }
    if (root.contains("env")) {
        const auto& e = root["env"]; auto& c = m.env;
        c.actuator = e.value("actuator", c.actuator);
        c.defaultKp = e.value("defaultKp", c.defaultKp); c.defaultKv = e.value("defaultKv", c.defaultKv);
        c.damping = e.value("damping", c.damping);
        c.skel_file = e.value("skel_file", c.skel_file); c.ground_file = e.value("ground_file", c.ground_file);
        c.muscle_file = e.value("muscle_file", c.muscle_file); c.bvh_file = e.value("bvh_file", c.bvh_file);
        c.bvh_symmetry = e.value("bvh_symmetry", c.bvh_symmetry);
        c.bvh_heightCalibration = e.value("bvh_heightCalibration", c.bvh_heightCalibration);
        c.cyclicbvh = e.value("cyclicbvh", c.cyclicbvh); c.residual = e.value("residual", c.residual);
        c.simHz = e.value("simHz", c.simHz); c.controlHz = e.value("controlHz", c.controlHz);
        c.inferencePerSim = e.value("inferencePerSim", c.inferencePerSim);
        c.heightCalibration = e.value("heightCalibration", c.heightCalibration);
        c.heightCalibrationStrict = e.value("heightCalibrationStrict", c.heightCalibrationStrict);
        c.musclePoseOptimization = e.value("musclePoseOptimization", c.musclePoseOptimization);
        c.musclePoseOptRot = e.value("musclePoseOptRot", c.musclePoseOptRot);
        c.enforceSymmetry = e.value("enforceSymmetry", c.enforceSymmetry);
        c.actionScale = e.value("actionScale", c.actionScale); c.timeWarping = e.value("timeWarping", c.timeWarping);
        c.stanceLearning = e.value("stanceLearning", c.stanceLearning);
        c.metabolicReward = e.value("metabolicReward", c.metabolicReward);
        c.meshLbsWeight = e.value("meshLbsWeight", c.meshLbsWeight);
        c.useVelocityForce = e.value("useVelocityForce", c.useVelocityForce);
        c.useJointState = e.value("useJointState", c.useJointState);
        c.learningStd = e.value("learningStd", c.learningStd);
        c.hardPhaseClipping = e.value("hardPhaseClipping", c.hardPhaseClipping);
        c.softPhaseClipping = e.value("softPhaseClipping", c.softPhaseClipping);
        c.torqueClipping = e.value("torqueClipping", c.torqueClipping);
        c.includeJtPinSPD = e.value("includeJtPinSPD", c.includeJtPinSPD);
        c.useNormalizedParamState = e.value("useNormalizedParamState", c.useNormalizedParamState);
        c.eoeType = e.value("eoeType", c.eoeType); c.rewardType = e.value("rewardType", c.rewardType);
        c.headLinearAccWeight = e.value("headLinearAccWeight", c.headLinearAccWeight);
        c.headRotWeight = e.value("headRotWeight", c.headRotWeight);
        c.stepWeight = e.value("stepWeight", c.stepWeight);
        c.metabolicWeight = e.value("metabolicWeight", c.metabolicWeight);
        c.avgVelWeight = e.value("avgVelWeight", c.avgVelWeight);
    }
    if (root.contains("params")) {
        const auto& p = root["params"];
        auto rd = [](const json& a, std::vector<ParamRange>& out) {
            for (const auto& o : a) {
                ParamRange r; r.name = o.value("name", "");
                r.min = o.value("min", 1.0); r.max = o.value("max", 1.0);
                if (o.contains("default"))  { r.def = o["default"].get<double>(); r.hasDefault = true; }
                if (o.contains("sampling")) { r.sampling = o["sampling"].get<std::string>(); r.hasSampling = true; }
                out.push_back(r);
            }
        };
        if (p.contains("gait")) rd(p["gait"], m.params.gait);
        if (p.contains("skeleton")) rd(p["skeleton"], m.params.skeleton);
        if (p.contains("torsion")) rd(p["torsion"], m.params.torsion);
        if (p.contains("muscle_length")) rd(p["muscle_length"], m.params.muscle_length);
        if (p.contains("muscle_force")) rd(p["muscle_force"], m.params.muscle_force);
    }
    return m;
}

std::optional<Model> Model::LoadMass(const std::string& path, std::string* err) {
    std::ifstream f(path);
    if (!f) { if (err) *err = "cannot open " + path; return std::nullopt; }
    try {
        json root; f >> root;
        return fromJson(root);
    } catch (const std::exception& e) {
        if (err) *err = e.what();
        return std::nullopt;
    }
}

bool Model::SaveMass(const std::string& path, std::string* err) const {
    std::ofstream f(path);
    if (!f) { if (err) *err = "cannot write " + path; return false; }
    try {
        f << toJson(*this).dump(2);
        return true;
    } catch (const std::exception& e) {
        if (err) *err = e.what();
        return false;
    }
}

} // namespace mass
