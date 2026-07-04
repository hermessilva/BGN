#include "Bootstrap.h"
#include <tinyxml2.h>
#include <fstream>
#include <sstream>
#include <cmath>
#include <cstdio>
#include <filesystem>

using namespace tinyxml2;
namespace fs = std::filesystem;

namespace ed {

// ---------- parsing helpers ----------
static std::vector<double> splitDoubles(const std::string& s) {
    std::vector<double> out; std::stringstream ss(s); double d;
    while (ss >> d) out.push_back(d);
    return out;
}
static Vec3 parseVec3(const char* s, Vec3 def = {0,0,0}) {
    if (!s) return def; auto v = splitDoubles(s);
    return v.size() >= 3 ? Vec3{v[0], v[1], v[2]} : def;
}
static Vec4 parseVec4(const char* s, Vec4 def = {0.2,0.2,0.2,1.0}) {
    if (!s) return def; auto v = splitDoubles(s);
    return v.size() >= 4 ? Vec4{v[0], v[1], v[2], v[3]} : def;
}
static Mat3 parseMat3(const char* s) {
    Mat3 m{1,0,0, 0,1,0, 0,0,1};
    if (!s) return m; auto v = splitDoubles(s);
    for (size_t i = 0; i < 9 && i < v.size(); i++) m[i] = v[i];
    return m;
}
static Transform parseTransform(XMLElement* parent) {
    Transform t;
    if (!parent) return t;
    XMLElement* tr = parent->FirstChildElement("Transformation");
    if (!tr) return t;
    t.linear = parseMat3(tr->Attribute("linear"));
    t.translation = parseVec3(tr->Attribute("translation"));
    return t;
}
static bool boolText(XMLElement* e, bool def = false) {
    if (!e || !e->GetText()) return def;
    std::string s = e->GetText();
    return s.find("true") != std::string::npos || s == "1";
}
static double dblText(XMLElement* e, double def) {
    if (!e || !e->GetText()) return def; try { return std::stod(e->GetText()); } catch (...) { return def; }
}
static int intText(XMLElement* e, int def) {
    if (!e || !e->GetText()) return def; try { return std::stoi(e->GetText()); } catch (...) { return def; }
}
static std::string txt(XMLElement* e, const std::string& def = "") {
    return (e && e->GetText()) ? std::string(e->GetText()) : def;
}
static std::string trim(std::string s) {
    size_t a = s.find_first_not_of(" \t\r\n"); size_t b = s.find_last_not_of(" \t\r\n");
    return (a == std::string::npos) ? "" : s.substr(a, b - a + 1);
}

// ---------- skeleton ----------
static void loadSkeleton(const std::string& path, Model& m, std::string* err) {
    XMLDocument doc;
    if (doc.LoadFile(path.c_str()) != XML_SUCCESS) { if (err) *err = "cannot load skeleton " + path; return; }
    XMLElement* skel = doc.FirstChildElement("Skeleton");
    if (!skel) { if (err) *err = "no <Skeleton> in " + path; return; }
    if (skel->Attribute("name")) m.meta.name = skel->Attribute("name");

    for (XMLElement* node = skel->FirstChildElement("Node"); node; node = node->NextSiblingElement("Node")) {
        Node nd;
        nd.id = node->Attribute("name") ? node->Attribute("name") : "";
        nd.parent = node->Attribute("parent") ? node->Attribute("parent") : "";
        if (nd.parent == "None") nd.parent = "";
        if (node->Attribute("endeffector"))
            nd.endeffector = std::string(node->Attribute("endeffector")) == "True";

        XMLElement* body = node->FirstChildElement("Body");
        if (body) {
            nd.body.type = body->Attribute("type") ? body->Attribute("type") : "Box";
            if (body->Attribute("mass")) nd.body.mass = std::stod(body->Attribute("mass"));
            if (body->Attribute("obj")) nd.body.obj = body->Attribute("obj");
            if (body->Attribute("size")) nd.body.size = parseVec3(body->Attribute("size"));
            if (body->Attribute("radius")) nd.body.radius = std::stod(body->Attribute("radius"));
            if (body->Attribute("height")) nd.body.height = std::stod(body->Attribute("height"));
            if (body->Attribute("contact")) nd.body.contact = std::string(body->Attribute("contact")) == "On";
            nd.body.color = parseVec4(body->Attribute("color"));
            nd.body.t = parseTransform(body);
        }
        XMLElement* joint = node->FirstChildElement("Joint");
        if (joint) {
            nd.joint.type = joint->Attribute("type") ? joint->Attribute("type") : "Ball";
            if (joint->Attribute("bvh")) nd.joint.bvh = joint->Attribute("bvh");
            if (joint->Attribute("axis")) nd.joint.axis = parseVec3(joint->Attribute("axis"), {1,0,0});
            if (nd.joint.type == "Revolute") {
                if (joint->Attribute("lower")) nd.joint.lower[0] = std::stod(joint->Attribute("lower"));
                if (joint->Attribute("upper")) nd.joint.upper[0] = std::stod(joint->Attribute("upper"));
                if (joint->Attribute("kp")) { nd.joint.kp[0] = std::stod(joint->Attribute("kp")); nd.joint.hasKp = true; }
                if (joint->Attribute("kv")) { nd.joint.kv[0] = std::stod(joint->Attribute("kv")); nd.joint.hasKv = true; }
            } else {
                if (joint->Attribute("lower")) nd.joint.lower = parseVec3(joint->Attribute("lower"), {-2,-2,-2});
                if (joint->Attribute("upper")) nd.joint.upper = parseVec3(joint->Attribute("upper"), {2,2,2});
                if (joint->Attribute("kp")) { nd.joint.kp = parseVec3(joint->Attribute("kp")); nd.joint.hasKp = true; }
                if (joint->Attribute("kv")) { nd.joint.kv = parseVec3(joint->Attribute("kv")); nd.joint.hasKv = true; }
            }
            nd.joint.t = parseTransform(joint);
        }
        m.skeleton.push_back(std::move(nd));
    }
}

// ---------- muscles ----------
static void loadMuscles(const std::string& path, Model& m, std::string* err) {
    XMLDocument doc;
    if (doc.LoadFile(path.c_str()) != XML_SUCCESS) { if (err) *err = "cannot load muscles " + path; return; }
    XMLElement* root = doc.FirstChildElement("Muscle");
    if (!root) { if (err) *err = "no <Muscle> in " + path; return; }
    for (XMLElement* unit = root->FirstChildElement("Unit"); unit; unit = unit->NextSiblingElement("Unit")) {
        Muscle x;
        x.name = unit->Attribute("name") ? unit->Attribute("name") : "";
        if (unit->Attribute("f0")) x.f0 = std::stod(unit->Attribute("f0"));
        if (unit->Attribute("lm")) x.lm = std::stod(unit->Attribute("lm"));
        if (unit->Attribute("lt")) x.lt = std::stod(unit->Attribute("lt"));
        if (unit->Attribute("pen_angle")) x.pen_angle = std::stod(unit->Attribute("pen_angle"));
        if (unit->Attribute("lmax")) x.lmax = std::stod(unit->Attribute("lmax"));
        if (x.name.rfind("L_", 0) == 0 || x.name.rfind("l_", 0) == 0) x.side = "L";
        else if (x.name.rfind("R_", 0) == 0 || x.name.rfind("r_", 0) == 0) x.side = "R";
        for (XMLElement* wp = unit->FirstChildElement("Waypoint"); wp; wp = wp->NextSiblingElement("Waypoint")) {
            Waypoint w;
            w.body = wp->Attribute("body") ? wp->Attribute("body") : "";
            w.p = parseVec3(wp->Attribute("p"));
            x.waypoints.push_back(w);
        }
        m.muscles.push_back(std::move(x));
    }
}

// ---------- <parameter> block ----------
static void loadParamGroup(XMLElement* group, std::vector<ParamRange>& out) {
    if (!group) return;
    for (XMLElement* c = group->FirstChildElement(); c; c = c->NextSiblingElement()) {
        ParamRange r;
        r.name = c->Name();
        if (c->Attribute("min")) r.min = std::stod(c->Attribute("min"));
        if (c->Attribute("max")) r.max = std::stod(c->Attribute("max"));
        if (c->Attribute("default")) { r.def = std::stod(c->Attribute("default")); r.hasDefault = true; }
        if (c->Attribute("sampling")) { r.sampling = c->Attribute("sampling"); r.hasSampling = true; }
        out.push_back(r);
    }
}

// ---------- env.xml import ----------
std::optional<Model> BootstrapFromLegacy(const std::string& env_path,
                                         const std::string& /*data_root*/,
                                         std::string* err) {
    std::ifstream f(env_path);
    if (!f) { if (err) *err = "cannot open " + env_path; return std::nullopt; }
    std::stringstream ss; ss << f.rdbuf();
    std::string content = ss.str();

    XMLDocument doc;
    if (doc.Parse(content.c_str()) != XML_SUCCESS) { if (err) *err = "malformed env.xml " + env_path; return std::nullopt; }

    Model m;
    EnvConfig& e = m.env;
    fs::path envDir = fs::path(env_path).parent_path();
    auto resolve = [&](const std::string& rel) -> std::string {
        if (rel.empty()) return std::string();
        return (envDir / rel).lexically_normal().string();
    };

    XMLElement* skel = doc.FirstChildElement("skeleton");
    if (skel) {
        if (skel->Attribute("actuactor")) e.actuator = skel->Attribute("actuactor");
        if (skel->Attribute("defaultKp")) e.defaultKp = std::stod(skel->Attribute("defaultKp"));
        if (skel->Attribute("defaultKv")) e.defaultKv = std::stod(skel->Attribute("defaultKv"));
        if (skel->Attribute("damping")) e.damping = std::stod(skel->Attribute("damping"));
        e.skel_file = trim(txt(skel, e.skel_file));
    }
    if (auto g = doc.FirstChildElement("ground")) e.ground_file = trim(txt(g, e.ground_file));
    if (auto b = doc.FirstChildElement("bvh")) {
        e.bvh_file = trim(txt(b, e.bvh_file));
        if (b->Attribute("symmetry")) e.bvh_symmetry = std::string(b->Attribute("symmetry")) == "true";
        if (b->Attribute("heightCalibration")) e.bvh_heightCalibration = std::string(b->Attribute("heightCalibration")) == "true";
    }
    e.cyclicbvh = boolText(doc.FirstChildElement("cyclicbvh"), e.cyclicbvh);
    e.residual = boolText(doc.FirstChildElement("residual"), e.residual);
    e.simHz = intText(doc.FirstChildElement("simHz"), e.simHz);
    e.controlHz = intText(doc.FirstChildElement("controlHz"), e.controlHz);
    if (auto mu = doc.FirstChildElement("muscle")) e.muscle_file = trim(txt(mu, e.muscle_file));
    e.inferencePerSim = intText(doc.FirstChildElement("inferencepersim"), e.inferencePerSim);
    if (auto hc = doc.FirstChildElement("heightCalibration")) {
        e.heightCalibration = boolText(hc, e.heightCalibration);
        if (hc->Attribute("strict")) e.heightCalibrationStrict = std::string(hc->Attribute("strict")) == "true";
    }
    if (auto mp = doc.FirstChildElement("musclePoseOptimization")) {
        e.musclePoseOptimization = boolText(mp, e.musclePoseOptimization);
        if (mp->Attribute("rot")) e.musclePoseOptRot = mp->Attribute("rot");
    }
    e.enforceSymmetry = boolText(doc.FirstChildElement("enforceSymmetry"), e.enforceSymmetry);
    e.actionScale = dblText(doc.FirstChildElement("actionScale"), e.actionScale);
    e.timeWarping = dblText(doc.FirstChildElement("timeWarping"), e.timeWarping);
    e.stanceLearning = boolText(doc.FirstChildElement("stanceLearning"), e.stanceLearning);
    e.metabolicReward = boolText(doc.FirstChildElement("metabolicReward"), e.metabolicReward);
    e.meshLbsWeight = boolText(doc.FirstChildElement("meshLbsWeight"), e.meshLbsWeight);
    e.useVelocityForce = boolText(doc.FirstChildElement("useVelocityForce"), e.useVelocityForce);
    e.useJointState = boolText(doc.FirstChildElement("useJointState"), e.useJointState);
    e.learningStd = boolText(doc.FirstChildElement("learningStd"), e.learningStd);
    e.hardPhaseClipping = boolText(doc.FirstChildElement("hardPhaseClipping"), e.hardPhaseClipping);
    e.softPhaseClipping = boolText(doc.FirstChildElement("softPhaseClipping"), e.softPhaseClipping);
    e.torqueClipping = boolText(doc.FirstChildElement("torqueClipping"), e.torqueClipping);
    e.includeJtPinSPD = boolText(doc.FirstChildElement("includeJtPinSPD"), e.includeJtPinSPD);
    e.useNormalizedParamState = boolText(doc.FirstChildElement("useNormalizedParamState"), e.useNormalizedParamState);
    if (auto eo = doc.FirstChildElement("eoeType")) e.eoeType = trim(txt(eo, e.eoeType));
    if (auto rw = doc.FirstChildElement("rewardType")) e.rewardType = trim(txt(rw, e.rewardType));
    e.headLinearAccWeight = dblText(doc.FirstChildElement("HeadLinearAccWeight"), e.headLinearAccWeight);
    e.headRotWeight = dblText(doc.FirstChildElement("HeadRotWeight"), e.headRotWeight);
    e.stepWeight = dblText(doc.FirstChildElement("StepWeight"), e.stepWeight);
    e.metabolicWeight = dblText(doc.FirstChildElement("MetabolicWeight"), e.metabolicWeight);
    e.avgVelWeight = dblText(doc.FirstChildElement("AvgVelWeight"), e.avgVelWeight);

    if (XMLElement* p = doc.FirstChildElement("parameter")) {
        loadParamGroup(p->FirstChildElement("gait"), m.params.gait);
        loadParamGroup(p->FirstChildElement("skeleton"), m.params.skeleton);
        loadParamGroup(p->FirstChildElement("torsion"), m.params.torsion);
        loadParamGroup(p->FirstChildElement("muscle_length"), m.params.muscle_length);
        loadParamGroup(p->FirstChildElement("muscle_force"), m.params.muscle_force);
    }

    // referenced assets, resolved relative to the env.xml directory
    if (!e.skel_file.empty()) loadSkeleton(resolve(e.skel_file), m, err);
    if (!e.muscle_file.empty()) loadMuscles(resolve(e.muscle_file), m, err);

    // mirror the reference motion into the editor's motion list
    if (!e.bvh_file.empty()) {
        Motion mo;
        std::string b = e.bvh_file;
        size_t slash = b.find_last_of("/\\"); size_t dot = b.find_last_of('.');
        mo.name = b.substr(slash == std::string::npos ? 0 : slash + 1,
                           (dot == std::string::npos ? b.size() : dot) - (slash == std::string::npos ? 0 : slash + 1));
        mo.bvh = e.bvh_file; mo.cyclic = e.cyclicbvh;
        m.motions.push_back(mo);
        m.training.default_motion = mo.name;
    }
    m.training.use_muscle = (e.actuator == "mass" || e.actuator == "muscle");
    m.training.con_hz = e.controlHz;
    m.training.sim_hz = e.simHz;
    return m;
}

// ---------- export ----------
static std::string fmt(double d) { char buf[64]; std::snprintf(buf, sizeof(buf), "%.6g", d); return buf; }
static std::string fmt(const Vec3& v) { return fmt(v[0]) + " " + fmt(v[1]) + " " + fmt(v[2]); }
static std::string fmt(const Vec4& v) { return fmt(v[0]) + " " + fmt(v[1]) + " " + fmt(v[2]) + " " + fmt(v[3]); }
static std::string fmt(const Mat3& m) { std::string s; for (int i=0;i<9;i++){ s+=fmt(m[i]); if(i<8)s+=" "; } return s; }

static bool writeSkeleton(const Model& m, const std::string& path, std::string* err) {
    std::ofstream f(path);
    if (!f) { if (err) *err = "cannot write " + path; return false; }
    f << "<Skeleton name=\"" << m.meta.name << "\">\n";
    for (const auto& n : m.skeleton) {
        f << "    <Node name=\"" << n.id << "\" parent=\"" << (n.parent.empty() ? "None" : n.parent) << "\" ";
        if (n.endeffector) f << "endeffector=\"True\" ";
        f << ">\n";
        f << "        <Body type=\"" << n.body.type << "\" mass=\"" << fmt(n.body.mass) << "\" ";
        if (n.body.type == "Box") f << "size=\"" << fmt(n.body.size) << "\" ";
        if (n.body.type == "Sphere") f << "radius=\"" << fmt(n.body.radius) << "\" ";
        if (n.body.type == "Capsule") f << "radius=\"" << fmt(n.body.radius) << "\" height=\"" << fmt(n.body.height) << "\" ";
        f << "contact=\"" << (n.body.contact ? "On" : "Off") << "\" color=\"" << fmt(n.body.color) << "\" ";
        if (!n.body.obj.empty()) f << "obj=\"" << n.body.obj << "\" ";
        f << ">\n";
        f << "            <Transformation linear=\"" << fmt(n.body.t.linear) << "\" translation=\"" << fmt(n.body.t.translation) << " \"/>\n";
        f << "        </Body>\n";
        f << "        <Joint type=\"" << n.joint.type << "\" ";
        if (!n.joint.bvh.empty()) f << "bvh=\"" << n.joint.bvh << "\" ";
        if (n.joint.type == "Revolute") {
            f << "axis=\"" << fmt(n.joint.axis) << "\" lower=\"" << fmt(n.joint.lower[0]) << "\" upper=\"" << fmt(n.joint.upper[0]) << "\" ";
            if (n.joint.hasKp) f << "kp=\"" << fmt(n.joint.kp[0]) << "\" ";
            if (n.joint.hasKv) f << "kv=\"" << fmt(n.joint.kv[0]) << "\" ";
        } else if (n.joint.type == "Ball") {
            f << "lower=\"" << fmt(n.joint.lower) << "\" upper=\"" << fmt(n.joint.upper) << "\" ";
            if (n.joint.hasKp) f << "kp=\"" << fmt(n.joint.kp) << "\" ";
            if (n.joint.hasKv) f << "kv=\"" << fmt(n.joint.kv) << "\" ";
        }
        f << ">\n";
        f << "            <Transformation linear=\"" << fmt(n.joint.t.linear) << "\" translation=\"" << fmt(n.joint.t.translation) << " \"/>\n";
        f << "        </Joint>\n";
        f << "    </Node>\n";
    }
    f << "</Skeleton>\n";
    return true;
}

static bool writeMuscles(const Model& m, const std::string& path, std::string* err) {
    std::ofstream f(path);
    if (!f) { if (err) *err = "cannot write " + path; return false; }
    f << "<Muscle>\n";
    for (const auto& x : m.muscles) {
        f << "    <Unit name=\"" << x.name << "\" f0=\"" << fmt(x.f0) << "\" lm=\"" << fmt(x.lm)
          << "\" lt=\"" << fmt(x.lt) << "\" pen_angle=\"" << fmt(x.pen_angle) << "\" lmax=\"" << fmt(x.lmax) << "\">\n";
        for (const auto& w : x.waypoints)
            f << "        <Waypoint body=\"" << w.body << "\" p=\"" << fmt(w.p) << " \" />\n";
        f << "    </Unit>\n";
    }
    f << "</Muscle>\n";
    return true;
}

static void writeParamGroup(std::ofstream& f, const char* tag, const std::vector<ParamRange>& v) {
    f << "    <" << tag << ">\n";
    for (const auto& p : v) {
        f << "        <" << p.name << " min=\"" << fmt(p.min) << "\" max=\"" << fmt(p.max) << "\"";
        if (p.hasDefault)  f << " default=\"" << fmt(p.def) << "\"";
        if (p.hasSampling) f << " sampling=\"" << p.sampling << "\"";
        f << "/>\n";
    }
    f << "    </" << tag << ">\n";
}

static bool writeEnv(const Model& m, const std::string& path,
                     const std::string& skelRef, const std::string& groundRef,
                     const std::string& muscleRef, const std::string& bvhRef, std::string* err) {
    std::ofstream f(path);
    if (!f) { if (err) *err = "cannot write " + path; return false; }
    const auto& e = m.env;
    auto b = [](bool v){ return v ? "true" : "false"; };
    f << "<!-- actuactor : 'pd', 'torque', 'muscle', 'mass'  -->\n";
    f << "<skeleton defaultKp=\"" << fmt(e.defaultKp) << "\" defaultKv=\"" << fmt(e.defaultKv)
      << "\" actuactor=\"" << e.actuator << "\" damping=\"" << fmt(e.damping) << "\">" << skelRef << "</skeleton>\n";
    f << "<ground>" << groundRef << "</ground>\n\n";
    f << "<bvh symmetry=\"" << b(e.bvh_symmetry) << "\" heightCalibration=\"" << b(e.bvh_heightCalibration) << "\">" << bvhRef << "</bvh>\n";
    f << "<cyclicbvh>" << b(e.cyclicbvh) << "</cyclicbvh>\n";
    f << "<residual>" << b(e.residual) << "</residual>\n";
    f << "<simHz>" << e.simHz << "</simHz>\n";
    f << "<controlHz>" << e.controlHz << "</controlHz>\n";
    f << "<muscle>" << muscleRef << "</muscle>\n";
    f << "<inferencepersim>" << e.inferencePerSim << "</inferencepersim>\n";
    f << "<heightCalibration strict=\"" << b(e.heightCalibrationStrict) << "\">" << b(e.heightCalibration) << "</heightCalibration>\n";
    f << "<musclePoseOptimization rot=\"" << e.musclePoseOptRot << "\">" << b(e.musclePoseOptimization) << "</musclePoseOptimization>\n";
    f << "<enforceSymmetry>" << b(e.enforceSymmetry) << "</enforceSymmetry>\n";
    f << "<actionScale>" << fmt(e.actionScale) << "</actionScale>\n";
    f << "<timeWarping>" << fmt(e.timeWarping) << "</timeWarping>\n";
    f << "<stanceLearning>" << b(e.stanceLearning) << "</stanceLearning>\n";
    f << "<metabolicReward>" << b(e.metabolicReward) << "</metabolicReward>\n";
    f << "<meshLbsWeight>" << b(e.meshLbsWeight) << "</meshLbsWeight>\n";
    f << "<useVelocityForce>" << b(e.useVelocityForce) << "</useVelocityForce>\n";
    f << "<useJointState>" << b(e.useJointState) << "</useJointState>\n";
    f << "<learningStd>" << b(e.learningStd) << "</learningStd>\n";
    f << "<hardPhaseClipping>" << b(e.hardPhaseClipping) << "</hardPhaseClipping>\n";
    f << "<softPhaseClipping>" << b(e.softPhaseClipping) << "</softPhaseClipping>\n";
    f << "<torqueClipping>" << b(e.torqueClipping) << "</torqueClipping>\n";
    f << "<includeJtPinSPD>" << b(e.includeJtPinSPD) << "</includeJtPinSPD>\n";
    f << "<useNormalizedParamState>" << b(e.useNormalizedParamState) << "</useNormalizedParamState>\n";
    f << "<eoeType>" << e.eoeType << "</eoeType>\n";
    f << "<rewardType>" << e.rewardType << "</rewardType>\n";
    f << "<HeadLinearAccWeight>" << fmt(e.headLinearAccWeight) << "</HeadLinearAccWeight>\n";
    f << "<HeadRotWeight>" << fmt(e.headRotWeight) << "</HeadRotWeight>\n";
    f << "<StepWeight>" << fmt(e.stepWeight) << "</StepWeight>\n";
    f << "<MetabolicWeight>" << fmt(e.metabolicWeight) << "</MetabolicWeight>\n";
    f << "<AvgVelWeight>" << fmt(e.avgVelWeight) << "</AvgVelWeight>\n";
    f << "<parameter>\n";
    writeParamGroup(f, "gait", m.params.gait);
    writeParamGroup(f, "skeleton", m.params.skeleton);
    writeParamGroup(f, "torsion", m.params.torsion);
    writeParamGroup(f, "muscle_length", m.params.muscle_length);
    writeParamGroup(f, "muscle_force", m.params.muscle_force);
    f << "</parameter>\n";
    return true;
}

bool ExportToLegacy(const Model& m, const std::string& out_dir, std::string* err,
                    bool absPaths, const std::string& dataRoot) {
    std::string d = out_dir;
    if (!d.empty() && d.back() != '/' && d.back() != '\\') d += "/";
    std::string skelPath = d + "skeleton_gaitnet_narrow_model.xml";
    std::string musclePath = d + "muscle_gaitnet.xml";
    if (!writeSkeleton(m, skelPath, err)) return false;
    if (!writeMuscles(m, musclePath, err)) return false;

    std::string skelRef, groundRef, muscleRef, bvhRef;
    if (absPaths) {
        auto abs = [](const std::string& p){ return fs::absolute(p).lexically_normal().string(); };
        skelRef = abs(skelPath);
        muscleRef = abs(musclePath);
        std::string root = dataRoot.empty() ? "." : dataRoot;
        // env.xml paths look like "../data/..." (relative to the runtime CWD = <root>/build),
        // so resolve them against <root>/build. Absolute paths are used as-is.
        auto resolveRel = [&](const std::string& rel) -> std::string {
            if (rel.empty()) return rel;
            if (fs::path(rel).is_absolute()) return abs(rel);
            return abs((fs::path(root) / "build" / rel).string());
        };
        groundRef = resolveRel(m.env.ground_file);
        if (!fs::exists(groundRef)) groundRef = abs((fs::path(root) / "data" / "ground.xml").string());
        bvhRef = resolveRel(m.env.bvh_file);
        if (!fs::exists(bvhRef)) bvhRef = m.env.bvh_file;
    } else {
        skelRef = m.env.skel_file; groundRef = m.env.ground_file;
        muscleRef = m.env.muscle_file; bvhRef = m.env.bvh_file;
    }
    return writeEnv(m, d + "env.xml", skelRef, groundRef, muscleRef, bvhRef, err);
}

// ---------- OpenSim .osim muscle atlas import ----------
static const char* childText(XMLElement* e, const char* name) {
    if (!e) return nullptr;
    XMLElement* c = e->FirstChildElement(name);
    return (c && c->GetText()) ? c->GetText() : nullptr;
}
static void collectMuscles(XMLElement* e, std::vector<AtlasEntry>& out) {
    for (XMLElement* c = e->FirstChildElement(); c; c = c->NextSiblingElement()) {
        if (c->FirstChildElement("max_isometric_force")) {
            AtlasEntry a;
            if (c->Attribute("name")) a.name = c->Attribute("name");
            if (auto t = childText(c, "max_isometric_force")) a.f0 = std::stod(t);
            if (auto t = childText(c, "optimal_fiber_length")) a.lm = std::stod(t);
            if (auto t = childText(c, "tendon_slack_length")) a.lt = std::stod(t);
            if (auto t = childText(c, "pennation_angle_at_optimal")) a.pen_angle = std::stod(t);
            XMLElement* gp = c->FirstChildElement("GeometryPath");
            if (gp) {
                XMLElement* pps = gp->FirstChildElement("PathPointSet");
                XMLElement* objs = pps ? pps->FirstChildElement("objects") : nullptr;
                if (objs) {
                    XMLElement* first = objs->FirstChildElement();
                    XMLElement* last = first;
                    for (XMLElement* p = first; p; p = p->NextSiblingElement()) last = p;
                    auto frame = [](XMLElement* pp) -> std::string {
                        if (!pp) return "";
                        const char* t = childText(pp, "socket_parent_frame");
                        if (!t) t = childText(pp, "body");
                        return t ? t : "";
                    };
                    a.origin_body = frame(first);
                    a.insertion_body = frame(last);
                }
            }
            out.push_back(a);
        }
        collectMuscles(c, out);
    }
}

std::vector<AtlasEntry> ImportOsimMuscles(const std::string& osim_path, std::string* err) {
    std::vector<AtlasEntry> out;
    XMLDocument doc;
    if (doc.LoadFile(osim_path.c_str()) != XML_SUCCESS) { if (err) *err = "cannot load " + osim_path; return out; }
    XMLElement* root = doc.RootElement();
    if (!root) { if (err) *err = "empty osim"; return out; }
    collectMuscles(root, out);
    if (out.empty() && err) *err = "no muscles found in .osim";
    return out;
}

} // namespace ed
