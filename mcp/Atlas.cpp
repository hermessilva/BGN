#include "Atlas.h"
#include <tinyxml2.h>
#include <cctype>
#include <cmath>

namespace mass {
using json = nlohmann::json;
namespace xml = tinyxml2;

std::string Atlas::normalize(const std::string& s) {
    // split on non-alphanumerics into lowercase tokens, dropping bare side tokens
    std::vector<std::string> toks;
    std::string cur;
    bool droppedSide = false;
    auto flush = [&]{ if (!cur.empty()) {
        if (cur == "l" || cur == "r") droppedSide = true; else toks.push_back(cur);
        cur.clear();
    } };
    for (char c : s) {
        if (std::isalnum((unsigned char)c)) cur += (char)std::tolower((unsigned char)c);
        else flush();
    }
    flush();
    std::string t;
    for (auto& tk : toks) t += tk;
    // Only when the side wasn't already a separate token (camelCase "FemurR" ->
    // "femurr"), strip a trailing side letter. If a side token was dropped
    // ("femur_r" -> "femur"), leave it — else we'd eat the real trailing letter.
    if (!droppedSide && t.size() > 1 && (t.back() == 'l' || t.back() == 'r')) t.pop_back();
    return t;
}

// last path component of an OpenSim socket path, e.g. "/bodyset/femur_r" -> "femur_r"
static std::string lastPath(const std::string& s) {
    size_t p = s.find_last_of("/");
    return p == std::string::npos ? s : s.substr(p + 1);
}

// resolve a PathPoint's body: <body> (3.x) or <socket_parent_frame> (4.x)
static std::string pointBody(xml::XMLElement* pt) {
    if (auto* b = pt->FirstChildElement("body"))
        if (b->GetText()) return b->GetText();
    if (auto* s = pt->FirstChildElement("socket_parent_frame"))
        if (s->GetText()) return lastPath(s->GetText());
    return "";
}

static double childD(xml::XMLElement* e, const char* tag) {
    auto* c = e->FirstChildElement(tag);
    if (c && c->GetText()) return std::atof(c->GetText());
    return 0.0;
}

// Recursively collect muscle elements (any element with a max_isometric_force child).
static void collect(xml::XMLElement* e, std::vector<AtlasMuscle>& out) {
    for (xml::XMLElement* c = e->FirstChildElement(); c; c = c->NextSiblingElement()) {
        if (c->FirstChildElement("max_isometric_force")) {
            AtlasMuscle a;
            const char* nm = c->Attribute("name");
            a.name = nm ? nm : "";
            a.f0  = childD(c, "max_isometric_force");
            a.lm  = childD(c, "optimal_fiber_length");
            a.lt  = childD(c, "tendon_slack_length");
            a.pen = childD(c, "pennation_angle_at_optimal");
            if (a.pen == 0.0) a.pen = childD(c, "pennation_angle");
            // path points -> origin (first) / insertion (last)
            std::vector<std::string> bodies;
            if (auto* gp = c->FirstChildElement("GeometryPath")) {
                if (auto* pps = gp->FirstChildElement("PathPointSet")) {
                    if (auto* objs = pps->FirstChildElement("objects")) {
                        for (xml::XMLElement* pt = objs->FirstChildElement(); pt; pt = pt->NextSiblingElement()) {
                            std::string b = pointBody(pt);
                            if (!b.empty()) bodies.push_back(b);
                        }
                    }
                }
            }
            if (!bodies.empty()) { a.originBody = bodies.front(); a.insertionBody = bodies.back(); }
            out.push_back(std::move(a));
        }
        collect(c, out);   // recurse (muscles live deep under Model/ForceSet/objects)
    }
}

bool Atlas::loadOsim(const std::string& path, std::string* err) {
    xml::XMLDocument doc;
    if (doc.LoadFile(path.c_str()) != xml::XML_SUCCESS) {
        if (err) *err = doc.ErrorStr() ? doc.ErrorStr() : "xml parse error";
        return false;
    }
    mMuscles.clear(); mByNorm.clear();
    xml::XMLElement* root = doc.RootElement();
    if (root) collect(root, mMuscles);
    for (int i = 0; i < (int)mMuscles.size(); i++)
        mByNorm[normalize(mMuscles[i].name)] = i;
    return true;
}

const AtlasMuscle* Atlas::find(const std::string& modelMuscleName) const {
    auto it = mByNorm.find(normalize(modelMuscleName));
    return it != mByNorm.end() ? &mMuscles[it->second] : nullptr;
}

json Atlas::validate(const Model& m, const Index& ix, const Atlas& atlas, double f0RelTol) {
    (void)ix;
    json findings = json::array();
    for (const auto& mu : m.muscles) {
        const AtlasMuscle* a = atlas.find(mu.name);
        if (!a) { findings.push_back({ {"muscle", mu.name}, {"issue", "not_in_atlas"} }); continue; }
        if (!mu.waypoints.empty()) {
            std::string org = mu.waypoints.front().body;
            std::string ins = mu.waypoints.back().body;
            if (!a->originBody.empty() && normalize(org) != normalize(a->originBody))
                findings.push_back({ {"muscle", mu.name}, {"issue", "origin_mismatch"},
                                     {"model", org}, {"atlas", a->originBody} });
            if (!a->insertionBody.empty() && normalize(ins) != normalize(a->insertionBody))
                findings.push_back({ {"muscle", mu.name}, {"issue", "insertion_mismatch"},
                                     {"model", ins}, {"atlas", a->insertionBody} });
        }
        if (a->f0 > 0 && mu.f0 > 0) {
            double rel = std::fabs(mu.f0 - a->f0) / a->f0;
            if (rel > f0RelTol)
                findings.push_back({ {"muscle", mu.name}, {"issue", "f0_deviation"},
                                     {"model", mu.f0}, {"atlas", a->f0}, {"rel", rel} });
        }
    }
    return findings;
}

int Atlas::sync(Model& m, const Atlas& atlas, bool fillHill, bool dryRun) {
    int changed = 0;
    for (auto& mu : m.muscles) {
        const AtlasMuscle* a = atlas.find(mu.name);
        if (!a) continue;
        bool touched = false;
        if (mu.side.empty()) {
            if (mu.name.rfind("L_", 0) == 0) { if (!dryRun) mu.side = "L"; touched = true; }
            else if (mu.name.rfind("R_", 0) == 0) { if (!dryRun) mu.side = "R"; touched = true; }
        }
        if (fillHill && a->f0 > 0) {
            if (!dryRun) { mu.f0 = a->f0; if (a->lm > 0) mu.lm = a->lm;
                           if (a->lt > 0) mu.lt = a->lt; mu.pen_angle = a->pen; }
            touched = true;
        }
        if (touched) changed++;
    }
    return changed;
}

} // namespace mass
