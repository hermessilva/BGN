#include "Query.h"

namespace mass {

using json = nlohmann::json;

json Query::describeModel(const Model& m, const Index& ix) {
    json bones = json::array();
    for (const auto& n : m.skeleton) {
        bones.push_back({
            {"uid", n.uid}, {"id", n.id}, {"parent", n.parent},
            {"muscles", (int)ix.musclesOfBone(n.id).size()},
            {"waypoints", (int)ix.waypointsOfBone(n.id).size()}
        });
    }
    json muscles = json::array();
    for (const auto& mu : m.muscles)
        muscles.push_back({ {"uid", mu.uid}, {"name", mu.name},
                            {"side", mu.side}, {"waypoints", (int)mu.waypoints.size()} });

    return {
        {"name", m.meta.name},
        {"counts", { {"bones", (int)m.skeleton.size()},
                     {"muscles", (int)m.muscles.size()},
                     {"motions", (int)m.motions.size()} }},
        {"bones", bones},
        {"muscles", muscles}
    };
}

json Query::getNode(const Model& m, const Index& ix, const std::string& key) {
    EntityId e = ix.resolve(key);
    if (e.type != (uint16_t)EntType::Bone) {
        int s = ix.boneSlot(key);
        if (s < 0) return json();
        e = ix.boneId(s);
    }
    const Node& n = m.skeleton[e.slot];
    json ms = json::array();
    for (int mi : ix.musclesOfBone(n.id)) ms.push_back(m.muscles[mi].name);
    return {
        {"uid", n.uid}, {"id", n.id}, {"parent", n.parent}, {"endeffector", n.endeffector},
        {"body", { {"type", n.body.type}, {"mass", n.body.mass}, {"obj", n.body.obj} }},
        {"joint", { {"type", n.joint.type}, {"bvh", n.joint.bvh} }},
        {"muscles", ms},
        {"waypointCount", (int)ix.waypointsOfBone(n.id).size()}
    };
}

json Query::getMuscle(const Model& m, const Index& ix, const std::string& key) {
    EntityId e = ix.resolve(key);
    int s = -1;
    if (e.type == (uint16_t)EntType::Muscle) s = e.slot;
    else s = ix.muscleSlot(key);
    if (s < 0) return json();
    const Muscle& mu = m.muscles[s];
    json bodies = json::array();
    for (const auto& w : mu.waypoints) bodies.push_back(w.body);
    std::string origin = mu.waypoints.empty() ? "" : mu.waypoints.front().body;
    std::string insertion = mu.waypoints.empty() ? "" : mu.waypoints.back().body;
    return {
        {"uid", mu.uid}, {"name", mu.name},
        {"hill", { {"f0", mu.f0}, {"lm", mu.lm}, {"lt", mu.lt},
                   {"pen_angle", mu.pen_angle}, {"lmax", mu.lmax} }},
        {"anatomy", { {"latin", mu.latin}, {"pt", mu.pt}, {"group", mu.group},
                      {"side", mu.side}, {"antagonist", mu.antagonist}, {"pcsa_cm2", mu.pcsa_cm2} }},
        {"waypointBodies", bodies},
        {"origin", origin}, {"insertion", insertion}
    };
}

json Query::select(const Model& m, const Index& ix, const std::string& expr) {
    json out = json::array();
    for (const EntityId& e : ix.select(expr)) {
        if (e.type == (uint16_t)EntType::Bone && e.slot < m.skeleton.size())
            out.push_back({ {"type", "bone"}, {"uid", m.skeleton[e.slot].uid}, {"name", m.skeleton[e.slot].id} });
        else if (e.type == (uint16_t)EntType::Muscle && e.slot < m.muscles.size())
            out.push_back({ {"type", "muscle"}, {"uid", m.muscles[e.slot].uid}, {"name", m.muscles[e.slot].name} });
    }
    return out;
}

} // namespace mass
