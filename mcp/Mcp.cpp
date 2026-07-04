#include "Mcp.h"
#include "Query.h"
#include "Batch.h"
#include "Kinematics.h"
#include "Complete.h"
#include "BindSkin.h"

namespace mass {
using json = nlohmann::json;

static Vec3 v3(const json& a, Vec3 def = {0,0,0}) {
    if (!a.is_array() || a.size() < 3) return def;
    return { a[0].get<double>(), a[1].get<double>(), a[2].get<double>() };
}
static json ok(const json& id, const json& result) {
    return { {"jsonrpc","2.0"}, {"id", id}, {"result", result} };
}
static json err(const json& id, int code, const std::string& msg) {
    return { {"jsonrpc","2.0"}, {"id", id}, {"error", { {"code", code}, {"message", msg} }} };
}

std::vector<std::string> McpServer::toolNames() {
    return { "describe_model", "get_node", "get_muscle", "select",
             "muscles_of_body", "muscles_crossing_joint",
             "scale_bone", "translate_subtree", "rotate_joint",
             "generate_fingers", "list_gaps",
             "load_atlas", "validate_anatomy", "sync_from_atlas",
             "bind_skin", "fit_bone", "set_body",
             "save", "load" };
}

json McpServer::callTool(const std::string& name, const json& a, Model& m, Index& ix, bool& mutated) {
    mutated = false;
    // ---- read-only ----
    if (name == "describe_model") return Query::describeModel(m, ix);
    if (name == "get_node")       return Query::getNode(m, ix, a.value("key", ""));
    if (name == "get_muscle")     return Query::getMuscle(m, ix, a.value("key", ""));
    if (name == "select")         return Query::select(m, ix, a.value("expr", ""));
    if (name == "muscles_of_body") {
        json out = json::array();
        for (int mi : ix.musclesOfBone(a.value("bone", ""))) out.push_back(m.muscles[mi].name);
        return out;
    }
    if (name == "muscles_crossing_joint") {
        json out = json::array();
        for (int mi : ix.musclesCrossingJoint(a.value("childBone", ""))) out.push_back(m.muscles[mi].name);
        return out;
    }
    if (name == "list_gaps") return Complete::listGaps(m, ix);

    // ---- mutating ----
    if (name == "scale_bone") {
        std::string bone = a.value("bone", "");
        Vec3 axis = v3(a.value("axis", json::array({0,1,0})));
        double f = a.value("factor", 1.0);
        int n = a.value("symmetric", false)
                    ? Batch::scaleBoneSymmetric(m, ix, bone, axis, f)
                    : (Batch::scaleBone(m, ix, bone, axis, f) ? 1 : 0);
        mutated = n > 0;
        return { {"scaled", n} };
    }
    if (name == "translate_subtree") {
        bool okk = Batch::translateSubtree(m, ix, a.value("bone", ""), v3(a.value("delta", json())));
        mutated = okk; return { {"ok", okk} };
    }
    if (name == "rotate_joint") {
        bool okk = Kinematics::rotateJoint(m, ix, a.value("bone", ""),
                                           v3(a.value("axis", json::array({0,0,1}))),
                                           a.value("angle", 0.0));
        mutated = okk; return { {"ok", okk} };
    }
    if (name == "generate_fingers") {
        FingerConfig cfg;
        cfg.count = a.value("count", 5); cfg.phalanges = a.value("phalanges", 3);
        cfg.length = a.value("length", 0.03); cfg.spacing = a.value("spacing", 0.02);
        std::vector<std::string> created = a.value("symmetric", false)
            ? Complete::generateFingersSymmetric(m, a.value("hand", ""), cfg)
            : Complete::generateFingers(m, a.value("hand", ""), cfg);
        mutated = !created.empty();
        return { {"created", created} };
    }

    // ---- skin ----
    if (name == "bind_skin") {
        std::string e;
        json r = BindSkin::bind(m, a.value("obj", ""),
                                v3(a.value("rot", json::array({0,0,0}))),
                                a.value("scale", 1.0),
                                v3(a.value("offset", json::array({0,0,0}))),
                                a.value("fit", true), &e);
        if (r.is_null()) return { {"ok", false}, {"error", e} };
        mutated = true;   // skin descriptor set + (if fit) skeleton morphed
        return r;
    }
    if (name == "fit_bone") {
        std::string e;
        json r = BindSkin::fitBone(m, a.value("bone", ""), a.value("margin", 1.05), &e);
        if (r.is_null()) return { {"ok", false}, {"error", e} };
        mutated = true;   // bone box resized (+ its mirror)
        return r;
    }
    if (name == "set_body") {
        Node* n = m.findNode(a.value("bone", ""));
        if (!n) return { {"ok", false}, {"error", "no bone"} };
        if (a.contains("size")) n->body.size = v3(a["size"]);
        if (a.contains("mass")) n->body.mass = a["mass"].get<double>();
        mutated = true;
        return { {"ok", true}, {"bone", n->id}, {"size", { n->body.size[0], n->body.size[1], n->body.size[2] }} };
    }

    // ---- atlas ----
    if (name == "load_atlas") {
        std::string e;
        mAtlasLoaded = mAtlas.loadOsim(a.value("path", ""), &e);
        return { {"ok", mAtlasLoaded}, {"muscles", (int)mAtlas.size()}, {"error", e} };
    }
    if (name == "validate_anatomy") {
        if (!mAtlasLoaded) return { {"error", "no atlas loaded"} };
        return Atlas::validate(m, ix, mAtlas);
    }
    if (name == "sync_from_atlas") {
        if (!mAtlasLoaded) return { {"error", "no atlas loaded"} };
        int changed = Atlas::sync(m, mAtlas, a.value("fillHill", false), a.value("dryRun", false));
        mutated = changed > 0 && !a.value("dryRun", false);
        return { {"changed", changed} };
    }

    // ---- io ----
    if (name == "save") {
        std::string e; bool okk = m.SaveMass(a.value("path", ""), &e);
        return { {"ok", okk}, {"error", e} };
    }
    if (name == "load") {
        std::string e; auto nm = Model::LoadMass(a.value("path", ""), &e);
        if (!nm) return { {"ok", false}, {"error", e} };
        m = std::move(*nm); m.assignUids(); mutated = true;
        return { {"ok", true}, {"bones", (int)m.skeleton.size()}, {"muscles", (int)m.muscles.size()} };
    }

    return json();  // unknown tool sentinel
}

json McpServer::handle(const json& req, Model& m, Index& ix) {
    json id = req.contains("id") ? req["id"] : json();
    std::string method = req.value("method", "");

    if (method == "initialize") {
        return ok(id, { {"protocolVersion", "2024-11-05"},
                        {"capabilities", { {"tools", json::object()} }},
                        {"serverInfo", { {"name", "mass-mcp"}, {"version", "0.1"} }} });
    }
    if (method == "tools/list") {
        json tools = json::array();
        for (const auto& n : McpServer::toolNames()) tools.push_back({ {"name", n} });
        return ok(id, { {"tools", tools} });
    }
    if (method == "tools/call") {
        const json& params = req.value("params", json::object());
        std::string name = params.value("name", "");
        json args = params.value("arguments", json::object());
        bool mutated = false;
        json result = callTool(name, args, m, ix, mutated);
        if (result.is_null()) return err(id, -32601, "unknown tool: " + name);
        if (mutated) ix.build(m);   // keep the index consistent after structural/geometry edits
        return ok(id, { {"content", json::array({ { {"type","json"}, {"json", result} } })} });
    }
    return err(id, -32601, "unknown method: " + method);
}

// ---- McpQueue ----
std::future<json> McpQueue::submit(json request) {
    std::lock_guard<std::mutex> lk(mMx);
    mQ.push_back(Item{ std::move(request), {} });
    return mQ.back().resp.get_future();
}
int McpQueue::drain(McpServer& server, Model& m, Index& ix) {
    std::deque<Item> local;
    { std::lock_guard<std::mutex> lk(mMx); local.swap(mQ); }
    int n = 0;
    for (auto& it : local) { it.resp.set_value(server.handle(it.req, m, ix)); ++n; }
    return n;
}
size_t McpQueue::pending() { std::lock_guard<std::mutex> lk(mMx); return mQ.size(); }

} // namespace mass
