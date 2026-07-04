#include "Complete.h"
#include "Kinematics.h"   // normalize
#include <cstdio>

namespace mass {

using json = nlohmann::json;

static Vec3 add(const Vec3& a, const Vec3& b) { return { a[0]+b[0], a[1]+b[1], a[2]+b[2] }; }
static Vec3 mul(const Vec3& a, double s) { return { a[0]*s, a[1]*s, a[2]*s }; }

std::vector<std::string> Complete::generateFingers(Model& m, const std::string& hand,
                                                   const FingerConfig& cfg) {
    std::vector<std::string> created;
    const Node* h = m.findNode(hand);
    if (!h) return created;

    const Vec3 palm   = h->body.t.translation;      // origin of the fingers
    const Vec3 fwd    = normalize(cfg.forward);
    const Vec3 spread = normalize(cfg.spread);
    // center the fan of fingers on the palm
    double span = cfg.spacing * (cfg.count - 1);

    for (int i = 0; i < cfg.count; i++) {
        double off = -span * 0.5 + cfg.spacing * i;
        Vec3 base = add(palm, mul(spread, off));
        std::string parent = hand;
        for (int jp = 0; jp < cfg.phalanges; jp++) {
            char name[64];
            std::snprintf(name, sizeof(name), "%s_f%dp%d", hand.c_str(), i + 1, jp + 1);
            Node n;
            n.id = name;
            n.parent = parent;
            // proximal joint at base + jp*length along fwd; body center half a length further
            Vec3 jpos = add(base, mul(fwd, cfg.length * jp));
            Vec3 cpos = add(base, mul(fwd, cfg.length * (jp + 0.5)));
            n.joint.type = "Revolute";
            n.joint.axis = spread;                 // flex about the lateral axis
            n.joint.lower[0] = -1.5; n.joint.upper[0] = 0.0;
            n.joint.t.translation = jpos;
            n.body.type = "Box";
            n.body.size = { cfg.length, cfg.thickness, cfg.thickness };
            n.body.t.translation = cpos;
            n.body.mass = cfg.density * cfg.length * cfg.thickness * cfg.thickness;
            m.skeleton.push_back(std::move(n));
            created.push_back(name);
            parent = name;
        }
    }
    return created;
}

std::vector<std::string> Complete::generateFingersSymmetric(Model& m, const std::string& hand,
                                                            const FingerConfig& cfg) {
    std::vector<std::string> created = generateFingers(m, hand, cfg);
    // counterpart hand name (…R<->…L)
    std::string other;
    if (!hand.empty() && hand.back() == 'R') other = hand.substr(0, hand.size()-1) + "L";
    else if (!hand.empty() && hand.back() == 'L') other = hand.substr(0, hand.size()-1) + "R";
    if (!other.empty() && m.findNode(other)) {
        FingerConfig mc = cfg;
        mc.spread = { -cfg.spread[0], cfg.spread[1], cfg.spread[2] };  // mirror across sagittal
        auto more = generateFingers(m, other, mc);
        created.insert(created.end(), more.begin(), more.end());
    }
    return created;
}

json Complete::listGaps(const Model& m, const Index& ix) {
    // small expectation table: these bones should articulate further
    static const std::pair<const char*, const char*> expect[] = {
        {"HandR", "fingers"}, {"HandL", "fingers"},
        {"FootR", "toes"},    {"FootL", "toes"},
    };
    json gaps = json::array();
    for (const auto& e : expect) {
        int s = ix.boneSlot(e.first);
        if (s < 0) continue;                         // bone not in model
        // a gap if no bone lists this one as parent
        bool hasChild = false;
        for (int b = 0; b < ix.boneCount(); b++)
            if (ix.parentOf(b) == s) { hasChild = true; break; }
        if (!hasChild) gaps.push_back({ {"bone", e.first}, {"expects", e.second} });
    }
    return gaps;
}

} // namespace mass
