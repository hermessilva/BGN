#include "Index.h"
#include <algorithm>
#include <cctype>

namespace mass {

// ---- small helpers ----
static std::string trim(const std::string& s) {
    size_t a = s.find_first_not_of(" \t");
    if (a == std::string::npos) return "";
    size_t b = s.find_last_not_of(" \t");
    return s.substr(a, b - a + 1);
}
static std::vector<std::string> splitOn(const std::string& s, const std::string& sep) {
    std::vector<std::string> out; size_t p = 0;
    while (true) {
        size_t q = s.find(sep, p);
        if (q == std::string::npos) { out.push_back(trim(s.substr(p))); break; }
        out.push_back(trim(s.substr(p, q - p)));
        p = q + sep.size();
    }
    return out;
}
static void dedupSort(std::vector<int>& v) {
    std::sort(v.begin(), v.end());
    v.erase(std::unique(v.begin(), v.end()), v.end());
}

// ---- build ----
void Index::build(const Model& m) {
    mModel = &m;
    const int nb = (int)m.skeleton.size();
    const int nm = (int)m.muscles.size();

    // grow + bump generations for renamed/removed slots
    auto ensure = [](std::vector<uint8_t>& gen, std::vector<std::string>& prev,
                     int n, auto nameAt) {
        if ((int)gen.size() < n) { gen.resize(n, 0); prev.resize(n, std::string()); }
        for (int i = 0; i < n; i++) {
            std::string nm = nameAt(i);
            if (prev[i] != nm) { if (!prev[i].empty()) gen[i]++; prev[i] = nm; }
        }
        // slots that disappeared: bump so their old handles go stale
        for (int i = n; i < (int)gen.size(); i++) {
            if (!prev[i].empty()) { gen[i]++; prev[i].clear(); }
        }
    };
    ensure(mBoneGen, mBoneName, nb, [&](int i){ return m.skeleton[i].id; });
    ensure(mMuscleGen, mMuscleName, nm, [&](int i){ return m.muscles[i].name; });

    mBoneSlot.clear(); mMuscleSlot.clear();
    mBoneUid.clear(); mMuscleUid.clear();
    mBoneMuscles.clear(); mBoneWaypoints.clear();
    mParent.assign(nb, -1);
    mMuscleBones.assign(nm, {});

    for (int i = 0; i < nb; i++) {
        mBoneSlot[m.skeleton[i].id] = i;
        if (!m.skeleton[i].uid.empty()) mBoneUid[m.skeleton[i].uid] = i;
    }
    for (int i = 0; i < nm; i++)
        if (!m.muscles[i].uid.empty()) mMuscleUid[m.muscles[i].uid] = i;
    for (int i = 0; i < nb; i++) {
        const std::string& par = m.skeleton[i].parent;
        auto it = mBoneSlot.find(par);
        mParent[i] = (it != mBoneSlot.end()) ? it->second : -1;
    }
    for (int i = 0; i < nm; i++) mMuscleSlot[m.muscles[i].name] = i;

    // reverse indices from waypoints
    for (int mi = 0; mi < nm; mi++) {
        const Muscle& mu = m.muscles[mi];
        std::vector<int> touched;
        for (int wi = 0; wi < (int)mu.waypoints.size(); wi++) {
            const std::string& b = mu.waypoints[wi].body;
            if (b.empty()) continue;
            mBoneMuscles[b].push_back(mi);
            mBoneWaypoints[b].push_back(WpRef{mi, wi});
            auto it = mBoneSlot.find(b);
            if (it != mBoneSlot.end()) touched.push_back(it->second);
        }
        dedupSort(touched);
        mMuscleBones[mi] = std::move(touched);
    }
    for (auto& kv : mBoneMuscles) dedupSort(kv.second);
}

// ---- handles ----
EntityId Index::boneId(int slot) const {
    if (slot < 0 || slot >= (int)mBoneGen.size()) return {};
    return EntityId{(uint16_t)EntType::Bone, (uint32_t)slot, mBoneGen[slot]};
}
EntityId Index::muscleId(int slot) const {
    if (slot < 0 || slot >= (int)mMuscleGen.size()) return {};
    return EntityId{(uint16_t)EntType::Muscle, (uint32_t)slot, mMuscleGen[slot]};
}
bool Index::isValid(EntityId id) const {
    if (id.type == (uint16_t)EntType::Bone)
        return id.slot < mBoneGen.size() && mBoneGen[id.slot] == id.gen && !mBoneName[id.slot].empty();
    if (id.type == (uint16_t)EntType::Muscle)
        return id.slot < mMuscleGen.size() && mMuscleGen[id.slot] == id.gen && !mMuscleName[id.slot].empty();
    return false;
}

// ---- name lookups ----
int Index::boneSlot(const std::string& n) const {
    auto it = mBoneSlot.find(n); return it != mBoneSlot.end() ? it->second : -1;
}
int Index::muscleSlot(const std::string& n) const {
    auto it = mMuscleSlot.find(n); return it != mMuscleSlot.end() ? it->second : -1;
}
EntityId Index::boneByName(const std::string& n) const { return boneId(boneSlot(n)); }
EntityId Index::muscleByName(const std::string& n) const { return muscleId(muscleSlot(n)); }

EntityId Index::boneByUid(const std::string& uid) const {
    auto it = mBoneUid.find(uid); return it != mBoneUid.end() ? boneId(it->second) : EntityId{};
}
EntityId Index::muscleByUid(const std::string& uid) const {
    auto it = mMuscleUid.find(uid); return it != mMuscleUid.end() ? muscleId(it->second) : EntityId{};
}
EntityId Index::resolve(const std::string& key) const {
    EntityId e = boneByUid(key);   if (e.valid()) return e;
    e = muscleByUid(key);          if (e.valid()) return e;
    e = boneByName(key);           if (e.valid()) return e;
    return muscleByName(key);
}

// ---- reverse indices ----
const std::vector<int>& Index::musclesOfBone(const std::string& bone) const {
    static const std::vector<int> empty;
    auto it = mBoneMuscles.find(bone); return it != mBoneMuscles.end() ? it->second : empty;
}
const std::vector<WpRef>& Index::waypointsOfBone(const std::string& bone) const {
    static const std::vector<WpRef> empty;
    auto it = mBoneWaypoints.find(bone); return it != mBoneWaypoints.end() ? it->second : empty;
}
std::vector<int> Index::musclesCrossingJoint(const std::string& childBone) const {
    std::vector<int> out;
    int c = boneSlot(childBone);
    if (c < 0) return out;
    std::vector<int> sub = subtreeBones(childBone);
    std::vector<int> anc = ancestorBones(childBone);
    std::sort(sub.begin(), sub.end()); std::sort(anc.begin(), anc.end());
    auto hits = [](const std::vector<int>& touched, const std::vector<int>& set) {
        for (int t : touched) if (std::binary_search(set.begin(), set.end(), t)) return true;
        return false;
    };
    for (int mi = 0; mi < (int)mMuscleBones.size(); mi++) {
        const auto& t = mMuscleBones[mi];
        if (hits(t, sub) && hits(t, anc)) out.push_back(mi);
    }
    return out;
}

// ---- topology ----
int Index::parentOf(int slot) const {
    return (slot >= 0 && slot < (int)mParent.size()) ? mParent[slot] : -1;
}
std::vector<int> Index::subtreeBones(const std::string& bone) const {
    std::vector<int> out;
    int root = boneSlot(bone);
    if (root < 0) return out;
    // children adjacency from parent map
    std::vector<int> stack{root};
    while (!stack.empty()) {
        int cur = stack.back(); stack.pop_back();
        out.push_back(cur);
        for (int i = 0; i < (int)mParent.size(); i++) if (mParent[i] == cur) stack.push_back(i);
    }
    return out;
}
std::vector<int> Index::ancestorBones(const std::string& bone) const {
    std::vector<int> out;
    int cur = parentOf(boneSlot(bone));
    while (cur >= 0) { out.push_back(cur); cur = parentOf(cur); }
    return out;
}

// ---- groups + selector ----
void Index::addGroup(const std::string& name, std::vector<EntityId> members) {
    mGroups[name] = Group{name, std::move(members)};
}

std::string Index::sideOfBone(const std::string& name) const {
    if (!name.empty()) {
        char c = name.back();
        if (c == 'R') return "R";
        if (c == 'L') return "L";
    }
    return "";
}
std::string Index::sideOfMuscle(int slot) const {
    if (slot < 0 || slot >= (int)mModel->muscles.size()) return "";
    const Muscle& mu = mModel->muscles[slot];
    if (mu.side == "L" || mu.side == "R") return mu.side;
    if (mu.name.rfind("L_", 0) == 0) return "L";
    if (mu.name.rfind("R_", 0) == 0) return "R";
    return "";
}

std::vector<EntityId> Index::evalPredicate(const std::string& pin) const {
    std::string p = trim(pin);
    std::vector<EntityId> out;
    auto allBones = [&](auto pred) {
        for (int i = 0; i < boneCount(); i++) if (pred(i)) out.push_back(boneId(i));
    };
    auto allMuscles = [&](auto pred) {
        for (int i = 0; i < muscleCount(); i++) if (pred(i)) out.push_back(muscleId(i));
    };

    auto paren = [&](const char* fn) -> std::string {
        std::string f = fn; f += "(";
        if (p.rfind(f, 0) == 0 && p.back() == ')')
            return trim(p.substr(f.size(), p.size() - f.size() - 1));
        return "";
    };

    if (p.rfind("side=", 0) == 0) {
        std::string s = trim(p.substr(5));
        allBones([&](int i){ return sideOfBone(mModel->skeleton[i].id) == s; });
        allMuscles([&](int i){ return sideOfMuscle(i) == s; });
    } else if (p.rfind("type=", 0) == 0) {
        std::string t = trim(p.substr(5));
        if (t == "bone") allBones([](int){ return true; });
        else if (t == "muscle") allMuscles([](int){ return true; });
    } else if (p.rfind("body=", 0) == 0) {
        std::string b = trim(p.substr(5));
        for (int mi : musclesOfBone(b)) out.push_back(muscleId(mi));
    } else if (!paren("subtree").empty()) {
        for (int s : subtreeBones(paren("subtree"))) out.push_back(boneId(s));
    } else if (!paren("crossing").empty()) {
        for (int mi : musclesCrossingJoint(paren("crossing"))) out.push_back(muscleId(mi));
    } else if (p.rfind("group=", 0) == 0) {
        auto it = mGroups.find(trim(p.substr(6)));
        if (it != mGroups.end()) out = it->second.members;
    }
    return out;
}

static std::vector<EntityId> intersect(std::vector<EntityId> a, const std::vector<EntityId>& b) {
    std::vector<uint64_t> keys; keys.reserve(b.size());
    for (auto& e : b) keys.push_back(e.pack());
    std::sort(keys.begin(), keys.end());
    std::vector<EntityId> out;
    for (auto& e : a) if (std::binary_search(keys.begin(), keys.end(), e.pack())) out.push_back(e);
    return out;
}
static void unionInto(std::vector<EntityId>& acc, const std::vector<EntityId>& b) {
    std::vector<uint64_t> keys; keys.reserve(acc.size());
    for (auto& e : acc) keys.push_back(e.pack());
    std::sort(keys.begin(), keys.end());
    for (auto& e : b) if (!std::binary_search(keys.begin(), keys.end(), e.pack())) acc.push_back(e);
}

std::vector<EntityId> Index::select(const std::string& expr) const {
    std::vector<EntityId> result;
    for (const std::string& orTerm : splitOn(expr, " OR ")) {
        std::vector<std::string> ands = splitOn(orTerm, " AND ");
        std::vector<EntityId> acc = evalPredicate(ands[0]);
        for (size_t i = 1; i < ands.size(); i++) acc = intersect(acc, evalPredicate(ands[i]));
        unionInto(result, acc);
    }
    return result;
}

} // namespace mass
