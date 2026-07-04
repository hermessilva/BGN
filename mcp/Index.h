#pragma once
// libmassedit — relational index over a mass::Model.
// Builds unique-id handles (generational) + reverse indices + a group selector,
// replacing name-string scans. Rebuilt on structural change, queried in O(1)/O(k).
// No DART, no JSON — operates purely on the in-memory Model struct.
#include "MassModel.h"
#include <cstdint>
#include <string>
#include <vector>
#include <unordered_map>

namespace mass {

// Entity kinds addressable by the index.
enum class EntType : uint16_t { None = 0, Bone = 1, Muscle = 2, Joint = 3, Waypoint = 4 };

// Stable, generational handle. `slot` indexes the Model's dense arrays; `gen`
// invalidates handles whose slot was reused/renamed after a rebuild.
struct EntityId {
    uint16_t type = 0;    // EntType
    uint32_t slot = 0;    // 24-bit index into the backing array
    uint8_t  gen  = 0;    // generation tag
    bool valid() const { return type != 0; }
    bool operator==(const EntityId& o) const { return type == o.type && slot == o.slot && gen == o.gen; }
    bool operator!=(const EntityId& o) const { return !(*this == o); }
    uint64_t pack() const { return (uint64_t(type) << 48) | (uint64_t(slot & 0xFFFFFF) << 8) | gen; }
};

// Reference to a single muscle waypoint (which muscle, which point in its chain).
struct WpRef { int muscle = -1; int wp = -1; };

// A named group of entities. Computed groups (side/subtree/crossing) are produced
// on demand by the selector; explicit groups are stored here.
struct Group {
    std::string name;
    std::vector<EntityId> members;
};

class Index {
public:
    // (Re)build every index from the model. Bumps generations for renamed/removed
    // slots so previously handed-out EntityIds go stale.
    void build(const Model& m);

    // ---- handles ----
    EntityId boneId(int slot) const;
    EntityId muscleId(int slot) const;
    bool isValid(EntityId id) const;

    // ---- O(1) name lookups (replaces findNode/findMuscle scans) ----
    int boneSlot(const std::string& name) const;    // -1 if absent
    int muscleSlot(const std::string& name) const;   // -1 if absent
    EntityId boneByName(const std::string& name) const;
    EntityId muscleByName(const std::string& name) const;

    // ---- stable-uid lookups ----
    EntityId boneByUid(const std::string& uid) const;
    EntityId muscleByUid(const std::string& uid) const;
    // Resolve a free-form key: tries bone/muscle uid, then bone/muscle name.
    EntityId resolve(const std::string& key) const;

    // ---- reverse indices ----
    // All distinct muscles anchoring any waypoint to a bone (e.g. Pelvis -> 112).
    const std::vector<int>& musclesOfBone(const std::string& bone) const;
    // Every waypoint anchored to a bone (used to re-anchor on pose/scale).
    const std::vector<WpRef>& waypointsOfBone(const std::string& bone) const;
    // Muscles that span the joint of `childBone` (touch its subtree and an ancestor).
    std::vector<int> musclesCrossingJoint(const std::string& childBone) const;

    // ---- skeleton topology ----
    int parentOf(int boneSlot) const;                 // -1 for root
    std::vector<int> subtreeBones(const std::string& bone) const;  // includes the bone
    std::vector<int> ancestorBones(const std::string& bone) const; // parent chain, excludes self

    // ---- groups + selector ----
    void addGroup(const std::string& name, std::vector<EntityId> members);
    // Evaluate a selector expression -> id set. Grammar (OR of ANDs):
    //   side=R | side=L | type=bone | type=muscle | group=NAME |
    //   body=BONE | subtree(BONE) | crossing(CHILDBONE)
    // Example: "side=R AND group=hip_flexor", "subtree(FemurL) OR crossing(TibiaR)".
    std::vector<EntityId> select(const std::string& expr) const;

    int boneCount() const { return (int)mBoneGen.size(); }
    int muscleCount() const { return (int)mMuscleGen.size(); }

private:
    const Model* mModel = nullptr;

    std::vector<uint8_t> mBoneGen, mMuscleGen;             // per-slot generation
    std::vector<std::string> mBoneName, mMuscleName;        // previous names (staleness)

    std::unordered_map<std::string, int> mBoneSlot, mMuscleSlot;
    std::unordered_map<std::string, int> mBoneUid, mMuscleUid;
    std::unordered_map<std::string, std::vector<int>>   mBoneMuscles;   // bone -> muscle slots
    std::unordered_map<std::string, std::vector<WpRef>> mBoneWaypoints; // bone -> waypoints
    std::vector<int> mParent;                               // bone slot -> parent slot
    std::vector<std::vector<int>> mMuscleBones;             // muscle slot -> touched bone slots
    std::unordered_map<std::string, Group> mGroups;         // explicit groups

    std::string sideOfBone(const std::string& name) const;
    std::string sideOfMuscle(int slot) const;
    std::vector<EntityId> evalPredicate(const std::string& p) const;
};

} // namespace mass
