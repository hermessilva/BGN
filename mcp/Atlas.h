#pragma once
// libmassedit — anatomy atlas (OpenSim .osim) as external ground truth.
// Parses muscles (Hill params + origin/insertion bodies), joins to the model by
// normalized name, validates the model against it, and can fill missing model
// metadata/Hill params from it.
#include "MassModel.h"
#include "Index.h"
#include <nlohmann/json.hpp>
#include <string>
#include <unordered_map>
#include <vector>

namespace mass {

struct AtlasMuscle {
    std::string name;
    std::string originBody, insertionBody;
    double f0 = 0, lm = 0, lt = 0, pen = 0;   // max iso force, optimal fiber len, tendon slack, pennation
};

class Atlas {
public:
    // Load an OpenSim .osim (XML). Returns false + err on parse failure.
    bool loadOsim(const std::string& path, std::string* err = nullptr);

    size_t size() const { return mByNorm.size(); }
    // Lookup by model muscle name using normalized matching (case/underscore/side
    // insensitive). Null if absent.
    const AtlasMuscle* find(const std::string& modelMuscleName) const;

    // Normalize a name for joins: lowercase, drop non-alphanumerics, strip a
    // leading/trailing side marker (l/r). Exposed for body-name comparison.
    static std::string normalize(const std::string& s);

    // Validate model muscles against the atlas. Returns [{muscle, issue, ...}]:
    // origin/insertion body mismatch, f0 deviation beyond tolerance.
    static nlohmann::json validate(const Model& m, const Index& ix, const Atlas& atlas,
                                   double f0RelTol = 0.25);

    // Fill model muscles from the atlas: infer empty `side` from the name; if
    // fillHill, copy f0/lm/lt/pen. Returns the number of muscles changed.
    static int sync(Model& m, const Atlas& atlas, bool fillHill, bool dryRun);

private:
    std::vector<AtlasMuscle> mMuscles;
    std::unordered_map<std::string, int> mByNorm;   // normalized name -> index
};

} // namespace mass
