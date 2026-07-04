#pragma once
// libmassedit — read-only query facade the MCP tool layer exposes. Produces
// JSON descriptions of the model using the Index for fast lookups. No mutation.
#include "MassModel.h"
#include "Index.h"
#include <nlohmann/json.hpp>
#include <string>

namespace mass {

struct Query {
    // High-level summary: counts + bone/muscle name lists + a few reverse-index
    // highlights (e.g. how many muscles touch each bone).
    static nlohmann::json describeModel(const Model& m, const Index& ix);

    // Detail of one bone by uid or name (id/parent/uid, body, joint, and the
    // muscles + waypoints anchored to it). Null json if unknown.
    static nlohmann::json getNode(const Model& m, const Index& ix, const std::string& key);

    // Detail of one muscle by uid or name (hill params, anatomy, waypoint bodies,
    // inferred origin/insertion bone). Null json if unknown.
    static nlohmann::json getMuscle(const Model& m, const Index& ix, const std::string& key);

    // Resolve a selector expression to a list of {type,uid,name} entries.
    static nlohmann::json select(const Model& m, const Index& ix, const std::string& expr);
};

} // namespace mass
