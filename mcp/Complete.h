#pragma once
// libmassedit — structural completion: generate missing articulations (finger
// phalanges, toes, ...) so parts like the hand can actually articulate, and
// report gaps against a small expectation table. Generated bones get Revolute
// flex joints and mass from volume; the caller rebuilds the Index afterwards.
#include "MassModel.h"
#include "Index.h"
#include <nlohmann/json.hpp>
#include <string>
#include <vector>

namespace mass {

struct FingerConfig {
    int    count      = 5;            // fingers
    int    phalanges  = 3;            // segments per finger
    Vec3   forward    { 1, 0, 0 };    // finger extension direction (world)
    Vec3   spread     { 0, 0, 1 };    // lateral spread + flex axis (world)
    double spacing    = 0.02;         // gap between fingers
    double length     = 0.03;         // phalanx length (along forward)
    double thickness  = 0.01;         // phalanx cross-section
    double density    = 1000.0;       // kg/m^3 for mass = density*volume
};

struct Complete {
    // Generate `count` fingers of `phalanges` segments each under `hand`, as
    // Revolute-flex chains. Names: "<hand>_f<i>p<j>" (1-based). Appends bones to
    // the model and returns the created bone names. The caller must rebuild the
    // Index before querying/animating them.
    static std::vector<std::string> generateFingers(Model& m, const std::string& hand,
                                                     const FingerConfig& cfg = {});

    // Same, applied to `hand` and its L/R counterpart with the spread X-flipped.
    static std::vector<std::string> generateFingersSymmetric(Model& m, const std::string& hand,
                                                             const FingerConfig& cfg = {});

    // Report bones from a small expectation table (hands/feet) that currently
    // have no children -> [{bone, expects}]. Uses the Index for child lookup.
    static nlohmann::json listGaps(const Model& m, const Index& ix);
};

} // namespace mass
