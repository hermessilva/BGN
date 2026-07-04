#pragma once
// Bridge between the BidirectionalGaitNet native format and the unified project.
//  - BootstrapFromLegacy: env.xml (+ referenced skeleton/muscle/bvh) -> Model
//  - ExportToLegacy:      Model -> env.xml + skeleton xml + muscle xml
// (names kept generic so callers/CLI read the same regardless of the concrete
//  legacy format; here "legacy" == this project's env.xml set.)
#include "MassModel.h"
#include <string>

namespace ed {

// Build a Model from an existing env.xml (its referenced skeleton/muscle/bvh are
// resolved relative to the env.xml's own directory). data_root is accepted for
// signature compatibility but path resolution is done from the env.xml location.
std::optional<Model> BootstrapFromLegacy(const std::string& env_path,
                                         const std::string& data_root,
                                         std::string* err = nullptr);

// Decompose a Model into env.xml + skeleton_gaitnet_narrow_model.xml +
// muscle_gaitnet.xml inside out_dir. When absPaths is true the env.xml refers to
// the written files (and ground/bvh under dataRoot) by absolute path — used by the
// live SimBridge so sim::Environment can load them from a temp directory.
bool ExportToLegacy(const Model& m, const std::string& out_dir, std::string* err = nullptr,
                    bool absPaths = false, const std::string& dataRoot = "");

// Import reference muscle Hill parameters + attachments from an OpenSim .osim model.
std::vector<AtlasEntry> ImportOsimMuscles(const std::string& osim_path, std::string* err = nullptr);

} // namespace ed
