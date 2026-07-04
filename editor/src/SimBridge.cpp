#include "SimBridge.h"
#include "Bootstrap.h"

// This project's simulation core (DART-based).
#include "Environment.h"
#include "Character.h"
#include "dart/dart.hpp"

#include <chrono>
#include <cmath>
#include <filesystem>

namespace fs = std::filesystem;

namespace ed {

SimBridge::~SimBridge() { stop(); }

void SimBridge::configure(const std::string& dataRoot, const std::string& tmpDir) {
    mDataRoot = dataRoot; mTmpDir = tmpDir;
}
void SimBridge::setModel(const Model& m) {
    std::lock_guard<std::mutex> lk(mMx);
    mPending = m;
    mRebuild.store(true);
}
void SimBridge::start() {
    if (mRunning.load()) return;
    mStop.store(false);
    mRunning.store(true);
    mThread = std::thread(&SimBridge::threadMain, this);
}
void SimBridge::stop() {
    mStop.store(true);
    if (mThread.joinable()) mThread.join();
    mRunning.store(false);
}
std::map<std::string, Transform> SimBridge::pose() {
    std::lock_guard<std::mutex> lk(mMx); return mPose;
}
std::string SimBridge::status() {
    std::lock_guard<std::mutex> lk(mMx); return mStatus;
}

static Transform toTransform(const Eigen::Isometry3d& T) {
    Transform t;
    const Eigen::Matrix3d& R = T.linear();
    t.linear = { R(0,0),R(0,1),R(0,2), R(1,0),R(1,1),R(1,2), R(2,0),R(2,1),R(2,2) };
    t.translation = { T.translation().x(), T.translation().y(), T.translation().z() };
    return t;
}

void SimBridge::threadMain() {
    Environment* env = nullptr;
    int subSteps = 16;

    auto setStatus = [&](const std::string& s){ std::lock_guard<std::mutex> lk(mMx); mStatus = s; };
    auto publish = [&](){
        if (!env) return;
        auto skel = env->getCharacter(0)->getSkeleton();
        std::map<std::string, Transform> snap;
        for (size_t i = 0; i < skel->getNumBodyNodes(); i++) {
            auto bn = skel->getBodyNode(i);
            snap[bn->getName()] = toTransform(bn->getTransform());
        }
        std::lock_guard<std::mutex> lk(mMx); mPose = std::move(snap);
    };

    auto rebuild = [&]() -> bool {
        Model m;
        { std::lock_guard<std::mutex> lk(mMx); m = mPending; }
        fs::create_directories(mTmpDir);
        std::string err;
        // Export absolute-path env.xml + skeleton + muscle so sim can load from tmp.
        if (!ExportToLegacy(m, mTmpDir, &err, /*absPaths*/ true, mDataRoot)) {
            setStatus("tmp export failed: " + err); return false;
        }
        if (env) { delete env; env = nullptr; }
        try {
            env = new Environment();
            env->initialize((fs::path(mTmpDir) / "env.xml").string());
            env->setIsRender(true);
            env->reset();
            subSteps = std::max(1, env->getSimulationHz() / std::max(1, env->getControlHz()));
            setStatus("simulating");
        } catch (const std::exception& e) {
            setStatus(std::string("build failed: ") + e.what());
            if (env) { delete env; env = nullptr; }
            return false;
        }
        return true;
    };

    auto clockLast = std::chrono::steady_clock::now();

    while (!mStop.load()) {
        if (mRebuild.exchange(false)) { rebuild(); clockLast = std::chrono::steady_clock::now(); }
        if (!env) { std::this_thread::sleep_for(std::chrono::milliseconds(30)); clockLast = std::chrono::steady_clock::now(); continue; }

        auto now = std::chrono::steady_clock::now();
        double realDt = std::chrono::duration<double>(now - clockLast).count();
        clockLast = now;
        if (realDt > 0.1) realDt = 0.1;

        if (mReset.exchange(false)) { try { env->reset(); } catch (...) {} }

        if (!mPaused.load()) {
            try {
                auto character = env->getCharacter(0);
                auto skel = character->getSkeleton();
                if (mMode.load() == Kinematic) {
                    // real-time reference-motion playback (treadmill: root stays at origin)
                    character->setLocalTime(character->getLocalTime() + realDt);
                    env->updateTargetPosAndVel();
                    Eigen::VectorXd p = env->getTargetPositions();
                    if (p.rows() >= 6) { p[3] = 0.0; p[5] = 0.0; }
                    skel->setPositions(p);
                    skel->computeForwardKinematics(true, false, false);
                } else { // Dynamic
                    Eigen::VectorXd a = Eigen::VectorXd::Zero(env->getNumAction());
                    env->setAction(a);
                    env->step(subSteps);
                }
            } catch (...) { setStatus("step failed (NaN?) - paused"); mPaused.store(true); }
        }
        mSimTime.store(env->getCharacter(0)->getLocalTime());
        publish();
        std::this_thread::sleep_for(std::chrono::milliseconds(4));
    }

    if (env) delete env;
    setStatus("stopped");
    mRunning.store(false);
}

} // namespace ed
