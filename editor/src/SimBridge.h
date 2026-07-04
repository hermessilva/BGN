#pragma once
// Live DART simulation running on a worker thread. Reuses the validated MASS
// core (Character/Muscle/BVH/World). The UI thread only reads pose snapshots;
// all DART access stays on the worker.
#include "MassModel.h"
#include <string>
#include <map>
#include <mutex>
#include <thread>
#include <atomic>

namespace ed {

class SimBridge {
public:
    ~SimBridge();

    enum Mode { Kinematic = 0, Dynamic = 1 };

    void configure(const std::string& dataRoot, const std::string& tmpDir);
    void setModel(const Model& m);      // copy + request (re)build
    void start();
    void stop();
    bool running() const { return mRunning.load(); }

    void setMode(int m) { mMode.store(m); }
    int  mode() const { return mMode.load(); }
    void setPaused(bool p) { mPaused.store(p); }
    bool paused() const { return mPaused.load(); }
    void setActivation(float a) { mActivation.store(a); }
    float activation() const { return mActivation.load(); }
    void requestReset() { mReset.store(true); }

    // snapshot of body world transforms, keyed by node id
    std::map<std::string, Transform> pose();
    std::string status();
    double simTime() const { return mSimTime.load(); }

private:
    void threadMain();

    std::string mDataRoot, mTmpDir;
    std::thread mThread;
    std::atomic<bool> mRunning{false}, mStop{false}, mRebuild{false};
    std::atomic<bool> mPaused{false}, mReset{false};
    std::atomic<int>  mMode{Kinematic};
    std::atomic<float> mActivation{0.0f};
    std::atomic<double> mSimTime{0.0};

    std::mutex mMx;
    Model mPending;                                   // guarded
    std::map<std::string, Transform> mPose;           // guarded
    std::string mStatus = "stopped";                   // guarded
};

} // namespace ed
