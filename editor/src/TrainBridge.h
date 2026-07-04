#pragma once
// Async telemetry bridge (Boost.Asio TCP, newline-delimited JSON).
// The Python trainer connects and pushes one JSON object per evaluation:
//   {"iteration":N,"reward":r,"loss_actor":..,"loss_critic":..,"loss_muscle":..}
// The editor also can spawn/stop the training process.
#include <string>
#include <vector>
#include <mutex>
#include <thread>
#include <atomic>
#include <memory>

namespace ed {

struct Telemetry {
    int iteration = 0;
    double reward = 0, loss_actor = 0, loss_critic = 0, loss_muscle = 0;
};

class TrainBridge {
public:
    ~TrainBridge();
    void start(unsigned short port = 8765);
    void stop();
    bool running() const { return mRunning.load(); }
    bool clientConnected() const { return mConnected.load(); }
    unsigned short port() const { return mPort; }

    Telemetry latest();
    std::vector<float> rewardHistory();
    std::string status();

    // spawn / kill the training process (Windows)
    void launchTraining(const std::string& cmdline);
    void stopTraining();
    bool trainingRunning() const { return mTrainPid != 0; }

    // called by the network layer
    void onMessage(const std::string& line);
    void setConnected(bool c) { mConnected.store(c); }
    void setStatus(const std::string& s);

private:
    struct Impl;
    std::shared_ptr<Impl> mImpl;
    std::thread mThread;
    std::atomic<bool> mRunning{false}, mConnected{false};
    unsigned short mPort = 8765;

    std::mutex mMx;
    Telemetry mLatest;
    std::vector<float> mRewards;
    std::string mStatus = "stopped";

    unsigned long mTrainPid = 0;
};

} // namespace ed
