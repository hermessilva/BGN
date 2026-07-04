#include "TrainBridge.h"
#include <nlohmann/json.hpp>
#include <boost/asio.hpp>
#ifdef _WIN32
#include <windows.h>
#endif

using boost::asio::ip::tcp;
using json = nlohmann::json;

namespace ed {

struct TrainBridge::Impl {
    boost::asio::io_context io;
    std::unique_ptr<tcp::acceptor> acceptor;
    std::shared_ptr<tcp::socket> socket;
    boost::asio::streambuf buf;
    TrainBridge* owner = nullptr;

    void doAccept() {
        socket = std::make_shared<tcp::socket>(io);
        acceptor->async_accept(*socket, [this](boost::system::error_code ec) {
            if (!ec) {
                owner->setConnected(true);
                owner->setStatus("training client connected");
                doRead();
            } else {
                doAccept();
            }
        });
    }
    void doRead() {
        boost::asio::async_read_until(*socket, buf, '\n',
            [this](boost::system::error_code ec, std::size_t n) {
                if (ec) {
                    owner->setConnected(false);
                    owner->setStatus("client disconnected");
                    boost::system::error_code ig; socket->close(ig);
                    doAccept();
                    return;
                }
                std::istream is(&buf);
                std::string line; std::getline(is, line);
                if (!line.empty()) owner->onMessage(line);
                doRead();
            });
    }
};

TrainBridge::~TrainBridge() { stop(); }

void TrainBridge::start(unsigned short port) {
    if (mRunning.load()) return;
    mPort = port;
    mImpl = std::make_shared<Impl>();
    mImpl->owner = this;
    try {
        mImpl->acceptor = std::make_unique<tcp::acceptor>(
            mImpl->io, tcp::endpoint(boost::asio::ip::make_address("127.0.0.1"), port));
        mImpl->doAccept();
    } catch (const std::exception& e) {
        setStatus(std::string("failed to open port: ") + e.what());
        return;
    }
    mRunning.store(true);
    setStatus("telemetry server on 127.0.0.1:" + std::to_string(port));
    auto impl = mImpl;
    mThread = std::thread([impl]() { impl->io.run(); });
}

void TrainBridge::stop() {
    if (mImpl) {
        boost::asio::post(mImpl->io, [this]() {
            if (mImpl->acceptor) { boost::system::error_code ig; mImpl->acceptor->close(ig); }
            if (mImpl->socket) { boost::system::error_code ig; mImpl->socket->close(ig); }
        });
        mImpl->io.stop();
    }
    if (mThread.joinable()) mThread.join();
    mRunning.store(false);
    mConnected.store(false);
    mImpl.reset();
    stopTraining();
}

void TrainBridge::onMessage(const std::string& line) {
    try {
        json j = json::parse(line);
        Telemetry t;
        t.iteration   = j.value("iteration", 0);
        t.reward      = j.value("reward", 0.0);
        t.loss_actor  = j.value("loss_actor", 0.0);
        t.loss_critic = j.value("loss_critic", 0.0);
        t.loss_muscle = j.value("loss_muscle", 0.0);
        std::lock_guard<std::mutex> lk(mMx);
        mLatest = t;
        mRewards.push_back((float)t.reward);
        if (mRewards.size() > 5000) mRewards.erase(mRewards.begin());
    } catch (...) { /* ignore malformed */ }
}

Telemetry TrainBridge::latest() { std::lock_guard<std::mutex> lk(mMx); return mLatest; }
std::vector<float> TrainBridge::rewardHistory() { std::lock_guard<std::mutex> lk(mMx); return mRewards; }
std::string TrainBridge::status() { std::lock_guard<std::mutex> lk(mMx); return mStatus; }
void TrainBridge::setStatus(const std::string& s) { std::lock_guard<std::mutex> lk(mMx); mStatus = s; }

void TrainBridge::launchTraining(const std::string& cmdline) {
#ifdef _WIN32
    STARTUPINFOA si = { sizeof(si) };
    PROCESS_INFORMATION pi = {};
    std::string cmd = cmdline;
    if (CreateProcessA(nullptr, cmd.data(), nullptr, nullptr, FALSE,
                       CREATE_NEW_CONSOLE, nullptr, nullptr, &si, &pi)) {
        mTrainPid = pi.dwProcessId;
        CloseHandle(pi.hThread);
        CloseHandle(pi.hProcess);
        setStatus("training started (pid " + std::to_string(mTrainPid) + ")");
    } else {
        setStatus("failed to start training");
    }
#endif
}
void TrainBridge::stopTraining() {
#ifdef _WIN32
    if (mTrainPid != 0) {
        // kill the process tree via taskkill
        std::string cmd = "taskkill /PID " + std::to_string(mTrainPid) + " /T /F";
        STARTUPINFOA si = { sizeof(si) }; PROCESS_INFORMATION pi = {};
        if (CreateProcessA(nullptr, cmd.data(), nullptr, nullptr, FALSE,
                           CREATE_NO_WINDOW, nullptr, nullptr, &si, &pi)) {
            CloseHandle(pi.hThread); CloseHandle(pi.hProcess);
        }
        mTrainPid = 0;
    }
#endif
}

} // namespace ed
