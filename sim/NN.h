#ifndef __MS_NN_H__
#define __MS_NN_H__
// Pure-Eigen inference for the Bidirectional GaitNet networks. Replaces the
// embedded-Python (torch) evaluation. Weights come from the `.safetensors`
// files produced once by tools/export_weights.py.
#include <Eigen/Dense>
#include <string>
#include <vector>
#include <map>
#include <random>

namespace nn
{
// --- safetensors reader (float32 only) ---------------------------------------
class SafeTensors
{
public:
    struct Tensor
    {
        std::vector<int64_t> shape;
        std::vector<float> data;
    };
    void load(const std::string &path);
    bool has(const std::string &name) const { return mTensors.count(name) > 0; }
    const Tensor &get(const std::string &name) const;
    Eigen::MatrixXf matrix(const std::string &name) const; // 2D [rows, cols]
    Eigen::VectorXf vector(const std::string &name) const; // 1D
    std::string meta(const std::string &key) const;        // __metadata__ entry

private:
    std::map<std::string, Tensor> mTensors;
    std::map<std::string, std::string> mMeta;
};

// --- fully connected layer: y = W x + b, W is [out, in] ----------------------
struct Linear
{
    Eigen::MatrixXf W;
    Eigen::VectorXf b;
    void load(const SafeTensors &st, const std::string &prefix); // prefix.weight / prefix.bias
    Eigen::VectorXf forward(const Eigen::VectorXf &x) const { return W * x + b; }
    bool loaded() const { return W.size() > 0; }
};

// --- MuscleNN: 4x Linear, LeakyReLU(0.2) between (two-level muscle control) ---
class MuscleNN
{
public:
    void load(const SafeTensors &st, const std::string &prefix = "muscle.");
    bool loaded() const { return mFc[3].loaded(); }
    // Mirrors ray_model.MuscleNN.unnormalized_no_grad_forward (raw fc output).
    // prev / weight null => non-cascaded input [muscle_tau/200, tau/200];
    // otherwise input is [0.5*prev, weight, muscle_tau/200, tau/200].
    Eigen::VectorXf unnormalizedForward(const Eigen::VectorXd &muscle_tau,
                                        const Eigen::VectorXd &tau,
                                        const Eigen::VectorXf *prev,
                                        const float *weight) const;
    // relu(tanh(x)), elementwise (ray_model.MuscleNN.forward_filter)
    static Eigen::VectorXf forwardFilter(const Eigen::VectorXf &x);

private:
    Linear mFc[4];
    float mStdMuscleTau = 200.0f;
    float mStdTau = 200.0f;
};

// --- SimulationNN policy actor: p_fc 4x Linear, ReLU between ------------------
class PolicyNN
{
public:
    void load(const SafeTensors &st, const std::string &prefix = "policy.");
    bool loaded() const { return mPfc[3].loaded(); }
    // get_action == deterministic mean (p.loc) == p_fc(state). No obs filter
    // (the shipped checkpoints carry an empty/identity MeanStdFilter).
    Eigen::VectorXd getAction(const Eigen::VectorXd &state) const;
    // sigmoid(1000 * (unnormalized - beta))  (ray_model.PolicyNN.weight_filter)
    static double weightFilter(double unnormalized, double beta);

private:
    Linear mPfc[4];
};

// --- Forward GaitNet (RefNN): fc 4x Linear, ReLU between ----------------------
class RefNN
{
public:
    void load(const SafeTensors &st, const std::string &prefix = "ref.");
    bool loaded() const { return mFc[3].loaded(); }
    Eigen::VectorXd forward(const Eigen::VectorXd &param) const;

private:
    Linear mFc[4];
};

// --- Backward GaitNet (AdvancedVAE) ------------------------------------------
class GaitVAE
{
public:
    // num_known: measurable conditions; frame_num: motion frames (60); the rest
    // of the dimensions are inferred from the weight shapes.
    void load(const SafeTensors &st, int num_known, int frame_num = 60,
              const std::string &prefix = "gvae.");
    bool loaded() const { return mLoaded; }
    // ray/advanced_vae.render_forward: motion in -> (predicted motion, full condition)
    void renderForward(const Eigen::VectorXf &input,
                       Eigen::VectorXd &motionOut, Eigen::VectorXd &conditionOut) const;

private:
    Eigen::VectorXf encode(const Eigen::VectorXf &x, Eigen::VectorXf &logVarOut) const;
    Eigen::VectorXf preDecoder(const Eigen::VectorXf &z) const;
    Eigen::VectorXf gaitnet(const Eigen::VectorXf &c) const; // decode condition -> motion

    std::vector<Linear> mEncoder;    // LeakyReLU(0.2)
    Linear mFcMu, mFcVar;
    std::vector<Linear> mPreDecoder; // LeakyReLU(0.2), final Sigmoid
    std::vector<Linear> mDecoder;    // ReLU
    int mNumKnown = 0;
    int mFrameNum = 60;
    bool mLoaded = false;
    mutable std::mt19937 mRng{std::random_device{}()};
};

// Load a cascading / policy checkpoint (<path>.safetensors): fills policy + muscle.
// Returns the env metadata (xml) stored in the file. Either out-ptr may be null.
std::string loadPolicyCheckpoint(const std::string &path, PolicyNN *policy, MuscleNN *muscle);
// Read just the metadata (env xml) from a checkpoint's safetensors.
std::string loadMetadata(const std::string &path);
// Load a Forward GaitNet (RefNN) checkpoint; returns its env metadata (xml).
std::string loadRefCheckpoint(const std::string &path, RefNN *ref);
// Load a Backward GaitNet (AdvancedVAE) checkpoint; returns its env metadata (xml).
std::string loadVAECheckpoint(const std::string &path, GaitVAE *vae, int num_known, int frame_num = 60);

} // namespace nn
#endif
