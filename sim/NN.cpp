#include "NN.h"

#include <fstream>
#include <stdexcept>
#include <cmath>
#include <cstring>
#include <nlohmann/json.hpp>

using json = nlohmann::json;

namespace nn
{
// ---------------------------------------------------------------- SafeTensors
void SafeTensors::load(const std::string &path)
{
    std::ifstream f(path, std::ios::binary);
    if (!f)
        throw std::runtime_error("NN: cannot open safetensors '" + path + "'");

    uint64_t headerLen = 0;
    f.read(reinterpret_cast<char *>(&headerLen), 8);
    std::string headerStr(headerLen, '\0');
    f.read(&headerStr[0], headerLen);

    // Everything after the header is the raw tensor data section.
    std::vector<char> blob((std::istreambuf_iterator<char>(f)), std::istreambuf_iterator<char>());

    json header = json::parse(headerStr);
    mTensors.clear();
    mMeta.clear();
    for (auto it = header.begin(); it != header.end(); ++it)
    {
        if (it.key() == "__metadata__")
        {
            for (auto mit = it.value().begin(); mit != it.value().end(); ++mit)
                mMeta[mit.key()] = mit.value().get<std::string>();
            continue;
        }
        const auto &e = it.value();
        if (e.at("dtype").get<std::string>() != "F32")
            throw std::runtime_error("NN: only F32 tensors supported ('" + it.key() + "')");
        Tensor t;
        for (auto &d : e.at("shape"))
            t.shape.push_back(d.get<int64_t>());
        auto offs = e.at("data_offsets");
        size_t begin = offs[0].get<size_t>();
        size_t end = offs[1].get<size_t>();
        size_t n = (end - begin) / sizeof(float);
        t.data.resize(n);
        std::memcpy(t.data.data(), blob.data() + begin, end - begin);
        mTensors.emplace(it.key(), std::move(t));
    }
}

const SafeTensors::Tensor &SafeTensors::get(const std::string &name) const
{
    auto it = mTensors.find(name);
    if (it == mTensors.end())
        throw std::runtime_error("NN: missing tensor '" + name + "'");
    return it->second;
}

Eigen::MatrixXf SafeTensors::matrix(const std::string &name) const
{
    const Tensor &t = get(name);
    if (t.shape.size() != 2)
        throw std::runtime_error("NN: tensor '" + name + "' is not 2D");
    int rows = (int)t.shape[0], cols = (int)t.shape[1];
    // safetensors stores row-major; map accordingly then copy into a plain MatrixXf.
    Eigen::Map<const Eigen::Matrix<float, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>>
        m(t.data.data(), rows, cols);
    return m;
}

Eigen::VectorXf SafeTensors::vector(const std::string &name) const
{
    const Tensor &t = get(name);
    return Eigen::Map<const Eigen::VectorXf>(t.data.data(), (int)t.data.size());
}

std::string SafeTensors::meta(const std::string &key) const
{
    auto it = mMeta.find(key);
    return it == mMeta.end() ? std::string() : it->second;
}

// ---------------------------------------------------------------------- Linear
void Linear::load(const SafeTensors &st, const std::string &prefix)
{
    W = st.matrix(prefix + ".weight");
    b = st.vector(prefix + ".bias");
}

// ------------------------------------------------------------------ activations
static inline void reluInplace(Eigen::VectorXf &x) { x = x.cwiseMax(0.0f); }
static inline void leakyReluInplace(Eigen::VectorXf &x, float s)
{
    for (int i = 0; i < x.size(); ++i)
        if (x[i] < 0.0f)
            x[i] *= s;
}

// --------------------------------------------------------------------- MuscleNN
void MuscleNN::load(const SafeTensors &st, const std::string &prefix)
{
    mFc[0].load(st, prefix + "fc.0");
    mFc[1].load(st, prefix + "fc.2");
    mFc[2].load(st, prefix + "fc.4");
    mFc[3].load(st, prefix + "fc.6");
}

Eigen::VectorXf MuscleNN::unnormalizedForward(const Eigen::VectorXd &muscle_tau,
                                              const Eigen::VectorXd &tau,
                                              const Eigen::VectorXf *prev,
                                              const float *weight) const
{
    Eigen::VectorXf mt = muscle_tau.cast<float>() / mStdMuscleTau;
    Eigen::VectorXf t = tau.cast<float>() / mStdTau;

    Eigen::VectorXf in;
    if (prev == nullptr && weight == nullptr)
    {
        in.resize(mt.size() + t.size());
        in << mt, t;
    }
    else
    {
        float w = weight ? *weight : 1.0f;
        in.resize(prev->size() + 1 + mt.size() + t.size());
        in << 0.5f * (*prev), w, mt, t;
    }

    Eigen::VectorXf h = mFc[0].forward(in);
    leakyReluInplace(h, 0.2f);
    h = mFc[1].forward(h);
    leakyReluInplace(h, 0.2f);
    h = mFc[2].forward(h);
    leakyReluInplace(h, 0.2f);
    h = mFc[3].forward(h);
    return h;
}

Eigen::VectorXf MuscleNN::forwardFilter(const Eigen::VectorXf &x)
{
    Eigen::VectorXf out(x.size());
    for (int i = 0; i < x.size(); ++i)
    {
        float v = std::tanh(x[i]);
        out[i] = v > 0.0f ? v : 0.0f;
    }
    return out;
}

// --------------------------------------------------------------------- PolicyNN
void PolicyNN::load(const SafeTensors &st, const std::string &prefix)
{
    mPfc[0].load(st, prefix + "p_fc.0");
    mPfc[1].load(st, prefix + "p_fc.2");
    mPfc[2].load(st, prefix + "p_fc.4");
    mPfc[3].load(st, prefix + "p_fc.6");
}

Eigen::VectorXd PolicyNN::getAction(const Eigen::VectorXd &state) const
{
    Eigen::VectorXf h = mPfc[0].forward(state.cast<float>());
    reluInplace(h);
    h = mPfc[1].forward(h);
    reluInplace(h);
    h = mPfc[2].forward(h);
    reluInplace(h);
    h = mPfc[3].forward(h);
    return h.cast<double>();
}

double PolicyNN::weightFilter(double unnormalized, double beta)
{
    return 1.0 / (1.0 + std::exp(-1000.0 * (unnormalized - beta)));
}

// ------------------------------------------------------------------------ RefNN
void RefNN::load(const SafeTensors &st, const std::string &prefix)
{
    mFc[0].load(st, prefix + "fc.0");
    mFc[1].load(st, prefix + "fc.2");
    mFc[2].load(st, prefix + "fc.4");
    mFc[3].load(st, prefix + "fc.6");
}

Eigen::VectorXd RefNN::forward(const Eigen::VectorXd &param) const
{
    Eigen::VectorXf h = mFc[0].forward(param.cast<float>());
    reluInplace(h);
    h = mFc[1].forward(h);
    reluInplace(h);
    h = mFc[2].forward(h);
    reluInplace(h);
    h = mFc[3].forward(h);
    return h.cast<double>();
}

// ----------------------------------------------------------------------- GaitVAE
void GaitVAE::load(const SafeTensors &st, int num_known, int frame_num, const std::string &prefix)
{
    mNumKnown = num_known;
    mFrameNum = frame_num;

    mEncoder.clear();
    for (const char *idx : {"0", "2", "4"})
    {
        Linear l;
        l.load(st, prefix + "encoder." + idx);
        mEncoder.push_back(l);
    }
    mFcMu.load(st, prefix + "fc_mu");
    mFcVar.load(st, prefix + "fc_var");

    mPreDecoder.clear();
    for (const char *idx : {"0", "2", "4", "6"})
    {
        Linear l;
        l.load(st, prefix + "pre_decoder." + idx);
        mPreDecoder.push_back(l);
    }
    mDecoder.clear();
    for (const char *idx : {"0", "2", "4", "6"})
    {
        Linear l;
        l.load(st, prefix + "decoder." + idx);
        mDecoder.push_back(l);
    }
    mLoaded = true;
}

Eigen::VectorXf GaitVAE::encode(const Eigen::VectorXf &x, Eigen::VectorXf &logVarOut) const
{
    Eigen::VectorXf h = x;
    for (const auto &l : mEncoder)
    {
        h = l.forward(h);
        leakyReluInplace(h, 0.2f);
    }
    logVarOut = mFcVar.forward(h);
    return mFcMu.forward(h); // mu
}

Eigen::VectorXf GaitVAE::preDecoder(const Eigen::VectorXf &z) const
{
    Eigen::VectorXf h = z;
    for (size_t i = 0; i < mPreDecoder.size(); ++i)
    {
        h = mPreDecoder[i].forward(h);
        if (i + 1 < mPreDecoder.size())
            leakyReluInplace(h, 0.2f);
        else
            for (int j = 0; j < h.size(); ++j) // final Sigmoid
                h[j] = 1.0f / (1.0f + std::exp(-h[j]));
    }
    return h;
}

Eigen::VectorXf GaitVAE::gaitnet(const Eigen::VectorXf &c) const
{
    const float PI = 3.14159265358979323846f;
    int poseDof = (int)mDecoder.back().b.size();
    Eigen::VectorXf motion(mFrameNum * poseDof);
    for (int i = 0; i < mFrameNum; ++i)
    {
        float angle = 4.0f * PI * ((float)i / (float)mFrameNum);
        Eigen::VectorXf in(c.size() + 2);
        in << c, std::sin(angle), std::cos(angle);
        Eigen::VectorXf h = in;
        for (size_t k = 0; k < mDecoder.size(); ++k)
        {
            h = mDecoder[k].forward(h);
            if (k + 1 < mDecoder.size())
                reluInplace(h);
        }
        motion.segment(i * poseDof, poseDof) = h;
    }
    return motion;
}

void GaitVAE::renderForward(const Eigen::VectorXf &input,
                            Eigen::VectorXd &motionOut, Eigen::VectorXd &conditionOut) const
{
    Eigen::VectorXf logVar;
    Eigen::VectorXf mu = encode(input, logVar);

    // reparameterize: z = mu + exp(0.5*logVar) * eps
    std::normal_distribution<float> gauss(0.0f, 1.0f);
    Eigen::VectorXf z(mu.size());
    for (int i = 0; i < mu.size(); ++i)
        z[i] = mu[i] + std::exp(0.5f * logVar[i]) * gauss(mRng);

    Eigen::VectorXf known = input.tail(mNumKnown);

    Eigen::VectorXf zc(z.size() + known.size());
    zc << z, known;
    Eigen::VectorXf c = preDecoder(zc); // (num_paramstate - num_known)

    Eigen::VectorXf cond(known.size() + c.size());
    cond << known, c; // full condition (num_paramstate)

    conditionOut = cond.cast<double>();
    motionOut = gaitnet(cond).cast<double>();
}

// ------------------------------------------------------------------ loaders
std::string loadPolicyCheckpoint(const std::string &path, PolicyNN *policy, MuscleNN *muscle)
{
    SafeTensors st;
    st.load(path + ".safetensors");
    if (policy)
        policy->load(st, "policy.");
    if (muscle && st.has("muscle.fc.0.weight"))
        muscle->load(st, "muscle.");
    return st.meta("env_xml");
}

std::string loadMetadata(const std::string &path)
{
    SafeTensors st;
    st.load(path + ".safetensors");
    return st.meta("env_xml");
}

std::string loadRefCheckpoint(const std::string &path, RefNN *ref)
{
    SafeTensors st;
    st.load(path + ".safetensors");
    if (ref)
        ref->load(st, "ref.");
    return st.meta("env_xml");
}

std::string loadVAECheckpoint(const std::string &path, GaitVAE *vae, int num_known, int frame_num)
{
    SafeTensors st;
    st.load(path + ".safetensors");
    if (vae)
        vae->load(st, num_known, frame_num, "gvae.");
    return st.meta("env_xml");
}

} // namespace nn
