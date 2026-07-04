#include "RiggedMesh.h"

#include <assimp/Importer.hpp>
#include <assimp/scene.h>
#include <assimp/postprocess.h>
#include <algorithm>
#include <cstring>

#define STB_IMAGE_IMPLEMENTATION
#include "stb_image.h"

namespace ed {

// aiMatrix4x4 is row-major; our M4 is column-major -> transpose on convert.
static M4 fromAi(const aiMatrix4x4& a) {
    M4 r;
    r.m[0]=a.a1; r.m[1]=a.b1; r.m[2]=a.c1; r.m[3]=a.d1;
    r.m[4]=a.a2; r.m[5]=a.b2; r.m[6]=a.c2; r.m[7]=a.d2;
    r.m[8]=a.a3; r.m[9]=a.b3; r.m[10]=a.c3; r.m[11]=a.d3;
    r.m[12]=a.a4; r.m[13]=a.b4; r.m[14]=a.c4; r.m[15]=a.d4;
    return r;
}
// general 4x4 inverse (column-major)
static M4 invert4(const M4& mm) {
    const float* m = mm.m; float inv[16];
    inv[0]=m[5]*m[10]*m[15]-m[5]*m[11]*m[14]-m[9]*m[6]*m[15]+m[9]*m[7]*m[14]+m[13]*m[6]*m[11]-m[13]*m[7]*m[10];
    inv[4]=-m[4]*m[10]*m[15]+m[4]*m[11]*m[14]+m[8]*m[6]*m[15]-m[8]*m[7]*m[14]-m[12]*m[6]*m[11]+m[12]*m[7]*m[10];
    inv[8]=m[4]*m[9]*m[15]-m[4]*m[11]*m[13]-m[8]*m[5]*m[15]+m[8]*m[7]*m[13]+m[12]*m[5]*m[11]-m[12]*m[7]*m[9];
    inv[12]=-m[4]*m[9]*m[14]+m[4]*m[10]*m[13]+m[8]*m[5]*m[14]-m[8]*m[6]*m[13]-m[12]*m[5]*m[10]+m[12]*m[6]*m[9];
    inv[1]=-m[1]*m[10]*m[15]+m[1]*m[11]*m[14]+m[9]*m[2]*m[15]-m[9]*m[3]*m[14]-m[13]*m[2]*m[11]+m[13]*m[3]*m[10];
    inv[5]=m[0]*m[10]*m[15]-m[0]*m[11]*m[14]-m[8]*m[2]*m[15]+m[8]*m[3]*m[14]+m[12]*m[2]*m[11]-m[12]*m[3]*m[10];
    inv[9]=-m[0]*m[9]*m[15]+m[0]*m[11]*m[13]+m[8]*m[1]*m[15]-m[8]*m[3]*m[13]-m[12]*m[1]*m[11]+m[12]*m[3]*m[9];
    inv[13]=m[0]*m[9]*m[14]-m[0]*m[10]*m[13]-m[8]*m[1]*m[14]+m[8]*m[2]*m[13]+m[12]*m[1]*m[10]-m[12]*m[2]*m[9];
    inv[2]=m[1]*m[6]*m[15]-m[1]*m[7]*m[14]-m[5]*m[2]*m[15]+m[5]*m[3]*m[14]+m[13]*m[2]*m[7]-m[13]*m[3]*m[6];
    inv[6]=-m[0]*m[6]*m[15]+m[0]*m[7]*m[14]+m[4]*m[2]*m[15]-m[4]*m[3]*m[14]-m[12]*m[2]*m[7]+m[12]*m[3]*m[6];
    inv[10]=m[0]*m[5]*m[15]-m[0]*m[7]*m[13]-m[4]*m[1]*m[15]+m[4]*m[3]*m[13]+m[12]*m[1]*m[7]-m[12]*m[3]*m[5];
    inv[14]=-m[0]*m[5]*m[14]+m[0]*m[6]*m[13]+m[4]*m[1]*m[14]-m[4]*m[2]*m[13]-m[12]*m[1]*m[6]+m[12]*m[2]*m[5];
    inv[3]=-m[1]*m[6]*m[11]+m[1]*m[7]*m[10]+m[5]*m[2]*m[11]-m[5]*m[3]*m[10]-m[9]*m[2]*m[7]+m[9]*m[3]*m[6];
    inv[7]=m[0]*m[6]*m[11]-m[0]*m[7]*m[10]-m[4]*m[2]*m[11]+m[4]*m[3]*m[10]+m[8]*m[2]*m[7]-m[8]*m[3]*m[6];
    inv[11]=-m[0]*m[5]*m[11]+m[0]*m[7]*m[9]+m[4]*m[1]*m[11]-m[4]*m[3]*m[9]-m[8]*m[1]*m[7]+m[8]*m[3]*m[5];
    inv[15]=m[0]*m[5]*m[10]-m[0]*m[6]*m[9]-m[4]*m[1]*m[10]+m[4]*m[2]*m[9]+m[8]*m[1]*m[6]-m[8]*m[2]*m[5];
    float det = m[0]*inv[0]+m[1]*inv[4]+m[2]*inv[8]+m[3]*inv[12];
    M4 r; if (std::fabs(det) < 1e-20f) return r; det = 1.0f/det;
    for (int i=0;i<16;i++) r.m[i]=inv[i]*det;
    return r;
}
static M4 transM(const V3& t){ M4 r; r.m[12]=t.x; r.m[13]=t.y; r.m[14]=t.z; return r; }
static M4 scaleM(const V3& s){ M4 r; r.m[0]=s.x; r.m[5]=s.y; r.m[10]=s.z; return r; }
static M4 quatM(float x,float y,float z,float w){ // column-major rotation
    float n=std::sqrt(x*x+y*y+z*z+w*w); if(n>1e-8f){x/=n;y/=n;z/=n;w/=n;}
    M4 r;
    r.m[0]=1-2*(y*y+z*z); r.m[1]=2*(x*y+z*w);   r.m[2]=2*(x*z-y*w);
    r.m[4]=2*(x*y-z*w);   r.m[5]=1-2*(x*x+z*z); r.m[6]=2*(y*z+x*w);
    r.m[8]=2*(x*z+y*w);   r.m[9]=2*(y*z-x*w);   r.m[10]=1-2*(x*x+y*y);
    return r;
}

// ---- build the flattened node tree (parent pushed before children) ----
static void addNode(RiggedMesh& rm, const aiNode* an, int parent) {
    int idx = (int)rm.nodes.size();
    RiggedMesh::Node nd; nd.name = an->mName.C_Str(); nd.local = fromAi(an->mTransformation); nd.parent = parent;
    rm.nodes.push_back(nd);
    rm.nodeByName[nd.name] = idx;
    for (unsigned i = 0; i < an->mNumChildren; i++) addNode(rm, an->mChildren[i], idx);
}

bool RiggedMesh::load(const std::string& path, std::string* err) {
    Assimp::Importer imp;
    // NOTE: no JoinIdenticalVertices (it can collapse UV/weight seams) and FlipUVs
    // for the FBX/GL texture-origin convention.
    const aiScene* sc = imp.ReadFile(path,
        aiProcess_Triangulate | aiProcess_GenSmoothNormals | aiProcess_LimitBoneWeights | aiProcess_FlipUVs);
    if (!sc || !sc->mRootNode || sc->mNumMeshes == 0) { if (err) *err = imp.GetErrorString(); return false; }

    *this = RiggedMesh();
    addNode(*this, sc->mRootNode, -1);
    globalInverse = invert4(fromAi(sc->mRootNode->mTransformation));

    // merge all meshes into one vertex buffer
    for (unsigned mi = 0; mi < sc->mNumMeshes; mi++) {
        const aiMesh* m = sc->mMeshes[mi];
        unsigned base = (unsigned)basePos.size();
        for (unsigned v = 0; v < m->mNumVertices; v++) {
            basePos.push_back(V3{ m->mVertices[v].x, m->mVertices[v].y, m->mVertices[v].z });
            if (m->HasNormals()) baseNrm.push_back(V3{ m->mNormals[v].x, m->mNormals[v].y, m->mNormals[v].z });
            else baseNrm.push_back(V3{ 0,1,0 });
            if (m->HasTextureCoords(0)) { uv.push_back(m->mTextureCoords[0][v].x); uv.push_back(m->mTextureCoords[0][v].y); }
            else { uv.push_back(0); uv.push_back(0); }
            vBone.push_back({ 0,0,0,0 }); vWeight.push_back({ 0,0,0,0 });
        }
        for (unsigned f = 0; f < m->mNumFaces; f++) {
            const aiFace& fc = m->mFaces[f];
            if (fc.mNumIndices != 3) continue;
            for (int k = 0; k < 3; k++) indices.push_back(base + fc.mIndices[k]);
        }
        // bones + weights
        for (unsigned b = 0; b < m->mNumBones; b++) {
            const aiBone* ab = m->mBones[b];
            std::string bn = ab->mName.C_Str();
            int bi;
            auto it = std::find_if(bones.begin(), bones.end(), [&](const Bone& x){ return x.name == bn; });
            if (it == bones.end()) {
                bi = (int)bones.size(); Bone nb; nb.name = bn; nb.offset = fromAi(ab->mOffsetMatrix);
                auto nit = nodeByName.find(bn); nb.node = (nit != nodeByName.end()) ? nit->second : -1;
                bones.push_back(nb);
            } else bi = (int)(it - bones.begin());
            for (unsigned w = 0; w < ab->mNumWeights; w++) {
                unsigned vid = base + ab->mWeights[w].mVertexId; float wt = ab->mWeights[w].mWeight;
                auto& vb = vBone[vid]; auto& vw = vWeight[vid];
                // insert into the smallest weight slot
                int slot = 0; for (int s = 1; s < 4; s++) if (vw[s] < vw[slot]) slot = s;
                if (wt > vw[slot]) { vw[slot] = wt; vb[slot] = bi; }
            }
        }
    }
    // normalize weights
    for (auto& vw : vWeight) { float s = vw[0]+vw[1]+vw[2]+vw[3]; if (s > 1e-6f) for (int k=0;k<4;k++) vw[k]/=s; else vw[0]=1; }

    // animation clip 0
    if (sc->mNumAnimations > 0) {
        const aiAnimation* an = sc->mAnimations[0];
        animDurTicks = an->mDuration;
        animTps = an->mTicksPerSecond > 1e-3 ? an->mTicksPerSecond : 25.0;
        for (unsigned c = 0; c < an->mNumChannels; c++) {
            const aiNodeAnim* na = an->mChannels[c];
            auto nit = nodeByName.find(na->mNodeName.C_Str());
            if (nit == nodeByName.end()) continue;
            Channel ch; ch.node = nit->second;
            for (unsigned k=0;k<na->mNumPositionKeys;k++){ auto& kk=na->mPositionKeys[k]; ch.pos.push_back({kk.mTime,{kk.mValue.x,kk.mValue.y,kk.mValue.z}}); }
            for (unsigned k=0;k<na->mNumScalingKeys;k++){ auto& kk=na->mScalingKeys[k]; ch.scl.push_back({kk.mTime,{kk.mValue.x,kk.mValue.y,kk.mValue.z}}); }
            for (unsigned k=0;k<na->mNumRotationKeys;k++){ auto& kk=na->mRotationKeys[k]; ch.rot.push_back({kk.mTime,kk.mValue.x,kk.mValue.y,kk.mValue.z,kk.mValue.w}); }
            channels.push_back(ch);
        }
    }

    // diffuse texture: prefer the embedded texture, else the material's external file
    const aiTexture* tex = nullptr;
    if (sc->mNumTextures > 0) tex = sc->mTextures[0];
    if (tex) {
        int w,h,comp;
        unsigned char* px = nullptr;
        if (tex->mHeight == 0) // compressed (png/jpg) in pcData, mWidth = byte size
            px = stbi_load_from_memory((const stbi_uc*)tex->pcData, (int)tex->mWidth, &w,&h,&comp,4);
        if (px) { texW=w; texH=h; texRGBA.assign(px, px+(size_t)w*h*4); stbi_image_free(px); }
    }
    if (texRGBA.empty()) {
        aiString tp;
        if (sc->mNumMaterials>0 && sc->mMaterials[0]->GetTexture(aiTextureType_DIFFUSE,0,&tp)==AI_SUCCESS) {
            std::string p = tp.C_Str();
            std::string dir = path; size_t s = dir.find_last_of("/\\"); if (s!=std::string::npos) dir=dir.substr(0,s+1);
            // strip any ".fbm/" folder prefix, try dir + basename
            std::string bn = p; size_t sl = bn.find_last_of("/\\"); if (sl!=std::string::npos) bn=bn.substr(sl+1);
            for (std::string cand : { dir+p, dir+bn, p }) {
                int w,h,comp; unsigned char* px = stbi_load(cand.c_str(), &w,&h,&comp,4);
                if (px) { texW=w; texH=h; texRGBA.assign(px,px+(size_t)w*h*4); stbi_image_free(px); break; }
            }
        }
    }
    return !basePos.empty();
}

static V3 lerpV(const V3&a,const V3&b,float t){ return a + (b-a)*t; }

void RiggedMesh::evaluate(double tSec, std::vector<V3>& outPos, std::vector<V3>& outNrm) const {
    size_t nv = basePos.size();
    outPos.assign(nv, V3{}); outNrm.assign(nv, V3{});
    // channel per node
    std::vector<int> chanOf(nodes.size(), -1);
    for (int c=0;c<(int)channels.size();c++) if (channels[c].node>=0) chanOf[channels[c].node]=c;

    double tick = animDurTicks>0 ? std::fmod(tSec*animTps, animDurTicks) : 0.0;
    auto sampleV=[&](const std::vector<Key>& ks, V3 def)->V3{
        if(ks.empty()) return def; if(ks.size()==1) return ks[0].v;
        for(size_t i=0;i+1<ks.size();i++) if(tick<ks[i+1].t){ float f=(float)((tick-ks[i].t)/std::max(1e-9,ks[i+1].t-ks[i].t)); return lerpV(ks[i].v,ks[i+1].v,f);} return ks.back().v; };
    auto sampleQ=[&](const std::vector<QKey>& ks)->M4{
        if(ks.empty()) return M4(); if(ks.size()==1) return quatM(ks[0].x,ks[0].y,ks[0].z,ks[0].w);
        for(size_t i=0;i+1<ks.size();i++) if(tick<ks[i+1].t){ float f=(float)((tick-ks[i].t)/std::max(1e-9,ks[i+1].t-ks[i].t));
            const QKey&a=ks[i]; QKey b=ks[i+1]; float d=a.x*b.x+a.y*b.y+a.z*b.z+a.w*b.w; if(d<0){b.x=-b.x;b.y=-b.y;b.z=-b.z;b.w=-b.w;}
            return quatM(a.x+(b.x-a.x)*f,a.y+(b.y-a.y)*f,a.z+(b.z-a.z)*f,a.w+(b.w-a.w)*f); }
        return quatM(ks.back().x,ks.back().y,ks.back().z,ks.back().w); };

    std::vector<M4> global(nodes.size());
    for (size_t i=0;i<nodes.size();i++) {
        M4 local = nodes[i].local;
        int c = chanOf[i];
        if (c>=0) {
            M4 T = transM(sampleV(channels[c].pos, V3{0,0,0}));
            M4 R = sampleQ(channels[c].rot);
            M4 S = scaleM(sampleV(channels[c].scl, V3{1,1,1}));
            local = mul(mul(T,R),S);
        }
        global[i] = (nodes[i].parent>=0) ? mul(global[nodes[i].parent], local) : local;
    }
    std::vector<M4> fb(bones.size());
    for (size_t b=0;b<bones.size();b++)
        fb[b] = (bones[b].node>=0) ? mul(mul(globalInverse, global[bones[b].node]), bones[b].offset) : M4();

    for (size_t v=0; v<nv; v++) {
        V3 p{0,0,0}, n{0,0,0};
        for (int j=0;j<4;j++){ float w=vWeight[v][j]; if(w<=0) continue; int b=vBone[v][j]; if(b<0||b>=(int)fb.size()) continue;
            p = p + mulPoint(fb[b], basePos[v])*w;
            n = n + mulDir(fb[b], baseNrm[v])*w; }
        outPos[v]=p; outNrm[v]=normalize(n);
    }
}

} // namespace ed
