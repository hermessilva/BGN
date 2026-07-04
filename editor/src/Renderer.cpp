#include "Renderer.h"
#include <glad/glad.h>
#include <cstdio>

namespace ed {

// flat shader (lines/points)
static const char* kVS = R"(#version 330 core
layout(location=0) in vec3 aPos;
layout(location=1) in vec4 aCol;
uniform mat4 uVP;
out vec4 vCol;
void main(){ vCol = aCol; gl_Position = uVP * vec4(aPos,1.0); gl_PointSize = 8.0; }
)";
static const char* kFS = R"(#version 330 core
in vec4 vCol; out vec4 FragColor;
void main(){ FragColor = vCol; }
)";

// lit shader (triangles) — supports uModel (GPU mesh transform) + optional uniform color
static const char* kVSL = R"(#version 330 core
layout(location=0) in vec3 aPos;
layout(location=1) in vec3 aNrm;
layout(location=2) in vec4 aCol;
uniform mat4 uVP;
uniform mat4 uModel;
out vec3 vN; out vec4 vCol; out vec3 vP;
void main(){
    vec4 wp = uModel * vec4(aPos,1.0);
    vP = wp.xyz;
    vN = mat3(uModel) * aNrm;
    vCol = aCol;
    gl_Position = uVP * wp;
}
)";
static const char* kFSL = R"(#version 330 core
in vec3 vN; in vec4 vCol; in vec3 vP;
uniform vec3 uEye;
uniform float uAmb;
uniform int uNumLights;
uniform int uType[8];
uniform vec3 uLdir[8];   // direction (dir) or position (point)
uniform vec3 uLcol[8];   // rgb * intensity
uniform int uUseColor;
uniform vec3 uColorU;
out vec4 FragColor;
void main(){
    vec3 N = normalize(vN);
    vec3 V = normalize(uEye - vP);
    vec3 base = (uUseColor == 1) ? uColorU : vCol.rgb;
    vec3 c = base * uAmb;
    for (int i = 0; i < uNumLights && i < 8; i++) {
        vec3 toL; float atten = 1.0;
        if (uType[i] == 0) {
            toL = normalize(uLdir[i]);           // directional: dir points toward the light
        } else {
            vec3 d = uLdir[i] - vP;              // point
            float dist = length(d);
            toL = d / max(dist, 1e-4);
            atten = 1.0 / (1.0 + 0.25 * dist * dist);
        }
        float diff = dot(N, toL) * 0.5 + 0.5;   // half-Lambert: lights wrap around
        diff = diff * diff;
        c += base * uLcol[i] * diff * atten;
    }
    float rim = pow(1.0 - max(dot(N, V), 0.0), 2.0) * 0.12;
    FragColor = vec4(min(c + rim, vec3(1.0)), vCol.a);
}
)";

static unsigned compile(unsigned type, const char* src){
    unsigned s = glCreateShader(type);
    glShaderSource(s,1,&src,nullptr); glCompileShader(s);
    int ok; glGetShaderiv(s,GL_COMPILE_STATUS,&ok);
    if(!ok){ char log[512]; glGetShaderInfoLog(s,512,nullptr,log); std::fprintf(stderr,"shader: %s\n",log); }
    return s;
}
static unsigned link(const char* vs, const char* fs){
    unsigned v=compile(GL_VERTEX_SHADER,vs), f=compile(GL_FRAGMENT_SHADER,fs);
    unsigned p=glCreateProgram(); glAttachShader(p,v); glAttachShader(p,f); glLinkProgram(p);
    glDeleteShader(v); glDeleteShader(f); return p;
}

bool Renderer::init(){
    mProg = link(kVS, kFS);
    mProgLit = link(kVSL, kFSL);
    glGenVertexArrays(1,&mVao); glGenBuffers(1,&mVbo);
    glGenVertexArrays(1,&mVaoT); glGenBuffers(1,&mVboT);
    glEnable(GL_PROGRAM_POINT_SIZE);
    glEnable(GL_DEPTH_TEST);
    glEnable(GL_BLEND); glBlendFunc(GL_SRC_ALPHA,GL_ONE_MINUS_SRC_ALPHA);
    return mProg!=0 && mProgLit!=0;
}
void Renderer::shutdown(){
    if(mVbo) glDeleteBuffers(1,&mVbo);
    if(mVao) glDeleteVertexArrays(1,&mVao);
    if(mVboT) glDeleteBuffers(1,&mVboT);
    if(mVaoT) glDeleteVertexArrays(1,&mVaoT);
    if(mProg) glDeleteProgram(mProg);
    if(mProgLit) glDeleteProgram(mProgLit);
    if(mColorTex) glDeleteTextures(1,&mColorTex);
    if(mDepthRbo) glDeleteRenderbuffers(1,&mDepthRbo);
    if(mFbo) glDeleteFramebuffers(1,&mFbo);
    if(mSkinVbo) glDeleteBuffers(1,&mSkinVbo);
    if(mSkinVao) glDeleteVertexArrays(1,&mSkinVao);
    freeMeshes();
}
void Renderer::beginTarget(int w, int h){
    if (w < 1) w = 1; if (h < 1) h = 1;
    if (mFbo == 0) glGenFramebuffers(1, &mFbo);
    if (w != mFbW || h != mFbH) {
        mFbW = w; mFbH = h;
        if (mColorTex == 0) glGenTextures(1, &mColorTex);
        glBindTexture(GL_TEXTURE_2D, mColorTex);
        glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA8, w, h, 0, GL_RGBA, GL_UNSIGNED_BYTE, nullptr);
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);
        if (mDepthRbo == 0) glGenRenderbuffers(1, &mDepthRbo);
        glBindRenderbuffer(GL_RENDERBUFFER, mDepthRbo);
        glRenderbufferStorage(GL_RENDERBUFFER, GL_DEPTH_COMPONENT24, w, h);
        glBindFramebuffer(GL_FRAMEBUFFER, mFbo);
        glFramebufferTexture2D(GL_FRAMEBUFFER, GL_COLOR_ATTACHMENT0, GL_TEXTURE_2D, mColorTex, 0);
        glFramebufferRenderbuffer(GL_FRAMEBUFFER, GL_DEPTH_ATTACHMENT, GL_RENDERBUFFER, mDepthRbo);
    }
    glBindFramebuffer(GL_FRAMEBUFFER, mFbo);
    glViewport(0, 0, w, h);
    glClearColor(0.10f, 0.11f, 0.13f, 1.0f);
    glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);
}
void Renderer::endTarget(){ glBindFramebuffer(GL_FRAMEBUFFER, 0); }

void Renderer::begin(const M4& vp, const V3& eye){ mVP=vp; mEye=eye; mLines.clear(); mPoints.clear(); mTris.clear(); mMeshCmds.clear(); mSkinCmd.active=false; }

unsigned Renderer::uploadMesh(const std::vector<V3>& pos, const std::vector<V3>& nrm){
    if (pos.empty() || pos.size() != nrm.size()) return 0;
    GpuMesh gm; gm.count = (int)pos.size();
    std::vector<float> data; data.reserve(pos.size()*6);
    for (size_t i=0;i<pos.size();i++){
        data.push_back(pos[i].x); data.push_back(pos[i].y); data.push_back(pos[i].z);
        data.push_back(nrm[i].x); data.push_back(nrm[i].y); data.push_back(nrm[i].z);
    }
    glGenVertexArrays(1,&gm.vao); glGenBuffers(1,&gm.vbo);
    glBindVertexArray(gm.vao); glBindBuffer(GL_ARRAY_BUFFER,gm.vbo);
    glBufferData(GL_ARRAY_BUFFER, data.size()*sizeof(float), data.data(), GL_STATIC_DRAW);
    glVertexAttribPointer(0,3,GL_FLOAT,GL_FALSE,6*sizeof(float),(void*)0); glEnableVertexAttribArray(0);
    glVertexAttribPointer(1,3,GL_FLOAT,GL_FALSE,6*sizeof(float),(void*)(3*sizeof(float))); glEnableVertexAttribArray(1);
    glBindVertexArray(0);
    mGpuMeshes.push_back(gm);
    return (unsigned)mGpuMeshes.size(); // id = index+1
}
void Renderer::drawMeshGPU(unsigned id, const M4& model, const V3& color){
    if (id == 0 || id > mGpuMeshes.size()) return;
    mMeshCmds.push_back({id, model, color});
}
void Renderer::setSkinMesh(const std::vector<V3>& pos, const std::vector<V3>& nrm){
    mSkinCount = 0;
    if (pos.empty() || pos.size() != nrm.size()) return;
    std::vector<float> data; data.reserve(pos.size()*6);
    for (size_t i=0;i<pos.size();i++){
        data.push_back(pos[i].x); data.push_back(pos[i].y); data.push_back(pos[i].z);
        data.push_back(nrm[i].x); data.push_back(nrm[i].y); data.push_back(nrm[i].z);
    }
    if (!mSkinVao) glGenVertexArrays(1,&mSkinVao);
    if (!mSkinVbo) glGenBuffers(1,&mSkinVbo);
    glBindVertexArray(mSkinVao); glBindBuffer(GL_ARRAY_BUFFER,mSkinVbo);
    glBufferData(GL_ARRAY_BUFFER, data.size()*sizeof(float), data.data(), GL_DYNAMIC_DRAW);
    glVertexAttribPointer(0,3,GL_FLOAT,GL_FALSE,6*sizeof(float),(void*)0); glEnableVertexAttribArray(0);
    glVertexAttribPointer(1,3,GL_FLOAT,GL_FALSE,6*sizeof(float),(void*)(3*sizeof(float))); glEnableVertexAttribArray(1);
    glBindVertexArray(0);
    mSkinCount = (int)pos.size();
}
void Renderer::drawSkin(const M4& model, const V3& color){
    mSkinCmd.active = true; mSkinCmd.model = model; mSkinCmd.color = color;
}
void Renderer::freeMeshes(){
    for (auto& g : mGpuMeshes){ if(g.vbo) glDeleteBuffers(1,&g.vbo); if(g.vao) glDeleteVertexArrays(1,&g.vao); }
    mGpuMeshes.clear(); mMeshCmds.clear();
}
void Renderer::setLights(const std::vector<LightGPU>& lights, float ambient){ mLightsGPU=lights; mAmbient=ambient; }
void Renderer::push(std::vector<Vtx>& buf, const V3& p, const V3& c, float a){ buf.push_back({p.x,p.y,p.z, c.x,c.y,c.z, a}); }
void Renderer::tri(const V3& a, const V3& b, const V3& c, const V3& n, const V3& col, float al){
    mTris.push_back({a.x,a.y,a.z, n.x,n.y,n.z, col.x,col.y,col.z,al});
    mTris.push_back({b.x,b.y,b.z, n.x,n.y,n.z, col.x,col.y,col.z,al});
    mTris.push_back({c.x,c.y,c.z, n.x,n.y,n.z, col.x,col.y,col.z,al});
}
void Renderer::line(const V3& a,const V3& b,const V3& c,float al){ push(mLines,a,c,al); push(mLines,b,c,al); }
void Renderer::point(const V3& p,const V3& c,float al){ push(mPoints,p,c,al); }

void Renderer::wireBox(const M4& model, const V3& he, const V3& c, float al){
    V3 v[8]; int k=0;
    for(int sx=-1;sx<=1;sx+=2) for(int sy=-1;sy<=1;sy+=2) for(int sz=-1;sz<=1;sz+=2)
        v[k++]=mulPoint(model,{he.x*sx,he.y*sy,he.z*sz});
    int e[12][2]={{0,1},{2,3},{4,5},{6,7},{0,2},{1,3},{4,6},{5,7},{0,4},{1,5},{2,6},{3,7}};
    for(auto& ed:e) line(v[ed[0]],v[ed[1]],c,al);
}
void Renderer::solidBox(const M4& model, const V3& he, const V3& col, float al){
    // 8 corners
    V3 v[8]; int k=0;
    for(int sx=-1;sx<=1;sx+=2) for(int sy=-1;sy<=1;sy+=2) for(int sz=-1;sz<=1;sz+=2)
        v[k++]=mulPoint(model,{he.x*sx,he.y*sy,he.z*sz});
    // rotation columns for normals
    V3 cx=normalize(V3{model.m[0],model.m[1],model.m[2]});
    V3 cy=normalize(V3{model.m[4],model.m[5],model.m[6]});
    V3 cz=normalize(V3{model.m[8],model.m[9],model.m[10]});
    // corner index bits: bit2=sx, bit1=sy, bit0=sz  (order matches loop above)
    auto Q=[&](int a,int b,int c,int d,const V3& n){ tri(v[a],v[b],v[c],n,col,al); tri(v[a],v[c],v[d],n,col,al); };
    // +x (sx=+1): indices with bit2 set = 4,5,6,7
    Q(4,5,7,6, cx);
    Q(1,0,2,3, cx*-1.0f);   // -x
    Q(2,6,7,3, cy);         // +y (sy=+1): 2,3,6,7
    Q(0,1,5,4, cy*-1.0f);   // -y
    Q(1,3,7,5, cz);         // +z (sz=+1): 1,3,5,7
    Q(0,4,6,2, cz*-1.0f);   // -z
}
void Renderer::solidSphere(const V3& ctr, float r, const V3& col, int seg){
    for(int i=0;i<seg;i++){
        float t0=(float)i/seg*3.14159f, t1=(float)(i+1)/seg*3.14159f;
        for(int j=0;j<seg;j++){
            float p0=(float)j/seg*6.2832f, p1=(float)(j+1)/seg*6.2832f;
            auto S=[&](float t,float p){ return V3{std::sin(t)*std::cos(p), std::cos(t), std::sin(t)*std::sin(p)}; };
            V3 a=S(t0,p0),b=S(t1,p0),c=S(t1,p1),d=S(t0,p1);
            tri(ctr+a*r,ctr+b*r,ctr+c*r,a,col,1); tri(ctr+a*r,ctr+c*r,ctr+d*r,a,col,1);
        }
    }
}
void Renderer::solidEllipsoid(const M4& model, const V3& he, const V3& col, int seg){
    auto P = [&](float t, float p){                 // unit-sphere dir
        return V3{ std::sin(t)*std::cos(p), std::cos(t), std::sin(t)*std::sin(p) };
    };
    for (int i = 0; i < seg; i++) {
        float t0 = (float)i/seg*3.14159265f, t1 = (float)(i+1)/seg*3.14159265f;
        for (int j = 0; j < seg*2; j++) {
            float p0 = (float)j/(seg*2)*6.2831853f, p1 = (float)(j+1)/(seg*2)*6.2831853f;
            V3 d[4] = { P(t0,p0), P(t1,p0), P(t1,p1), P(t0,p1) };
            V3 wp[4], wn[4];
            for (int k = 0; k < 4; k++) {
                V3 local{ d[k].x*he.x, d[k].y*he.y, d[k].z*he.z };
                wp[k] = mulPoint(model, local);
                // ellipsoid normal ~ (dx/hx, dy/hy, dz/hz) rotated to world
                V3 nl{ d[k].x/he.x, d[k].y/he.y, d[k].z/he.z };
                wn[k] = normalize(mulDir(model, nl));
            }
            triSmooth(wp[0],wn[0], wp[1],wn[1], wp[2],wn[2], col);
            triSmooth(wp[0],wn[0], wp[2],wn[2], wp[3],wn[3], col);
        }
    }
}
void Renderer::triWorld(const V3& a, const V3& b, const V3& c, const V3& col, float al){
    V3 n = normalize(cross(b - a, c - a));
    tri(a, b, c, n, col, al);
}
void Renderer::triSmooth(const V3& a, const V3& na, const V3& b, const V3& nb,
                         const V3& c, const V3& nc, const V3& col, float al){
    mTris.push_back({a.x,a.y,a.z, na.x,na.y,na.z, col.x,col.y,col.z,al});
    mTris.push_back({b.x,b.y,b.z, nb.x,nb.y,nb.z, col.x,col.y,col.z,al});
    mTris.push_back({c.x,c.y,c.z, nc.x,nc.y,nc.z, col.x,col.y,col.z,al});
}
void Renderer::axes(const M4& model,float s){
    V3 o=mulPoint(model,{0,0,0});
    line(o, mulPoint(model,{s,0,0}), {0.9f,0.2f,0.2f});
    line(o, mulPoint(model,{0,s,0}), {0.2f,0.9f,0.2f});
    line(o, mulPoint(model,{0,0,s}), {0.3f,0.5f,0.95f});
}
void Renderer::grid(float size,int divs,const V3& c){
    float step=size/divs, h=size*0.5f;
    for(int i=0;i<=divs;i++){ float x=-h+i*step; line({x,0,-h},{x,0,h},c,0.35f); line({-h,0,x},{h,0,x},c,0.35f); }
}
void Renderer::checkerGround(float size,int divs,const V3& a,const V3& b){
    float step=size/divs, h=size*0.5f;
    V3 up{0,1,0};
    for(int i=0;i<divs;i++) for(int j=0;j<divs;j++){
        float x0=-h+i*step, x1=x0+step, z0=-h+j*step, z1=z0+step;
        const V3& col=((i+j)&1)?a:b;
        V3 p0{x0,0,z0},p1{x1,0,z0},p2{x1,0,z1},p3{x0,0,z1};
        tri(p0,p1,p2,up,col,1); tri(p0,p2,p3,up,col,1);
    }
}
void Renderer::draw(const std::vector<Vtx>& buf, unsigned mode){
    if(buf.empty()) return;
    glBindVertexArray(mVao); glBindBuffer(GL_ARRAY_BUFFER,mVbo);
    glBufferData(GL_ARRAY_BUFFER, buf.size()*sizeof(Vtx), buf.data(), GL_DYNAMIC_DRAW);
    glVertexAttribPointer(0,3,GL_FLOAT,GL_FALSE,sizeof(Vtx),(void*)0); glEnableVertexAttribArray(0);
    glVertexAttribPointer(1,4,GL_FLOAT,GL_FALSE,sizeof(Vtx),(void*)(3*sizeof(float))); glEnableVertexAttribArray(1);
    glDrawArrays(mode,0,(int)buf.size());
}
void Renderer::flush(){
    // batched world-space tris (boxes/ground) + GPU meshes share the lit program
    if (!mTris.empty() || !mMeshCmds.empty() || (mSkinCmd.active && mSkinCount>0)) {
        glUseProgram(mProgLit);
        glUniformMatrix4fv(glGetUniformLocation(mProgLit,"uVP"),1,GL_FALSE,mVP.data());
        glUniform3f(glGetUniformLocation(mProgLit,"uEye"),mEye.x,mEye.y,mEye.z);
        int n2 = (int)mLightsGPU.size(); if (n2>8) n2=8;
        glUniform1f(glGetUniformLocation(mProgLit,"uAmb"), mAmbient);
        glUniform1i(glGetUniformLocation(mProgLit,"uNumLights"), n2);
        if (n2>0){
            int types[8]; float dir[24], col[24];
            for(int i=0;i<n2;i++){ types[i]=mLightsGPU[i].type;
                dir[i*3]=mLightsGPU[i].dir.x; dir[i*3+1]=mLightsGPU[i].dir.y; dir[i*3+2]=mLightsGPU[i].dir.z;
                col[i*3]=mLightsGPU[i].color.x; col[i*3+1]=mLightsGPU[i].color.y; col[i*3+2]=mLightsGPU[i].color.z; }
            glUniform1iv(glGetUniformLocation(mProgLit,"uType[0]"), n2, types);
            glUniform3fv(glGetUniformLocation(mProgLit,"uLdir[0]"), n2, dir);
            glUniform3fv(glGetUniformLocation(mProgLit,"uLcol[0]"), n2, col);
        }
        M4 ident;
        // batched tris: model=identity, per-vertex color
        if (!mTris.empty()) {
            glUniformMatrix4fv(glGetUniformLocation(mProgLit,"uModel"),1,GL_FALSE,ident.data());
            glUniform1i(glGetUniformLocation(mProgLit,"uUseColor"),0);
            glBindVertexArray(mVaoT); glBindBuffer(GL_ARRAY_BUFFER,mVboT);
            glBufferData(GL_ARRAY_BUFFER, mTris.size()*sizeof(VtxN), mTris.data(), GL_DYNAMIC_DRAW);
            glVertexAttribPointer(0,3,GL_FLOAT,GL_FALSE,sizeof(VtxN),(void*)0); glEnableVertexAttribArray(0);
            glVertexAttribPointer(1,3,GL_FLOAT,GL_FALSE,sizeof(VtxN),(void*)(3*sizeof(float))); glEnableVertexAttribArray(1);
            glVertexAttribPointer(2,4,GL_FLOAT,GL_FALSE,sizeof(VtxN),(void*)(6*sizeof(float))); glEnableVertexAttribArray(2);
            glDrawArrays(GL_TRIANGLES,0,(int)mTris.size());
        }
        // GPU meshes: per-command model matrix + uniform color
        glUniform1i(glGetUniformLocation(mProgLit,"uUseColor"),1);
        for (auto& cmd : mMeshCmds) {
            const GpuMesh& g = mGpuMeshes[cmd.id-1];
            glUniformMatrix4fv(glGetUniformLocation(mProgLit,"uModel"),1,GL_FALSE,cmd.model.data());
            glUniform3f(glGetUniformLocation(mProgLit,"uColorU"), cmd.color.x, cmd.color.y, cmd.color.z);
            glBindVertexArray(g.vao);
            glDrawArrays(GL_TRIANGLES,0,g.count);
        }
        // skin (dedicated buffer)
        if (mSkinCmd.active && mSkinCount > 0) {
            glUniform1i(glGetUniformLocation(mProgLit,"uUseColor"),1);
            glUniformMatrix4fv(glGetUniformLocation(mProgLit,"uModel"),1,GL_FALSE,mSkinCmd.model.data());
            glUniform3f(glGetUniformLocation(mProgLit,"uColorU"), mSkinCmd.color.x, mSkinCmd.color.y, mSkinCmd.color.z);
            glBindVertexArray(mSkinVao);
            glDrawArrays(GL_TRIANGLES,0,mSkinCount);
        }
    }
    // lines/points (flat) on top (depth-tested, like the reference viewer)
    glUseProgram(mProg);
    glUniformMatrix4fv(glGetUniformLocation(mProg,"uVP"),1,GL_FALSE,mVP.data());
    draw(mLines,GL_LINES);
    draw(mPoints,GL_POINTS);
}

} // namespace ed
