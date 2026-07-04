#include "App.h"
#include <glad/glad.h>
#include <GLFW/glfw3.h>
#include <imgui.h>
#include <imgui_internal.h>
#include <ImGuizmo.h>
#include <imgui_impl_glfw.h>
#include <imgui_impl_opengl3.h>
#include <cstring>
#include <cmath>
#include <cctype>

namespace ed {

// small helpers for editing arrays
static bool editVec3(const char* label, Vec3& v, float step = 0.01f) {
    float f[3] = {(float)v[0], (float)v[1], (float)v[2]};
    if (ImGui::DragFloat3(label, f, step)) { v = {f[0], f[1], f[2]}; return true; }
    return false;
}
static bool editText(const char* label, std::string& s) {
    char buf[256]; std::strncpy(buf, s.c_str(), sizeof(buf)-1); buf[sizeof(buf)-1]=0;
    if (ImGui::InputText(label, buf, sizeof(buf))) { s = buf; return true; }
    return false;
}

void App::drawMenuBar() {
    if (ImGui::BeginMenuBar()) {
        if (ImGui::BeginMenu("File")) {
            if (ImGui::MenuItem("New")) newModel();
            if (ImGui::MenuItem("Open .mass...", "Ctrl+O")) openMass();
            if (ImGui::MenuItem("Save", "Ctrl+S")) saveMass(false);
            if (ImGui::MenuItem("Save As...")) saveMass(true);
            ImGui::Separator();
            if (ImGui::MenuItem("Import env.xml (skeleton+muscle+params)")) bootstrap();
            if (ImGui::MenuItem("Export env.xml + skeleton + muscle...")) exportLegacy();
            ImGui::EndMenu();
        }
        if (ImGui::BeginMenu("Edit")) {
            if (ImGui::MenuItem("Undo", "Ctrl+Z")) undo();
            if (ImGui::MenuItem("Redo", "Ctrl+Y")) redo();
            ImGui::EndMenu();
        }
        if (ImGui::BeginMenu("Anatomy")) {
            if (ImGui::MenuItem("Import OpenSim atlas (.osim)...")) importOsim();
            if (ImGui::MenuItem("Import mesh as skin (obj/glb/fbx/stl)...")) importSkinMesh();
            if (ImGui::MenuItem("Mirror muscle L<->R")) mirrorSelectedMuscle();
            ImGui::EndMenu();
        }
        if (ImGui::BeginMenu("View")) {
            ImGui::MenuItem("Floor", nullptr, &mDrawGrid);
            ImGui::MenuItem("Muscles", nullptr, &mShowMuscles);
            ImGui::EndMenu();
        }
        ImGui::EndMenuBar();
    }
}

// ---- vector icons drawn with ImDrawList (no font asset) ----
enum Icon { IC_NEW, IC_OPEN, IC_SAVE, IC_PLAY, IC_STOP, IC_RESET, IC_FILL, IC_EYE, IC_EXPORT };

static void drawIcon(ImDrawList* dl, ImVec2 p, float s, int icon, ImU32 col) {
    ImVec2 c(p.x + s*0.5f, p.y + s*0.5f);
    float t = 1.6f;
    switch (icon) {
        case IC_NEW: {
            dl->AddRect({p.x+s*0.28f,p.y+s*0.18f},{p.x+s*0.72f,p.y+s*0.82f}, col, 2, 0, t);
            dl->AddLine({p.x+s*0.6f,p.y+s*0.18f},{p.x+s*0.6f,p.y+s*0.32f}, col, t);
            dl->AddLine({p.x+s*0.6f,p.y+s*0.32f},{p.x+s*0.72f,p.y+s*0.32f}, col, t);
        } break;
        case IC_OPEN: {
            dl->AddRect({p.x+s*0.22f,p.y+s*0.34f},{p.x+s*0.78f,p.y+s*0.74f}, col, 2, 0, t);
            dl->AddLine({p.x+s*0.22f,p.y+s*0.34f},{p.x+s*0.38f,p.y+s*0.34f}, col, t);
            dl->AddLine({p.x+s*0.38f,p.y+s*0.34f},{p.x+s*0.46f,p.y+s*0.26f}, col, t);
            dl->AddLine({p.x+s*0.46f,p.y+s*0.26f},{p.x+s*0.62f,p.y+s*0.26f}, col, t);
        } break;
        case IC_SAVE: {
            dl->AddRect({p.x+s*0.24f,p.y+s*0.24f},{p.x+s*0.76f,p.y+s*0.76f}, col, 2, 0, t);
            dl->AddRectFilled({p.x+s*0.38f,p.y+s*0.24f},{p.x+s*0.62f,p.y+s*0.4f}, col, 0);
            dl->AddRect({p.x+s*0.36f,p.y+s*0.52f},{p.x+s*0.64f,p.y+s*0.72f}, col, 0, 0, t);
        } break;
        case IC_PLAY:
            dl->AddTriangleFilled({p.x+s*0.36f,p.y+s*0.28f},{p.x+s*0.36f,p.y+s*0.72f},{p.x+s*0.72f,c.y}, col);
            break;
        case IC_STOP:
            dl->AddRectFilled({p.x+s*0.32f,p.y+s*0.32f},{p.x+s*0.68f,p.y+s*0.68f}, col, 1);
            break;
        case IC_RESET: {
            dl->PathArcTo(c, s*0.26f, 0.5f, 6.0f, 20); dl->PathStroke(col, 0, t);
            dl->AddTriangleFilled({c.x+s*0.26f,p.y+s*0.2f},{c.x+s*0.12f,p.y+s*0.28f},{c.x+s*0.3f,p.y+s*0.34f}, col);
        } break;
        case IC_FILL: { // a little body silhouette
            dl->AddCircleFilled({c.x,p.y+s*0.3f}, s*0.12f, col);
            dl->AddRectFilled({c.x-s*0.14f,p.y+s*0.44f},{c.x+s*0.14f,p.y+s*0.78f}, col, 3);
        } break;
        case IC_EYE: {
            dl->AddCircle(c, s*0.26f, col, 20, t);
            dl->AddCircleFilled(c, s*0.09f, col);
        } break;
        case IC_EXPORT: {
            dl->AddLine({c.x,p.y+s*0.24f},{c.x,p.y+s*0.56f}, col, t);
            dl->AddLine({c.x,p.y+s*0.24f},{c.x-s*0.1f,p.y+s*0.36f}, col, t);
            dl->AddLine({c.x,p.y+s*0.24f},{c.x+s*0.1f,p.y+s*0.36f}, col, t);
            dl->AddLine({p.x+s*0.28f,p.y+s*0.7f},{p.x+s*0.72f,p.y+s*0.7f}, col, t);
        } break;
    }
}

static bool iconButton(const char* id, int icon, const char* tip, bool active=false) {
    const float s = 30.0f;
    ImGui::PushID(id);
    ImVec2 p = ImGui::GetCursorScreenPos();
    if (active) ImGui::PushStyleColor(ImGuiCol_Button, ImGui::GetColorU32(ImGuiCol_ButtonActive));
    bool clicked = ImGui::Button("##ic", ImVec2(s, s));
    if (active) ImGui::PopStyleColor();
    ImU32 col = ImGui::GetColorU32(active ? ImGuiCol_Text : ImGuiCol_Text);
    drawIcon(ImGui::GetWindowDrawList(), p, s, icon, col);
    if (ImGui::IsItemHovered()) ImGui::SetTooltip("%s", tip);
    ImGui::PopID();
    return clicked;
}

// horizontal action bar rendered at the top of the dock host (under the menu)
void App::drawTopToolbar() {
    ImGui::PushStyleVar(ImGuiStyleVar_ItemSpacing, ImVec2(4, 4));
    if (iconButton("new", IC_NEW, "New model")) newModel();          ImGui::SameLine();
    if (iconButton("open", IC_OPEN, "Open .mass...")) openMass();    ImGui::SameLine();
    if (iconButton("save", IC_SAVE, "Save (Ctrl+S)")) saveMass(false); ImGui::SameLine();
    ImGui::TextDisabled("|"); ImGui::SameLine();
    if (iconButton("sim", mSimActive ? IC_STOP : IC_PLAY, mSimActive ? "Stop simulation" : "Simulate", mSimActive)) toggleSim();
    ImGui::SameLine();
    if (iconButton("reset", IC_RESET, "Reset view (Ctrl+1)")) resetView(); ImGui::SameLine();
    ImGui::TextDisabled("|"); ImGui::SameLine();
    ImGui::BeginDisabled(mFillRunning);
    if (iconButton("genfill", IC_FILL, "Generate fill")) startFillGeneration(mFillNewName);
    ImGui::EndDisabled(); ImGui::SameLine();
    if (iconButton("showfill", IC_EYE, mShowSkin ? "Hide fill" : "Show fill", mShowSkin)) mShowSkin = !mShowSkin;
    ImGui::SameLine();
    ImGui::TextDisabled("|"); ImGui::SameLine();
    if (iconButton("export", IC_EXPORT, "Export for training")) exportLegacy();
    ImGui::PopStyleVar();
    ImGui::Separator();
}

void App::drawToolbar() {
    ImGui::Begin("Tools");
    ImGui::Text("Gizmo:");
    if (ImGui::RadioButton("Move", mGizmoOp == 7)) mGizmoOp = 7;    // TRANSLATE
    ImGui::SameLine();
    if (ImGui::RadioButton("Rotate", mGizmoOp == 56)) mGizmoOp = 56; // ROTATE
    ImGui::SameLine();
    if (ImGui::RadioButton("Scale", mGizmoOp == 896)) mGizmoOp = 896;  // SCALE
    if (ImGui::RadioButton("World", mGizmoMode == 1)) mGizmoMode = 1;
    ImGui::SameLine();
    if (ImGui::RadioButton("Local", mGizmoMode == 0)) mGizmoMode = 0;
    ImGui::Separator();
    if (ImGui::Button("+ Body")) addBody();
    ImGui::SameLine();
    if (ImGui::Button("Remove")) removeSelected();
    if (ImGui::Button("Duplicate muscle")) duplicateSelectedMuscle();
    ImGui::SameLine();
    if (ImGui::Button("Mirror L<->R")) mirrorSelectedMuscle();

    ImGui::SeparatorText("DART simulation (live)");
    if (ImGui::Button(mSimActive ? "Stop" : "Simulate")) toggleSim();
    if (mSimActive) {
        ImGui::SameLine();
        if (ImGui::Button("Reset")) mSim.requestReset();
        int md = mSim.mode();
        if (ImGui::RadioButton("Kinematic (BVH)", md == SimBridge::Kinematic)) mSim.setMode(SimBridge::Kinematic);
        ImGui::SameLine();
        if (ImGui::RadioButton("Dynamic (physics)", md == SimBridge::Dynamic)) mSim.setMode(SimBridge::Dynamic);
        bool paused = mSim.paused();
        if (ImGui::Checkbox("Pause", &paused)) mSim.setPaused(paused);
        if (ImGui::SliderFloat("Muscle activation", &mActivation, 0.0f, 1.0f)) mSim.setActivation(mActivation);
        ImGui::Checkbox("Live apply edits", &mLiveSimApply);
        if (!mLiveSimApply) { ImGui::SameLine(); if (ImGui::Button("Apply now")) applySimNow(); }
        bool dirty = (mSimSigApplied != mSimSigPending);
        ImGui::Text("t = %.2fs  |  %s%s", mSim.simTime(), mSim.status().c_str(), dirty ? "  (edits pending)" : "");
    }
    ImGui::End();
}

void App::drawTree() {
    ImGui::Begin("Scene");
    if (ImGui::CollapsingHeader("Visibility", ImGuiTreeNodeFlags_DefaultOpen)) {
        struct { const char* label; bool* v; } items[] = {
            {"Bones##vis", &mShowBones}, {"Muscles##vis", &mShowMuscles},
            {"Joints##vis", &mShowJoints}, {"Waypoints##vis", &mShowWaypoints},
            {"Lights##vis", &mShowLightMarkers}, {"Floor##vis", &mDrawGrid},
            {"Meshes##vis", &mShowMesh}, {"Muscle vol##vis", &mMuscleVolume},
            {"Skin##vis", &mShowSkin},
        };
        const int n = (int)(sizeof(items)/sizeof(items[0]));
        ImGuiStyle& st = ImGui::GetStyle();
        float rightX = ImGui::GetWindowPos().x + ImGui::GetWindowContentRegionMax().x;
        for (int i = 0; i < n; i++) {
            ImGui::Checkbox(items[i].label, items[i].v);
            if (i + 1 < n) {
                float nextW = ImGui::GetFrameHeight() + st.ItemInnerSpacing.x
                            + ImGui::CalcTextSize(items[i+1].label, nullptr, true).x;
                float lastX2 = ImGui::GetItemRectMax().x;
                if (lastX2 + st.ItemSpacing.x + nextW < rightX) ImGui::SameLine();
            }
        }
    }
    if (ImGui::CollapsingHeader("Skeleton", ImGuiTreeNodeFlags_DefaultOpen)) {
        for (int i = 0; i < (int)mModel.skeleton.size(); i++) {
            const Node& n = mModel.skeleton[i];
            bool sel = ((mSel.type==SelType::Body||mSel.type==SelType::Joint) && mSel.index==i);
            ImGui::PushID(i);
            if (ImGui::Selectable((n.id + "  [" + n.joint.type + "]").c_str(), sel))
                mSel = {SelType::Body, i, -1};
            if (ImGui::IsItemClicked(1)) mSel = {SelType::Joint, i, -1};
            ImGui::PopID();
        }
    }
    if (ImGui::CollapsingHeader("Muscles")) {
        ImGui::Text("%d muscles", (int)mModel.muscles.size());
        ImGui::BeginChild("mlist", ImVec2(0, 300));
        for (int i = 0; i < (int)mModel.muscles.size(); i++) {
            bool sel = ((mSel.type==SelType::Muscle||mSel.type==SelType::Waypoint) && mSel.index==i);
            ImGui::PushID(1000+i);
            if (ImGui::Selectable(mModel.muscles[i].name.c_str(), sel))
                mSel = {SelType::Muscle, i, -1};
            ImGui::PopID();
        }
        ImGui::EndChild();
    }
    ImGui::End();
}

void App::drawProperties() {
    ImGui::Begin("Properties");
    if (Node* n = selNode()) {
        ImGui::TextColored(ImVec4(1,0.8f,0.2f,1), "Node: %s", n->id.c_str());
        editText("id", n->id);
        editText("parent", n->parent);
        ImGui::Checkbox("end-effector", &n->endeffector);
        ImGui::SeparatorText("Body");
        const char* btypes[] = {"Box","Sphere","Capsule"};
        int bcur = 0; for (int k=0;k<3;k++) if (n->body.type==btypes[k]) bcur=k;
        if (ImGui::Combo("body type", &bcur, btypes, 3)) n->body.type = btypes[bcur];
        float mass = (float)n->body.mass;
        if (ImGui::DragFloat("mass (kg)", &mass, 0.1f, 0.01f, 500.0f)) n->body.mass = mass;
        if (n->body.type == "Box") editVec3("size", n->body.size);
        else { float r=(float)n->body.radius; if(ImGui::DragFloat("radius",&r,0.005f)) n->body.radius=r; }
        ImGui::Checkbox("contact", &n->body.contact);
        float col[4]={(float)n->body.color[0],(float)n->body.color[1],(float)n->body.color[2],(float)n->body.color[3]};
        if (ImGui::ColorEdit4("color", col)) n->body.color={col[0],col[1],col[2],col[3]};
        editText("obj", n->body.obj);
        editVec3("body pos", n->body.t.translation);
        ImGui::SeparatorText("Joint");
        const char* types[] = {"Free","Ball","Revolute","Weld","Planar"};
        int cur = 1;
        for (int k=0;k<5;k++) if (n->joint.type==types[k]) cur=k;
        if (ImGui::Combo("joint type", &cur, types, 5)) n->joint.type = types[cur];
        editText("bvh", n->joint.bvh);
        if (n->joint.type=="Ball") { editVec3("lower", n->joint.lower); editVec3("upper", n->joint.upper); }
        else if (n->joint.type=="Revolute") {
            editVec3("axis", n->joint.axis, 0.001f);
            float lo=(float)n->joint.lower[0], hi=(float)n->joint.upper[0];
            if(ImGui::DragFloat("lower",&lo,0.01f)) n->joint.lower[0]=lo;
            if(ImGui::DragFloat("upper",&hi,0.01f)) n->joint.upper[0]=hi;
        }
        editVec3("joint pos", n->joint.t.translation);
    } else if (Muscle* mu = selMuscle()) {
        ImGui::TextColored(ImVec4(1,1,0.3f,1), "Muscle: %s", mu->name.c_str());
        editText("name", mu->name);
        ImGui::SeparatorText("Hill");
        float f0=(float)mu->f0, lm=(float)mu->lm, lt=(float)mu->lt, pa=(float)mu->pen_angle;
        if(ImGui::DragFloat("f0 (N)",&f0,1.0f,0.0f,5000.0f)) mu->f0=f0;
        if(ImGui::DragFloat("lm",&lm,0.001f)) mu->lm=lm;
        if(ImGui::DragFloat("lt",&lt,0.001f)) mu->lt=lt;
        if(ImGui::DragFloat("pen_angle",&pa,0.001f)) mu->pen_angle=pa;
        float pcsa=(float)mu->pcsa_cm2;
        if(ImGui::DragFloat("PCSA (cm2)",&pcsa,0.1f,0.0f,300.0f)) {
            mu->pcsa_cm2=pcsa; mu->f0 = pcsa * mModel.meta.specific_tension_N_cm2; // f0 = sigma*PCSA
        }
        ImGui::SeparatorText("Anatomy");
        editText("latin", mu->latin);
        editText("pt", mu->pt);
        editText("group", mu->group);
        editText("side (L/R)", mu->side);
        editText("antagonist", mu->antagonist);
        ImGui::SeparatorText("Waypoints");
        for (int k=0;k<(int)mu->waypoints.size();k++) {
            ImGui::PushID(k);
            bool selW=(mSel.type==SelType::Waypoint && mSel.sub==k);
            if (ImGui::Selectable(("wp "+std::to_string(k)+" @"+mu->waypoints[k].body).c_str(), selW))
                mSel={SelType::Waypoint, mSel.index, k};
            ImGui::PopID();
        }
    } else {
        ImGui::TextDisabled("Nothing selected.");
        ImGui::TextWrapped("Click a bone, a joint (right-click in the tree) or a muscle.");
    }
    ImGui::End();
}

void App::drawValidation() {
    ImGui::Begin("Anatomy check");
    int problems = 0;
    for (const auto& mu : mModel.muscles) {
        if (mu.waypoints.size() < 2) { ImGui::TextColored(ImVec4(1,0.4f,0.3f,1), "%s: <2 waypoints", mu.name.c_str()); problems++; }
        for (const auto& w : {mu.waypoints.front(), mu.waypoints.back()}) {
            if (mModel.findNode(w.body) == nullptr) {
                ImGui::TextColored(ImVec4(1,0.4f,0.3f,1), "%s: body '%s' missing", mu.name.c_str(), w.body.c_str()); problems++;
            }
        }
        double fmax = mu.pcsa_cm2 * mModel.meta.specific_tension_N_cm2;
        if (mu.pcsa_cm2 > 0 && std::fabs(fmax - mu.f0) > 1.0)
            { ImGui::TextColored(ImVec4(0.9f,0.8f,0.2f,1), "%s: f0 != sigma*PCSA", mu.name.c_str()); problems++; }
    }
    if (problems == 0) ImGui::TextColored(ImVec4(0.3f,1,0.4f,1), "No problems detected.");
    ImGui::End();
}

void App::drawAtlas() {
    ImGui::Begin("Atlas (OpenSim)");
    if (ImGui::Button("Import .osim...")) importOsim();
    ImGui::SameLine();
    ImGui::TextDisabled("%d refs", (int)mAtlas.size());
    ImGui::InputText("filter", mAtlasFilter, sizeof(mAtlasFilter));
    ImGui::Separator();
    ImGui::BeginChild("atlaslist");
    std::string flt = mAtlasFilter;
    for (auto& c : flt) c = (char)tolower(c);
    for (int i = 0; i < (int)mAtlas.size(); i++) {
        const AtlasEntry& a = mAtlas[i];
        if (!flt.empty()) {
            std::string ln = a.name; for (auto& c : ln) c = (char)tolower(c);
            if (ln.find(flt) == std::string::npos) continue;
        }
        ImGui::PushID(i);
        ImGui::Text("%s", a.name.c_str());
        ImGui::SameLine();
        ImGui::TextDisabled("f0=%.0f pen=%.2f", a.f0, a.pen_angle);
        ImGui::SameLine();
        if (ImGui::SmallButton("apply")) applyAtlas(a);
        ImGui::PopID();
    }
    ImGui::EndChild();
    ImGui::End();
}

void App::drawTrain() {
    ImGui::Begin("Training (PPO)");
    ImGui::Text("Telemetry: %s", mTrain.status().c_str());
    ImGui::Text("Client: %s", mTrain.clientConnected() ? "CONNECTED" : "waiting...");
    ImGui::Separator();
    Telemetry t = mTrain.latest();
    ImGui::Text("Iteration : %d", t.iteration);
    ImGui::Text("Reward    : %.3f", t.reward);
    ImGui::Text("Loss A/C/M: %.3f / %.4f / %.3f", t.loss_actor, t.loss_critic, t.loss_muscle);
    auto rh = mTrain.rewardHistory();
    if (!rh.empty())
        ImGui::PlotLines("reward", rh.data(), (int)rh.size(), 0, nullptr, 0.0f, 1.0f, ImVec2(0, 120));
    ImGui::Separator();
    if (!mTrain.trainingRunning()) {
        if (ImGui::Button("Export to data + train")) startTraining();
        ImGui::TextDisabled("(writes data/env.xml + skeleton + muscle)");
    } else {
        if (ImGui::Button("Stop training")) mTrain.stopTraining();
    }
    ImGui::End();
}

// --- editor for one <parameter> sub-group (gait/skeleton/torsion/muscle_*) ----
static void drawParamGroup(const char* title, std::vector<ed::ParamRange>& v) {
    if (!ImGui::TreeNode(title)) return;
    int removeAt = -1;
    for (int i = 0; i < (int)v.size(); i++) {
        ed::ParamRange& p = v[i];
        ImGui::PushID(i);
        char nm[64]; std::snprintf(nm, sizeof(nm), "%s", p.name.c_str());
        ImGui::SetNextItemWidth(120);
        if (ImGui::InputText("name", nm, sizeof(nm))) p.name = nm;
        ImGui::SameLine(); ImGui::SetNextItemWidth(80);
        ImGui::InputDouble("min", &p.min, 0, 0, "%.3f");
        ImGui::SameLine(); ImGui::SetNextItemWidth(80);
        ImGui::InputDouble("max", &p.max, 0, 0, "%.3f");
        ImGui::SameLine();
        if (ImGui::Checkbox("def", &p.hasDefault)) {}
        if (p.hasDefault) { ImGui::SameLine(); ImGui::SetNextItemWidth(80); ImGui::InputDouble("##d", &p.def, 0, 0, "%.3f"); }
        ImGui::SameLine();
        bool samp = p.hasSampling;
        if (ImGui::Checkbox("unif", &samp)) { p.hasSampling = samp; p.sampling = samp ? "uniform" : ""; }
        ImGui::SameLine();
        if (ImGui::SmallButton("x")) removeAt = i;
        ImGui::PopID();
    }
    if (removeAt >= 0) v.erase(v.begin() + removeAt);
    if (ImGui::SmallButton("+ add")) { ed::ParamRange r; r.name = "new"; v.push_back(r); }
    ImGui::TreePop();
}

void App::drawGaitNet() {
    ImGui::Begin("GaitNet");
    ed::EnvConfig& e = mModel.env;
    if (ImGui::CollapsingHeader("Environment (env.xml)", ImGuiTreeNodeFlags_DefaultOpen)) {
        const char* acts[] = { "pd", "torque", "muscle", "mass" };
        int cur = 3;
        for (int i = 0; i < 4; i++) if (e.actuator == acts[i]) cur = i;
        if (ImGui::Combo("actuator", &cur, acts, 4)) e.actuator = acts[cur];
        ImGui::InputDouble("defaultKp", &e.defaultKp);
        ImGui::InputDouble("defaultKv", &e.defaultKv);
        ImGui::InputDouble("damping", &e.damping);
        ImGui::InputInt("simHz", &e.simHz);
        ImGui::InputInt("controlHz", &e.controlHz);
        ImGui::InputDouble("actionScale", &e.actionScale);
        ImGui::InputDouble("timeWarping", &e.timeWarping);
        const char* rews[] = { "gaitnet", "deepmimic" };
        int rcur = (e.rewardType == "deepmimic") ? 1 : 0;
        if (ImGui::Combo("rewardType", &rcur, rews, 2)) e.rewardType = rews[rcur];
        const char* eoes[] = { "tuple", "time" };
        int ecur = (e.eoeType == "time") ? 1 : 0;
        if (ImGui::Combo("eoeType", &ecur, eoes, 2)) e.eoeType = eoes[ecur];
    }
    if (ImGui::CollapsingHeader("Reward weights")) {
        ImGui::InputDouble("HeadLinearAcc", &e.headLinearAccWeight);
        ImGui::InputDouble("HeadRot", &e.headRotWeight);
        ImGui::InputDouble("Step", &e.stepWeight);
        ImGui::InputDouble("Metabolic", &e.metabolicWeight);
        ImGui::InputDouble("AvgVel", &e.avgVelWeight);
    }
    if (ImGui::CollapsingHeader("Flags")) {
        ImGui::Checkbox("residual", &e.residual);
        ImGui::Checkbox("metabolicReward", &e.metabolicReward);
        ImGui::Checkbox("enforceSymmetry", &e.enforceSymmetry);
        ImGui::Checkbox("hardPhaseClipping", &e.hardPhaseClipping);
        ImGui::Checkbox("learningStd", &e.learningStd);
        ImGui::Checkbox("useJointState", &e.useJointState);
        ImGui::Checkbox("musclePoseOptimization", &e.musclePoseOptimization);
        ImGui::Checkbox("stanceLearning", &e.stanceLearning);
    }
    if (ImGui::CollapsingHeader("Parameters (<parameter> block)", ImGuiTreeNodeFlags_DefaultOpen)) {
        drawParamGroup("gait", mModel.params.gait);
        drawParamGroup("skeleton", mModel.params.skeleton);
        drawParamGroup("torsion", mModel.params.torsion);
        drawParamGroup("muscle_length", mModel.params.muscle_length);
        drawParamGroup("muscle_force", mModel.params.muscle_force);
    }
    ImGui::End();
}

void App::drawFillsPanel() {
    ImGui::Begin("Fills");
    ImGui::TextWrapped("Tissue fill generated by inference (metaballs + marching cubes) over the "
                       "anatomy, skinned to the skeleton. Finer resolution -> more faces (for "
                       "texturing, deformation and hair).");
    ImGui::Checkbox("Show fill", &mShowSkin);

    ImGui::SeparatorText("Generation parameters");
    ImGui::SliderFloat("Thickness (m)", &mSkinParams.inflate, 0.0f, 0.06f, "%.3f");
    ImGui::SliderFloat("Fusion smoothness", &mSkinParams.smooth, 0.0f, 0.15f, "%.3f");
    ImGui::SliderFloat("Body scale", &mSkinParams.bodyScale, 0.8f, 1.6f, "%.2f");
    ImGui::Checkbox("Wrap muscles", &mSkinParams.includeMuscles);
    ImGui::SliderFloat("Muscle radius x", &mSkinParams.muscleScale, 0.5f, 2.0f, "%.2f");
    ImGui::SliderFloat("Resolution (m/cell)", &mSkinParams.cell, 0.005f, 0.04f, "%.3f");
    ImGui::TextDisabled("smaller = more faces, slower (0.008 ~ tens of thousands)");

    ImGui::SeparatorText("Generate");
    ImGui::InputText("name", mFillNewName, sizeof(mFillNewName));
    ImGui::BeginDisabled(mFillRunning);
    if (ImGui::Button("Generate fill", ImVec2(-1, 0))) startFillGeneration(mFillNewName);
    ImGui::EndDisabled();

    ImGui::SeparatorText("Fills");
    for (int i = 0; i < (int)mModel.fills.size(); i++) {
        Fill& f = mModel.fills[i];
        ImGui::PushID(i);
        bool def = f.isDefault;
        if (ImGui::Checkbox("##def", &def)) { if (def) setDefaultFill(i); else f.isDefault = false; }
        ImGui::SameLine();
        bool active = (mActiveFill == i);
        if (ImGui::Selectable(f.name.c_str(), active, 0, ImVec2(120,0)))
            applyFill(i);
        ImGui::SameLine();
        if (ImGui::SmallButton("Apply")) applyFill(i);
        ImGui::SameLine();
        if (ImGui::SmallButton("X")) { deleteFill(i); ImGui::PopID(); break; }
        ImGui::PopID();
    }
    if (mModel.fills.empty()) ImGui::TextDisabled("No fills yet. Set params and Generate.");
    ImGui::TextDisabled("Checkbox = default (auto-applied on open).");
    ImGui::End();
}

void App::drawLights() {
    ImGui::Begin("Lights");
    float amb = (float)mModel.ambient;
    if (ImGui::SliderFloat("Ambient", &amb, 0.0f, 2.0f)) mModel.ambient = amb;
    ImGui::Separator();
    if (ImGui::Button("+ Directional")) {
        Light l; l.name = "Directional " + std::to_string(mModel.lights.size());
        l.type = 0; l.dir = {0.3, 0.8, 0.4}; mModel.lights.push_back(l);
        mSelLight = (int)mModel.lights.size()-1; mSel = {SelType::Light, mSelLight, -1};
    }
    ImGui::SameLine();
    if (ImGui::Button("+ Point")) {
        Light l; l.name = "Point " + std::to_string(mModel.lights.size());
        l.type = 1; l.dir = {0.0, 2.0, 1.5}; mModel.lights.push_back(l);
        mSelLight = (int)mModel.lights.size()-1; mSel = {SelType::Light, mSelLight, -1};
    }
    ImGui::SameLine();
    if (ImGui::Button("Remove") && mSelLight >= 0 && mSelLight < (int)mModel.lights.size()) {
        mModel.lights.erase(mModel.lights.begin() + mSelLight); mSelLight = -1;
    }
    ImGui::Separator();
    for (int i = 0; i < (int)mModel.lights.size(); i++) {
        ImGui::PushID(i);
        Light& L = mModel.lights[i];
        bool en = L.enabled;
        if (ImGui::Checkbox("##en", &en)) L.enabled = en;
        ImGui::SameLine();
        if (ImGui::Selectable((L.name + (L.type ? "  (point)" : "  (dir)")).c_str(), mSelLight == i)) {
            mSelLight = i; mSel = {SelType::Light, i, -1};
        }
        ImGui::PopID();
    }
    if (mSelLight >= 0 && mSelLight < (int)mModel.lights.size()) {
        Light& L = mModel.lights[mSelLight];
        ImGui::SeparatorText("Light properties");
        char buf[128]; std::strncpy(buf, L.name.c_str(), sizeof(buf)-1); buf[sizeof(buf)-1]=0;
        if (ImGui::InputText("name", buf, sizeof(buf))) L.name = buf;
        int ty = L.type;
        const char* tys[] = {"Directional", "Point"};
        if (ImGui::Combo("type", &ty, tys, 2)) L.type = ty;
        ImGui::Checkbox("enabled", &L.enabled);
        float col[3] = {(float)L.color[0], (float)L.color[1], (float)L.color[2]};
        if (ImGui::ColorEdit3("color", col)) L.color = {col[0], col[1], col[2]};
        float inten = (float)L.intensity;
        if (ImGui::SliderFloat("intensity", &inten, 0.0f, 4.0f)) L.intensity = inten;
        float d[3] = {(float)L.dir[0], (float)L.dir[1], (float)L.dir[2]};
        if (ImGui::DragFloat3(L.type ? "position" : "direction", d, 0.02f)) L.dir = {d[0], d[1], d[2]};
    }
    ImGui::End();
}

// ---- main frame ----
void App::frame() {
    ImGui_ImplOpenGL3_NewFrame();
    ImGui_ImplGlfw_NewFrame();
    ImGui::NewFrame();
    ImGuizmo::BeginFrame();

    // fullscreen dockspace host
    ImGuiViewport* vp = ImGui::GetMainViewport();
    ImGui::SetNextWindowPos(vp->WorkPos);
    ImGui::SetNextWindowSize(vp->WorkSize);
    ImGui::SetNextWindowViewport(vp->ID);
    ImGuiWindowFlags host = ImGuiWindowFlags_NoDocking | ImGuiWindowFlags_NoTitleBar |
        ImGuiWindowFlags_NoCollapse | ImGuiWindowFlags_NoResize | ImGuiWindowFlags_NoMove |
        ImGuiWindowFlags_NoBringToFrontOnFocus | ImGuiWindowFlags_NoNavFocus | ImGuiWindowFlags_MenuBar |
        ImGuiWindowFlags_NoBackground;  // let the passthru central node reveal the 3D backbuffer
    ImGui::PushStyleVar(ImGuiStyleVar_WindowRounding, 0);
    ImGui::PushStyleVar(ImGuiStyleVar_WindowBorderSize, 0);
    ImGui::PushStyleVar(ImGuiStyleVar_WindowPadding, ImVec2(0,0));
    ImGui::Begin("DockHost", nullptr, host);
    ImGui::PopStyleVar(3);
    drawMenuBar();
    drawTopToolbar();
    ImGuiID dockId = ImGui::GetID("MassDock");
    ImGui::DockSpace(dockId, ImVec2(0,0), ImGuiDockNodeFlags_PassthruCentralNode);

    // build a sensible default layout on first run
    static bool builtLayout = false;
    if (!builtLayout) {
        builtLayout = true;
        ImGui::DockBuilderRemoveNode(dockId);
        ImGui::DockBuilderAddNode(dockId, ImGuiDockNodeFlags_PassthruCentralNode | ImGuiDockNodeFlags_DockSpace);
        ImGui::DockBuilderSetNodeSize(dockId, ImGui::GetMainViewport()->WorkSize);
        ImGuiID left, right, center = dockId;
        left  = ImGui::DockBuilderSplitNode(center, ImGuiDir_Left,  0.20f, nullptr, &center);
        right = ImGui::DockBuilderSplitNode(center, ImGuiDir_Right, 0.26f, nullptr, &center);
        ImGuiID leftBottom = ImGui::DockBuilderSplitNode(left, ImGuiDir_Down, 0.55f, nullptr, &left);
        ImGuiID rightBottom = ImGui::DockBuilderSplitNode(right, ImGuiDir_Down, 0.5f, nullptr, &right);
        ImGui::DockBuilderDockWindow("Scene", left);
        ImGui::DockBuilderDockWindow("Properties", leftBottom);
        ImGui::DockBuilderDockWindow("Tools", right);
        ImGui::DockBuilderDockWindow("Training (PPO)", rightBottom);
        ImGui::DockBuilderDockWindow("Atlas (OpenSim)", rightBottom);
        ImGui::DockBuilderDockWindow("Anatomy check", rightBottom);
        ImGui::DockBuilderDockWindow("Lights", leftBottom);
        ImGui::DockBuilderDockWindow("Fills", rightBottom);
        ImGui::DockBuilderDockWindow("Viewport", center);
        ImGui::DockBuilderFinish(dockId);
    }
    ImGui::End();

    drawToolbar();
    drawTree();
    drawProperties();
    drawValidation();
    drawAtlas();
    drawTrain();
    drawGaitNet();
    drawLights();
    drawFillsPanel();

    // reflect edits in the running sim (debounced rebuild)
    maybeAutoApplySim();

    // fill generation: finalize when the worker finishes, and show a progress modal
    if (mFillRunning && mFillDone.load()) finalizeFill();
    if (mFillRunning) ImGui::OpenPopup("Generating fill");
    if (ImGui::BeginPopupModal("Generating fill", nullptr,
            ImGuiWindowFlags_AlwaysAutoResize | ImGuiWindowFlags_NoMove)) {
        ImGui::Text("Generating tissue fill (marching cubes)...");
        ImGui::ProgressBar(mFillProgress.load(), ImVec2(340, 0));
        if (!mFillRunning) ImGui::CloseCurrentPopup();
        ImGui::EndPopup();
    }

    // viewport window: render 3D to FBO, then show as an image (no compositing issues)
    ImGui::PushStyleVar(ImGuiStyleVar_WindowPadding, ImVec2(0,0));
    ImGui::Begin("Viewport");
    ImVec2 pos = ImGui::GetCursorScreenPos();
    ImVec2 size = ImGui::GetContentRegionAvail();
    if (size.x < 1) size.x = 1; if (size.y < 1) size.y = 1;
    mVpX = pos.x; mVpY = pos.y; mVpW = size.x; mVpH = size.y;

    drawScene();  // renders into the offscreen texture at (mVpW x mVpH)
    ImGui::Image((ImTextureID)(intptr_t)mRen.targetTexture(), size, ImVec2(0,1), ImVec2(1,0));

    bool hovered = ImGui::IsItemHovered();
    ImGuiIO& io = ImGui::GetIO();
    if (hovered && !ImGuizmo::IsUsing()) {
        // LEFT drag = orbit (pivot on selection); wheel = zoom
        bool orbiting = ImGui::IsMouseDragging(ImGuiMouseButton_Left) && !ImGuizmo::IsOver();
        if (mSel.type != SelType::None && (io.MouseWheel != 0 || orbiting))
            mCam.target = selectionCenter();
        if (io.MouseWheel != 0) mCam.zoom(io.MouseWheel);
        if (orbiting) mCam.orbit(io.MouseDelta.x, io.MouseDelta.y);

        // RIGHT drag = pan
        if (ImGui::IsMouseDragging(ImGuiMouseButton_Right))
            mCam.pan(io.MouseDelta.x, io.MouseDelta.y);

        // MIDDLE drag = pan
        if (ImGui::IsMouseDragging(ImGuiMouseButton_Middle))
            mCam.pan(io.MouseDelta.x, io.MouseDelta.y);
        if (ImGui::IsMouseReleased(ImGuiMouseButton_Left) && !ImGuizmo::IsOver()) {
            ImVec2 dr = ImGui::GetMouseDragDelta(ImGuiMouseButton_Left, 0.0f);
            if (std::fabs(dr.x) < 3 && std::fabs(dr.y) < 3)
                pickAt(io.MousePos.x, io.MousePos.y);
        }
    }
    drawGizmo();
    ImGui::End();
    ImGui::PopStyleVar();

    // keyboard shortcuts
    if (io.KeyCtrl && ImGui::IsKeyPressed(ImGuiKey_Z)) undo();
    if (io.KeyCtrl && ImGui::IsKeyPressed(ImGuiKey_Y)) redo();
    if (io.KeyCtrl && ImGui::IsKeyPressed(ImGuiKey_S)) saveMass(false);
    if (io.KeyCtrl && ImGui::IsKeyPressed(ImGuiKey_O)) openMass();
    if (io.KeyCtrl && (ImGui::IsKeyPressed(ImGuiKey_1) || ImGui::IsKeyPressed(ImGuiKey_Keypad1)))
        resetView();  // reset camera + pose (top-row or numpad 1)
    if (ImGui::IsKeyPressed(ImGuiKey_Delete)) removeSelected();

    // status bar
    ImGui::SetNextWindowBgAlpha(0.6f);
    if (ImGui::BeginViewportSideBar("##status", vp, ImGuiDir_Down, ImGui::GetFrameHeight(),
            ImGuiWindowFlags_NoDecoration|ImGuiWindowFlags_NoDocking)) {
        if (ImGui::BeginMenuBar()) { ImGui::Text("%s", mStatus.c_str()); ImGui::EndMenuBar(); }
        ImGui::End();
    }

    // ---- render (3D already rendered to FBO during the Viewport window) ----
    ImGui::Render();
    int w, h; glfwGetFramebufferSize(mWin, &w, &h);
    glViewport(0, 0, w, h);
    glClearColor(0.06f, 0.06f, 0.07f, 1.0f);
    glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);
    ImGui_ImplOpenGL3_RenderDrawData(ImGui::GetDrawData());
}

} // namespace ed
