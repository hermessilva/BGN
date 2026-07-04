#include <glad/glad.h>
#include <GLFW/glfw3.h>
#include <imgui.h>
#include <imgui_impl_glfw.h>
#include <imgui_impl_opengl3.h>
#include <cstdio>
#include <string>
#include <thread>
#include <chrono>
#include <cstdlib>
#include "App.h"
#include "Bootstrap.h"
#include "SimBridge.h"

static void glfwError(int e, const char* d){ std::fprintf(stderr, "GLFW %d: %s\n", e, d); }

// Headless helpers: derive project root from a metadata path (.../data/metadata.txt -> ...).
static std::string rootFromMeta(const std::string& meta) {
    std::string dir = meta;
    size_t s = dir.find_last_of("/\\"); if (s != std::string::npos) dir = dir.substr(0, s);
    size_t s2 = dir.find_last_of("/\\"); return (s2 != std::string::npos) ? dir.substr(0, s2) : ".";
}

int main(int argc, char** argv) {
    // ---- headless CLI ----
    if (argc >= 2 && std::string(argv[1]) == "--roundtrip" && argc >= 4) {
        std::string err;
        auto m = ed::BootstrapFromLegacy(argv[2], rootFromMeta(argv[2]), &err);
        if (!m) { std::fprintf(stderr, "bootstrap: %s\n", err.c_str()); return 1; }
        if (!ed::ExportToLegacy(*m, argv[3], &err)) { std::fprintf(stderr, "export: %s\n", err.c_str()); return 1; }
        std::printf("roundtrip OK: %zu nodes, %zu muscles -> %s\n",
                    m->skeleton.size(), m->muscles.size(), argv[3]);
        return 0;
    }
    if (argc >= 2 && std::string(argv[1]) == "--tomass" && argc >= 4) {
        std::string err;
        auto m = ed::BootstrapFromLegacy(argv[2], rootFromMeta(argv[2]), &err);
        if (!m) { std::fprintf(stderr, "bootstrap: %s\n", err.c_str()); return 1; }
        if (!m->SaveMass(argv[3], &err)) { std::fprintf(stderr, "save: %s\n", err.c_str()); return 1; }
        std::printf("saved %s (%zu nodes, %zu muscles)\n", argv[3], m->skeleton.size(), m->muscles.size());
        return 0;
    }

    if (argc >= 3 && std::string(argv[1]) == "--simtest") {
        std::string err;
        auto m = ed::Model::LoadMass(argv[2], &err);
        if (!m) { std::fprintf(stderr, "load: %s\n", err.c_str()); return 1; }
        std::string root = rootFromMeta(argv[2]);
        ed::SimBridge sim;
        sim.configure(root, root + "/build/editor_tmp");
        std::system(("cmd /c mkdir \"" + root + "\\build\\editor_tmp\" 2>nul").c_str());
        sim.setModel(*m);
        sim.setMode(ed::SimBridge::Kinematic);
        sim.start();
        for (int i = 0; i < 30; i++) {
            std::this_thread::sleep_for(std::chrono::milliseconds(100));
            if (i == 15) sim.setMode(ed::SimBridge::Dynamic);
        }
        auto pose = sim.pose();
        std::printf("simtest: mode-switch OK, t=%.2f, bodies in pose=%zu, status=%s\n",
                    sim.simTime(), pose.size(), sim.status().c_str());
        sim.stop();
        return pose.empty() ? 2 : 0;
    }

    if (argc >= 2 && std::string(argv[1]) == "--traintest") {
        ed::TrainBridge tb;
        tb.start(8765);
        std::this_thread::sleep_for(std::chrono::milliseconds(300));
        // spawn a python client that sends two telemetry lines
        const char* py =
            "python -c \"import socket,time,json;"
            "s=socket.create_connection(('127.0.0.1',8765));"
            "s.sendall((json.dumps({'iteration':1,'reward':0.42,'loss_actor':-0.1})+chr(10)).encode());"
            "time.sleep(0.2);"
            "s.sendall((json.dumps({'iteration':2,'reward':0.77})+chr(10)).encode());"
            "time.sleep(0.2)\"";
        std::system(py);
        std::this_thread::sleep_for(std::chrono::milliseconds(400));
        auto t = tb.latest();
        std::printf("traintest: connected=%d iteration=%d reward=%.3f history=%zu status=%s\n",
                    (int)tb.clientConnected(), t.iteration, t.reward, tb.rewardHistory().size(), tb.status().c_str());
        tb.stop();
        return (t.iteration == 2) ? 0 : 3;
    }

    glfwSetErrorCallback(glfwError);
    if (!glfwInit()) { std::fprintf(stderr, "glfwInit failed\n"); return 1; }

    glfwWindowHint(GLFW_CONTEXT_VERSION_MAJOR, 3);
    glfwWindowHint(GLFW_CONTEXT_VERSION_MINOR, 3);
    glfwWindowHint(GLFW_OPENGL_PROFILE, GLFW_OPENGL_CORE_PROFILE);
#ifdef __APPLE__
    glfwWindowHint(GLFW_OPENGL_FORWARD_COMPAT, GL_TRUE);
#endif

    GLFWwindow* win = glfwCreateWindow(1600, 950, "GaitNet Editor", nullptr, nullptr);
    if (!win) { std::fprintf(stderr, "window failed\n"); glfwTerminate(); return 1; }
    glfwMakeContextCurrent(win);
    glfwSwapInterval(1);

    if (!gladLoadGLLoader((GLADloadproc)glfwGetProcAddress)) {
        std::fprintf(stderr, "glad load failed\n"); return 1;
    }

    IMGUI_CHECKVERSION();
    ImGui::CreateContext();
    ImGuiIO& io = ImGui::GetIO();
    io.ConfigFlags |= ImGuiConfigFlags_DockingEnable;
    ImGui::StyleColorsDark();
    ImGui_ImplGlfw_InitForOpenGL(win, true);
    ImGui_ImplOpenGL3_Init("#version 330");

    ed::App app;
    if (!app.init(win)) return 1;

    // optional: open a .mass passed as the first argument
    bool genSkin = false;
    for (int i = 1; i < argc; i++) {
        std::string a = argv[i];
        if (a.size() > 5 && a.substr(a.size()-5) == ".mass") app.loadProjectPath(a);
        else if (a == "--skin") genSkin = true;
        else if (a == "--sim") app.startKinematicSim();
        else if (a == "--genfill") app.beginFill("default");
    }
    if (genSkin) app.generateSkin();

    while (!glfwWindowShouldClose(win)) {
        glfwPollEvents();
        app.frame();
        glfwSwapBuffers(win);
    }

    app.shutdown();
    ImGui_ImplOpenGL3_Shutdown();
    ImGui_ImplGlfw_Shutdown();
    ImGui::DestroyContext();
    glfwDestroyWindow(win);
    glfwTerminate();
    return 0;
}
