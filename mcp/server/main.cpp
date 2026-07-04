// mass-mcp — standalone MCP server over TCP (newline-delimited JSON-RPC), hosting
// the libmassedit McpServer on a live .mass model. Single-threaded: each request
// is handled inline, so the model has exactly one writer (this thread) — the same
// single-writer contract the in-process Arena bridge enforces via McpQueue.
//
// Usage: mass-mcp <model.mass> [port]     (default port 8766)
#include "../Mcp.h"
#include "../MassModel.h"
#include "../Index.h"
#include <boost/asio.hpp>
#include <nlohmann/json.hpp>
#include <cstdio>
#include <string>

using boost::asio::ip::tcp;
using namespace mass;
using json = nlohmann::json;

int main(int argc, char** argv) {
    if (argc < 2) { std::fprintf(stderr, "usage: mass-mcp <model.mass> [port]\n"); return 1; }
    std::string path = argv[1];
    unsigned short port = (argc > 2) ? (unsigned short)std::atoi(argv[2]) : 8766;

    std::string err;
    auto mo = Model::LoadMass(path, &err);
    if (!mo) { std::fprintf(stderr, "load %s: %s\n", path.c_str(), err.c_str()); return 2; }
    Model model = std::move(*mo);
    model.assignUids();
    Index ix; ix.build(model);
    McpServer server;
    std::printf("mass-mcp: %s (%zu bones, %zu muscles) on port %u\n",
                path.c_str(), model.skeleton.size(), model.muscles.size(), port);
    std::fflush(stdout);

    try {
        boost::asio::io_context io;
        tcp::acceptor acc(io, tcp::endpoint(tcp::v4(), port));
        for (;;) {
            tcp::socket sock(io);
            acc.accept(sock);
            boost::asio::streambuf buf;
            boost::system::error_code ec;
            for (;;) {
                std::size_t n = boost::asio::read_until(sock, buf, '\n', ec);
                if (ec || n == 0) break;
                std::istream is(&buf);
                std::string line;
                std::getline(is, line);
                if (line.empty()) continue;
                json req = json::parse(line, nullptr, /*allow_exceptions*/false);
                json resp;
                if (req.is_discarded())
                    resp = { {"jsonrpc","2.0"}, {"id", nullptr},
                             {"error", { {"code",-32700}, {"message","parse error"} }} };
                else
                    resp = server.handle(req, model, ix);
                std::string out = resp.dump() + "\n";
                boost::asio::write(sock, boost::asio::buffer(out), ec);
                if (ec) break;
            }
        }
    } catch (const std::exception& e) {
        std::fprintf(stderr, "server error: %s\n", e.what());
        return 3;
    }
    return 0;
}
