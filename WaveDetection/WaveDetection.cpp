
#include <iostream>
#include <fstream>
#include <vector>
#include <string>
#include <cmath>
#include <algorithm>
#include <sstream>
#include <random>
#include <omp.h>                

 // ───────────────────────── utilities ──────────────────────────────
inline double sqr(double x) { return x * x; }
inline double distXY(double x1, double y1, double x2, double y2) {
    double dx = x2 - x1, dy = y2 - y1; return std::sqrt(dx * dx + dy * dy);
}

// ───────────────────────── data structs ───────────────────────────
struct Spike {
    int    node;
    double start_time, end_time;
    double x, y;
};
struct WaveDetectionParams {
    double dt_factor, dt_th, dist_th;
    bool   use_structural, use_overlap, disallow_same_location;
};
struct WaveEdge {
    int wave_id, src_node, dst_node;
    double weight;
};

// ───────────────────────── I/O helpers ────────────────────────────
bool loadPositions(const std::string& file,
    std::vector<std::pair<double, double>>& pos) {
    std::ifstream f(file); if (!f) { std::cerr << "! " << file << "\n"; return false; }
    pos.clear(); double x, y; while (f >> x >> y) pos.push_back({ x,y }); return true;
}
bool loadSpikeList(const std::string& file, std::vector<Spike>& spikes,
    const std::vector<std::pair<double, double>>& pos) {
    std::ifstream f(file); if (!f) { std::cerr << "! " << file << "\n"; return false; }
    spikes.clear(); std::string line;
    while (std::getline(f, line)) {
        if (line.empty() || line[0] == '#') continue;
        std::stringstream ss(line); int n; double on, off;
        if (!(ss >> n >> on >> off)) continue;
        if (n < 0 || n >= static_cast<int>(pos.size())) continue;
        spikes.push_back({ n,on,off,pos[n].first,pos[n].second });
    }
    return true;
}
bool loadAdjacency(const std::string& file, std::vector<std::vector<int>>& A) {
    std::ifstream f(file); if (!f) { return false; }
    A.clear(); std::string line;
    while (std::getline(f, line)) {
        if (line.empty()) continue;
        std::stringstream ss(line); int v; std::vector<int> row;
        while (ss >> v) row.push_back(v); A.push_back(row);
    }
    return true;
}

// ───────────────────────── DFS & wave logic ───────────────────────
void exploreWaveForwardTime(int idx_in_sorted, int wave_id,
    std::vector<int>& wave_of,
    const std::vector<Spike>& spikes,
    const std::vector<int>& sorted_idx,
    const WaveDetectionParams& P,
    const std::vector<std::vector<int>>* adj,
    std::vector<WaveEdge>& edges,
    double pos_tol = 1e-6)
{
    int N = static_cast<int>(spikes.size());
    int i = sorted_idx[idx_in_sorted];
    int end = std::min(idx_in_sorted + 2000, N);

    for (int j_idx = idx_in_sorted + 1;j_idx < end;++j_idx) {
        int j = sorted_idx[j_idx];

        /* add edge inside the wave if dst already assigned */
        if (wave_of[j] == wave_id) {
            int ok = 0;
            double dt = spikes[j].start_time - spikes[i].start_time;
            if (!P.use_overlap) { if (dt<0 || dt>P.dt_th) ok = 1; }
            else {
                if (spikes[j].start_time<spikes[i].start_time ||
                    spikes[j].start_time>spikes[i].end_time) ok = 1;
            }
            if (P.use_structural) {
                if (adj && (*adj)[spikes[i].node][spikes[j].node] == 0) ok = 1;
            }
            else {
                if (distXY(spikes[i].x, spikes[i].y, spikes[j].x, spikes[j].y) > P.dist_th) ok = 1;
            }
            if (ok == 0) {
                if (dt == 0) {
                    std::uniform_real_distribution<double>d(0, 0.01);
                    std::mt19937 g(std::random_device{}()); dt = d(g);
                }
                edges.push_back({ wave_id,spikes[i].node,spikes[j].node,dt });
            }
        }

        if (wave_of[j] != 0) continue;

        double dt = spikes[j].start_time - spikes[i].start_time;
        if (!P.use_overlap) {
            if (dt<0 || dt>P.dt_th) continue;
        }
        else {
            if (spikes[j].start_time<spikes[i].start_time ||
                spikes[j].start_time>spikes[i].end_time) continue;
        }
        if (P.use_structural) {
            if (adj && (*adj)[spikes[i].node][spikes[j].node] == 0) continue;
        }
        else {
            if (distXY(spikes[i].x, spikes[i].y, spikes[j].x, spikes[j].y) > P.dist_th) continue;
        }
        if (P.disallow_same_location) {
            bool dup = false;
            for (int k = 0;k < N;++k) if (wave_of[k] == wave_id) {
                if (std::fabs(spikes[k].x - spikes[j].x) < pos_tol &&
                    std::fabs(spikes[k].y - spikes[j].y) < pos_tol) {
                    dup = true; break;
                }
            }
            if (dup) continue;
        }
        wave_of[j] = wave_id;
        if (dt == 0) {
            std::uniform_real_distribution<double>d(0, 0.01);
            std::mt19937 g(std::random_device{}()); dt = d(g);
        }
        edges.push_back({ wave_id,spikes[i].node,spikes[j].node,dt });
        exploreWaveForwardTime(j_idx, wave_id, wave_of, spikes, sorted_idx,
            P, adj, edges, pos_tol);
    }
}

std::vector<int> findWavesNoMergingForwardTime(std::vector<Spike>& spikes,
    const WaveDetectionParams& P,
    const std::vector<std::vector<int>>* adj,
    std::vector<WaveEdge>& edges)
{
    int N = static_cast<int>(spikes.size());
    std::vector<int> wave_of(N, 0);
    std::vector<int> sorted_idx(N); for (int i = 0;i < N;++i) sorted_idx[i] = i;
    std::sort(sorted_idx.begin(), sorted_idx.end(),
        [&](int a, int b) {return spikes[a].start_time < spikes[b].start_time;});

    int wave_cnt = 0;
    for (int k = 0;k < N;++k) {
        int i = sorted_idx[k];
        if (wave_of[i] == 0) {
            ++wave_cnt; wave_of[i] = wave_cnt;
            exploreWaveForwardTime(k, wave_cnt, wave_of, spikes, sorted_idx,
                P, adj, edges);
            if (wave_cnt > 100000) break;
        }
    }
    return wave_of;
}

// ───────────────────────── runner per file ───────────────────────
void runWaveDetection(const std::string& spikeFile, const std::string& label,
    const std::string& base,
    WaveDetectionParams P,
    const std::vector<std::vector<int>>* adj)
{
    std::string posFile =
        spikeFile.substr(0, spikeFile.find_last_of("/\\\\")) + "/node_positions.txt";
    std::vector<std::pair<double, double>> pos;
    if (!loadPositions(posFile, pos)) return;

    std::vector<Spike> spikes;
    if (!loadSpikeList(spikeFile, spikes, pos)) return;

    /* local edge list */
    std::vector<WaveEdge> edges;

    double sum = 0; for (auto& s : spikes) sum += s.end_time - s.start_time;
    P.dt_th = spikes.empty() ? 0 : sum / spikes.size() * P.dt_factor;

    auto wave_of = findWavesNoMergingForwardTime(spikes, P, adj, edges);

    /* outputs */
    {
        std::ofstream f(base + "wave_assignments_" + label + ".txt");
        f << "idx wave node start end\n";
        for (size_t i = 0;i < spikes.size();++i)
            f << i << ' ' << wave_of[i] << ' ' << spikes[i].node << ' '
            << spikes[i].start_time << ' ' << spikes[i].end_time << '\n';
    }
    {
        std::ofstream f(base + "wave_edges_" + label + ".txt");
        f << "wave src dst wt\n";
        for (auto& e : edges) f << e.wave_id << ' ' << e.src_node << ' '
            << e.dst_node << ' ' << e.weight << '\n';
    }
}

int main()
{
    const std::string root =
        "C:/UM FNM/Insulin Secretion/SL_Izh_Secretion_Model/"
        "Kuramoto_Izhikevich/results/output_iteration_";

    const bool useStructural = true;

    WaveDetectionParams proto;
    proto.dt_factor = 0.50;
    proto.dist_th = 0.10;
    proto.use_structural = useStructural;
    proto.use_overlap = true;
    proto.disallow_same_location = true;

    WaveDetectionParams protoSL;
    protoSL.dt_factor = 0.30;
    protoSL.dist_th = 0.10;
    protoSL.use_structural = useStructural;
    protoSL.use_overlap = true;
    protoSL.disallow_same_location = true;

    const int NUM_THREADS = 15;
    omp_set_dynamic(0);
    omp_set_num_threads(NUM_THREADS);

#pragma omp parallel for schedule(dynamic)
    for (int it = 0; it < 100; ++it) {
        std::string base = root + std::to_string(it) + "/";

        std::vector<std::vector<int>> adj;
        if (useStructural &&
            !loadAdjacency(base + "original_adjacency.txt", adj)) {
            std::cerr << "[skip] no adjacency in " << base << "\n";
            continue;
        }

        runWaveDetection(base + "stuart_landau_events.txt",
            "StuartLandau", base, protoSL,
            useStructural ? &adj : nullptr);

        runWaveDetection(base + "izhikevich_burst_events.txt",
            "IzhikevichBursts", base, proto,
            useStructural ? &adj : nullptr);

        runWaveDetection(base + "izhikevich_spike_events.txt",
            "IzhikevichSpikes", base, proto,
            useStructural ? &adj : nullptr);
    }

    std::cout << "\nAll done!\n";
    return 0;
}
