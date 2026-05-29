#include <iostream>
#include <vector>
#include <random>
#include <cmath>
#include <fstream>
#include <algorithm>
#include <chrono>
#include <cassert>
#include <tuple>
#include <iomanip>
#include <sstream>
#include <string>
#include <errno.h>
#include <numeric>
#include <atomic>

#ifdef _WIN32
#include <direct.h>
#else
#include <sys/stat.h>
#endif

#include <omp.h>

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif


// helpers
bool create_directory(const std::string& path)
{
#ifdef _WIN32
    int ret = _mkdir(path.c_str());
#else
    int ret = mkdir(path.c_str(), 0777);
#endif
    return (ret == 0 || errno == EEXIST);
}

inline double sqr(double x) { return x * x; }

template <typename T>
constexpr const T& clamp(const T& v, const T& lo, const T& hi)
{
    return (v < lo) ? lo : (hi < v) ? hi : v;
}

// data containers
struct SimulationResults {
    double avgSL;    // avg Kuramoto order parameter
    double avgBurst;
    double avgSpike;
};

struct Event {
    int    node;
    double t_on;
    double t_off;
};

// random geometric network
void createRandomGeometricNetwork(
    int N,
    double avg_degree,
    std::vector<std::vector<bool>>& adjacency,
    std::vector<std::vector<double>>& weights,
    std::vector<std::vector<int>>& neighborList,
    double& D,
    bool buildAdjMatrix = true,
    const std::string& output_dir = "")
{
    D = std::sqrt(avg_degree / (N * M_PI));
    double min_dist = 0.75 / std::sqrt(N);

    std::mt19937_64 rng(std::random_device{}());
    std::uniform_real_distribution<double> dist01(0.0, 1.0);
    std::exponential_distribution<double>  distExp(1.0);

    std::vector<std::pair<double, double>> pos;
    pos.reserve(N);

    while ((int)pos.size() < N) {
        double x = dist01(rng), y = dist01(rng);
        bool ok = true;
        for (auto& p : pos) {
            if (std::hypot(x - p.first, y - p.second) < min_dist) { ok = false; break; }
        }
        if (ok) pos.emplace_back(x, y);
    }

    if (buildAdjMatrix) adjacency.assign(N, std::vector<bool>(N, false));
    weights.assign(N, std::vector<double>(N, 0.0));
    neighborList.assign(N, {});

    for (int i = 0; i < N; ++i)
        for (int j = i + 1; j < N; ++j) {
            double d = std::hypot(pos[i].first - pos[j].first,
                pos[i].second - pos[j].second);
            if (d < D) {
                neighborList[i].push_back(j);
                neighborList[j].push_back(i);
                if (buildAdjMatrix) {
                    adjacency[i][j] = adjacency[j][i] = true;
                    double w = distExp(rng);
                    weights[i][j] = weights[j][i] = w;
                }
            }
        }

    std::string pre = output_dir.empty() ? "" : output_dir + "/";

    {   // positions
        std::ofstream f(pre + "node_positions.txt");
        for (auto& p : pos) f << p.first << ' ' << p.second << '\n';
    }
    if (buildAdjMatrix) {   // adjacency
        std::ofstream f(pre + "original_adjacency.txt");
        for (int i = 0; i < N; ++i) {
            for (int j = 0; j < N; ++j)
                f << (adjacency[i][j] ? 1 : 0) << (j < N - 1 ? ' ' : '\n');
        }
    }
    {   // weighted adjacency
        std::ofstream f(pre + "original_weighted_adjacency.txt");
        for (int i = 0; i < N; ++i) {
            for (int j = 0; j < N; ++j)
                f << weights[i][j] << (j < N - 1 ? ' ' : '\n');
        }
    }
}

// Kuramoto RK2 update
void updateKuramotoRK2(
    const std::vector<std::vector<int>>& neigh,
    const std::vector<double>& wdeg,                 // not used
    const std::vector<std::vector<double>>& W,
    const std::vector<double>& omega,
    double K, double dt,
    std::vector<double>& theta)
{
    (void)wdeg; // not used

    const int N = (int)theta.size();
    std::vector<double> k1(N), k2(N);

    // k1
    for (int i = 0; i < N; ++i) {
        double s = 0.0;
        for (int j : neigh[i]) {
            s += W[i][j] * std::sin(theta[j] - theta[i]);
        }
        k1[i] = omega[i] + K * s;
    }

    // midpoint
    std::vector<double> th_mid(N);
    for (int i = 0; i < N; ++i)
        th_mid[i] = theta[i] + 0.5 * dt * k1[i];

    // k2
    for (int i = 0; i < N; ++i) {
        double s = 0.0;
        for (int j : neigh[i]) {
            s += W[i][j] * std::sin(th_mid[j] - th_mid[i]);
        }
        k2[i] = omega[i] + K * s;
    }

    // update
    for (int i = 0; i < N; ++i) {
        theta[i] += dt * k2[i];

        // wrap to keep theta bounded
        if (theta[i] > M_PI)        theta[i] -= 2.0 * M_PI;
        else if (theta[i] < -M_PI)  theta[i] += 2.0 * M_PI;
    }
}


// Izhikevich RK2 update
void updateIzhikevichRK2(
    const std::vector<std::vector<int>>& neigh,
    const std::vector<double>& wdeg,
    const std::vector<std::vector<double>>& W,
    double a, double b, const std::vector<double>& c, double d,
    double K, double dt, const std::vector<double>& theta, 
    std::vector<double>& v, std::vector<double>& u,
    double tau_F, double f_up, double alpha_g, double Gmax,
    double p0, double pmax, double Fsat,
    std::mt19937_64& rng,
    std::vector<double>& F, std::vector<double>& G, std::vector<double>& insulin)
{
    int N = (int)v.size();
    std::vector<double> I(N), cV(N), cU(N);

    
    // K is the fast coupling (K_izh)
    const double K_max = 20.0;
    const double K_min = 20.0 * 0.5;  // 50% reduction -> 0.5
    double Imin = -15.0 + (10.0) * (K_max - K) / (K_max - K_min);
    //Imin = clamp(Imin, -15.0, -5.0);
    double Imax = 5.0 + (15.0) * (K_max - K) / (K_max - K_min);

    Imax = 5.0;
    //Imin = -15.0;
    // drive: x_slow = cos(theta) in [-1,1], then same mapping to [Imin, Imax]
    for (int i = 0; i < N; ++i) {
        double x_slow = std::cos(theta[i]);
        I[i] = Imin + ((x_slow + 1.0) / 2.0) * (Imax - Imin);
    }
    

    //K = 0.5;
    double Delta_eff = 30.0;
    // coupling term on v
    /*for (int i = 0; i < N; ++i) {
        double s = 0.0;
        double su = 0.0;
        for (int j : neigh[i]) {
            s += W[i][j] * v[j];
            su += W[i][j] * u[j];

        }
        cV[i] = K * (s - wdeg[i] * v[i]);
        cU[i] = K * (su - wdeg[i] * u[i]);
    }*/
    for (int i = 0; i < N; ++i) {
        double s = 0.0;
        for (int j : neigh[i]) {
            const double dv = (v[j] - v[i]) / Delta_eff;
            s += W[i][j] * std::tanh(dv);
        }
        cV[i] = K * s;

        cU[i] = 0.0;// unused
    }

    // RK2
    std::vector<double> k1v(N), k1u(N);
    for (int i = 0; i < N; ++i) {
        k1v[i] = 0.04 * sqr(v[i]) + 5 * v[i] + 140 - u[i] + I[i] + cV[i];
        k1u[i] = a * (b * v[i] - u[i]);// +cU[i];
    }

    std::vector<double> vm(N), um(N);
    for (int i = 0; i < N; ++i) {
        vm[i] = v[i] + 0.5 * dt * k1v[i];
        um[i] = u[i] + 0.5 * dt * k1u[i];
    }

    std::vector<double> cV2(N), cU2(N);
    /*for (int i = 0; i < N; ++i) {
        double s = 0.0;
        double su = 0.0;
        for (int j : neigh[i]) {
            s += W[i][j] * vm[j];
            su += W[i][j] * um[j];

        }
        cV2[i] = K * (s - wdeg[i] * vm[i]);
        cU2[i] = K * (su - wdeg[i] * um[i]);
    }*/
    for (int i = 0; i < N; ++i) {
        double s = 0.0;
        for (int j : neigh[i]) {
            const double dv = (vm[j] - vm[i]) / Delta_eff;
            s += W[i][j] * std::tanh(dv);
        }
        cV2[i] = K * s;
        cU2[i] = 0.0;
    }

    std::vector<double> k2v(N), k2u(N);
    for (int i = 0; i < N; ++i) {
        k2v[i] = 0.04 * sqr(vm[i]) + 5 * vm[i] + 140 - um[i] + I[i] + cV2[i];
        k2u[i] = a * (b * vm[i] - um[i]);// +cU2[i];
    }

    for (int i = 0; i < N; ++i) {
        v[i] += dt * k2v[i];
        u[i] += dt * k2u[i];
    }

    // insulin
    for (int i = 0; i < N; ++i) {
        F[i] += dt * (-F[i] / tau_F);
        G[i] += dt * (alpha_g * (Gmax - G[i]));
        if (v[i] >= 30.0) {
            v[i] = c[i];
            u[i] += d / (1 + 0.02 * u[i]);
            F[i] += f_up;
            if (G[i] > 0.0) {
                double p = clamp(p0 + (F[i] / (Fsat + F[i])) * pmax, 0.0, 1.0);
                if (std::bernoulli_distribution(p)(rng)) {
                    G[i] += 1.0; // unused
                    insulin[i] += 1.0;
                    F[i] = 0.0;
                }
            }
        }
    }
}

// simulation
SimulationResults simulate(
    int N, double T, double dt, double T_tr, double avg_deg,
    double mu, double mean_w0, double rng_w0, double K_sl,     // mu unused
    double a, double b, double d, double K_izh,
    bool save_data, const std::string& out_dir)
{
    (void)mu; // unused in Kuramoto version

    int steps = int(T / dt), trans = int(T_tr / dt);
    std::mt19937_64 rng(std::random_device{}());

    // 1  initial Kuramoto
    std::uniform_real_distribution<double> dTh(-M_PI, M_PI);
    std::uniform_real_distribution<double> dW0(mean_w0 - rng_w0, mean_w0 + rng_w0);
    std::vector<double> theta(N), omega(N);
    for (int i = 0; i < N; ++i) {
        theta[i] = dTh(rng);
        omega[i] = dW0(rng);
    }

    // 2  initial Izh
    std::vector<double> v(N, -70.0), u(N), c_reset(N);
    std::uniform_real_distribution<double> dC(-35.0, -30.0);
    for (int i = 0; i < N; ++i) { c_reset[i] = dC(rng); u[i] = b * v[i]; }

    // 3  network
    std::vector<std::vector<bool>> A;
    std::vector<std::vector<double>> W;
    std::vector<std::vector<int>> neigh;
    double D;
    createRandomGeometricNetwork(N, avg_deg, A, W, neigh, D, true, out_dir);

    std::vector<double> wdeg(N, 0.0);
    for (int i = 0; i < N; ++i) for (int j : neigh[i]) wdeg[i] += W[i][j];

    // 4  book-keeping 
    std::vector<std::vector<int>> coSL(N, std::vector<int>(N, 0));
    std::vector<std::vector<int>> coBurst(N, std::vector<int>(N, 0));
    std::vector<std::vector<int>> coSpike(N, std::vector<int>(N, 0));
    std::vector<int> cntSL(N, 0), cntBurst(N, 0), cntSpike(N, 0);
    int binsSL = 0, binsIzh = 0;

    // events
    std::vector<Event> evSL, evBurst, evSpike;
    std::vector<char> prevSL(N, 0), prevBurst(N, 0), prevSpike(N, 0);
    std::vector<double> tOnSL(N, 0.0), tOnBurst(N, 0.0), tOnSpike(N, 0.0);

    // burst stats
    std::vector<int>    burstCnt(N, 0);        // finished bursts
    std::vector<int>    spikeCntSum(N, 0);     // Σ spikes over bursts
    std::vector<double> fracTimeSum(N, 0.0);   // Σ spikeTime / burstTime

    std::vector<int>    spikeCntInBurst(N, 0);
    std::vector<double> spikeTimeInBurst(N, 0.0);

    // insulin-per-burst stats
    std::vector<int>    secrCntSum(N, 0);
    std::vector<int>    secrCntInBurst(N, 0);
    std::vector<int>    totalSecr(N, 0);
    std::vector<double> relTimeSum(N, 0.0);

    // rank-wise release timing
    std::vector<std::vector<double>> rankTimeSum(N);
    std::vector<std::vector<int>>    rankCount(N);

    // insulin arrays
    double tau_F = 30.0, f_up = 0.3, alpha_g = 0.002;
    double Gmax = 100.0, p0 = 0.005, pmax = 0.4, Fsat = 3.0;
    std::vector<double> F(N, 0.0), G(N, Gmax), insulin(N, 0.0);

    // mean traces
    std::vector<double> tSL, meanSL, tIzh, meanIzh;

    // Kuramoto order parameter tracking
    double Rsum = 0.0;
    int    Rcnt = 0;

    std::vector<double> insulinTimes;
    std::vector<double> insulinStep;
    double prevInsulinSum = 0.0;

    double prevI_dbg[10] = { 0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0 };
    std::vector<double> prevIns(N, 0.0);

    /*FILE* out = nullptr;
    std::string pre = out_dir.empty() ? "" : out_dir + "/";
    std::string full_path = pre + "insulin_fast_trace.txt";
#ifdef _WIN32
    fopen_s(&out, full_path.c_str(), "w");
#else
    out = std::fopen(full_path.c_str(), "w");
#endif*/

    // ---------------------------------------------------------------------
    for (int s = 0; s < steps; ++s) {

        // slow: Kuramoto
        updateKuramotoRK2(neigh, wdeg, W, omega, K_sl, dt, theta);

        // fast + insulin
        updateIzhikevichRK2(neigh, wdeg, W, a, b, c_reset, d,
            K_izh, dt, theta, v, u,
            tau_F, f_up, alpha_g, Gmax, p0, pmax, Fsat,
            rng, F, G, insulin);

        if (s < trans) continue;

        // Instantaneous insulin secretion: Δ(total) since previous step
        double curSum = std::accumulate(insulin.begin(), insulin.end(), 0.0);
        double releasedThisStep = curSum - prevInsulinSum;

        if (releasedThisStep > 0.0) {
            insulinTimes.push_back(s * dt);
            insulinStep.push_back(releasedThisStep);
        }
        prevInsulinSum = curSum;

        // -------- current binary states ----------------------------------
        std::vector<char> bSL(N), bBurst(N), bSpike(N);
        double sumSL = 0.0, sumI = 0.0;

        // Kuramoto order parameter
        double Rx = 0.0, Ry = 0.0;

        double t_now = s * dt;
        for (int i = 0; i < N; ++i)
        {
            // slow proxy variable in [-1,1]
            double x_slow = std::cos(theta[i]);

            // order parameter components
            Rx += std::cos(theta[i]);
            Ry += std::sin(theta[i]);

            // define binary "slow active" state
            bSL[i] = (x_slow > 0.0);

            // Spike
            char spike_now = (v[i] > -30.0);
            bSpike[i] = spike_now;

            // Burst state machine
            char burst_now = prevBurst[i];
            if (!burst_now && v[i] > -30.0) {
                burst_now = 1;
                tOnBurst[i] = t_now;
                secrCntInBurst[i] = 0;
                spikeCntInBurst[i] = 0;
                spikeTimeInBurst[i] = 0.0;
            }
            if (burst_now && v[i] < -60.0) {
                burst_now = 0;
            }
            bBurst[i] = burst_now;

            // mean traces
            sumSL += x_slow;
            sumI += v[i];

            // spike-time fraction inside burst
            if (burst_now && spike_now) spikeTimeInBurst[i] += dt;
            if (spike_now && !prevSpike[i] && burst_now) ++spikeCntInBurst[i];

            // rank-wise insulin release detection
            int rel_now = int(std::round(insulin[i] - prevIns[i]));
            if (rel_now > 0 && burst_now) {
                double dt_rel = t_now - tOnBurst[i];
                for (int r = 0; r < rel_now; ++r) {
                    int rank = secrCntInBurst[i];

                    if ((int)rankTimeSum[i].size() <= rank) {
                        rankTimeSum[i].resize(rank + 1, 0.0);
                        rankCount[i].resize(rank + 1, 0);
                    }
                    rankTimeSum[i][rank] += dt_rel;
                    rankCount[i][rank] += 1;

                    ++secrCntInBurst[i];
                    ++totalSecr[i];
                }
            }
            prevIns[i] = insulin[i];
        }

        // store mean traces
        tSL.push_back(t_now);   meanSL.push_back(sumSL / N);   // mean(cos theta)
        tIzh.push_back(t_now);  meanIzh.push_back(sumI / N);

        // Kuramoto order parameter R(t)
        double R = std::sqrt(Rx * Rx + Ry * Ry) / double(N);
        Rsum += R;
        Rcnt += 1;

        // -------- event detection ---------------------------------------
        for (int i = 0; i < N; ++i) {

            // slow
            if (bSL[i] && !prevSL[i])          tOnSL[i] = t_now;
            if (!bSL[i] && prevSL[i])          evSL.push_back({ i, tOnSL[i], t_now });
            prevSL[i] = bSL[i];

            // Burst (closing)
            if (!bBurst[i] && prevBurst[i]) {
                evBurst.push_back({ i, tOnBurst[i], t_now });

                double dur = t_now - tOnBurst[i];
                if (dur > 0.0) {
                    ++burstCnt[i];
                    spikeCntSum[i] += spikeCntInBurst[i];
                    fracTimeSum[i] += spikeTimeInBurst[i] / dur;

                    secrCntSum[i] += secrCntInBurst[i];
                    secrCntInBurst[i] = 0;
                }
            }

            // Spike events
            if (bSpike[i] && !prevSpike[i])    tOnSpike[i] = t_now;
            if (!bSpike[i] && prevSpike[i])    evSpike.push_back({ i, tOnSpike[i], t_now });

            // IMPORTANT memory updates
            prevBurst[i] = bBurst[i];
            prevSpike[i] = bSpike[i];
        }

        // -------- co-activity -------------------------------------------
        for (int i = 0; i < N; ++i) {
            if (bSL[i])    cntSL[i]++;
            if (bBurst[i]) cntBurst[i]++;
            if (bSpike[i]) cntSpike[i]++;
            for (int j = i + 1; j < N; ++j) {
                if (bSL[i] && bSL[j])       coSL[i][j]++, coSL[j][i]++;
                if (bBurst[i] && bBurst[j]) coBurst[i][j]++, coBurst[j][i]++;
                if (bSpike[i] && bSpike[j]) coSpike[i][j]++, coSpike[j][i]++;
            }
        }
        binsSL++; binsIzh++;

        // Debug printing
        /*if (s > 1000000 && out) {
            std::fprintf(out, "%i ", s);
            for (int i = 0; i < 10; ++i) std::fprintf(out, "%.3lf ", v[i]);
            for (int i = 0; i < 10; ++i) {
                std::fprintf(out, "%i ", int(insulin[i]));
                prevI_dbg[i] = insulin[i];
            }
            std::fprintf(out, "\n");
        }*/
    } // main loop

    //if (out) std::fclose(out);

    // close events that run until the last step
    double t_end = T;
    for (int i = 0; i < N; ++i) {
        if (prevSL[i])    evSL.push_back({ i, tOnSL[i], t_end });
        if (prevBurst[i]) {
            evBurst.push_back({ i, tOnBurst[i], t_end });
            double dur = t_end - tOnBurst[i];
            if (dur > 0.0) {
                ++burstCnt[i];
                spikeCntSum[i] += spikeCntInBurst[i];
                fracTimeSum[i] += spikeTimeInBurst[i] / dur;
            }
        }
        if (prevSpike[i]) evSpike.push_back({ i, tOnSpike[i], t_end });
    }

    // outputs
    if (save_data) {
        std::string pre2 = out_dir.empty() ? "" : out_dir + "/";

        // insulin
        {
            std::ofstream f(pre2 + "insulin_secretion.txt");
            for (int i = 0; i < N; ++i) f << i << ' ' << insulin[i] << '\n';

            std::ofstream f_it(pre2 + "insulin_time_trace.txt");
            f_it << std::fixed << std::setprecision(2);
            for (size_t k = 0; k < insulinTimes.size(); ++k)
                f_it << insulinTimes[k] << ' ' << insulinStep[k] << '\n';
        }

        // mean traces
        {
            std::ofstream f(pre2 + "stuart_landau_mean.txt"); // mean(cos(theta))
            f << std::fixed << std::setprecision(2);
            for (size_t k = 0; k < tSL.size(); k += 100)
                f << tSL[k] << ' ' << meanSL[k] << '\n';
        }
        {
            std::ofstream f(pre2 + "izhikevich_mean.txt");
            for (size_t k = 0; k < tIzh.size(); ++k)
                f << tIzh[k] << ' ' << meanIzh[k] << '\n';
        }

        // events
        auto dumpEvents = [&](const std::vector<Event>& ev, const std::string& fn) {
            std::ofstream f(pre2 + fn);
            f << "# node  t_on  t_off\n";
            f << std::fixed << std::setprecision(2);
            for (auto& e : ev) f << e.node << ' ' << e.t_on << ' ' << e.t_off << '\n';
            };
        dumpEvents(evSL, "stuart_landau_events.txt");       
        dumpEvents(evBurst, "izhikevich_burst_events.txt");
        dumpEvents(evSpike, "izhikevich_spike_events.txt");

        // burst_spike_stats
        {
            std::ofstream f(pre2 + "burst_spike_stats.txt");
            f << "# node  avg_spikes_per_burst  avg_spike_fraction\n";
            f << std::fixed << std::setprecision(4);
            for (int i = 0; i < N; ++i) {
                double avgN = burstCnt[i] ? double(spikeCntSum[i]) / burstCnt[i] : 0.0;
                double avgFr = burstCnt[i] ? fracTimeSum[i] / burstCnt[i] : 0.0;
                f << i << ' ' << avgN << ' ' << avgFr << '\n';
            }
        }

        // burst_insulin_stats
        {
            std::ofstream f(pre2 + "burst_insulin_stats.txt");
            f << "# node  avg_secretions_per_burst  avg_release_time_from_burst_start\n";
            f << std::fixed << std::setprecision(4);
            for (int i = 0; i < N; ++i) {
                double avgN = burstCnt[i] ? double(secrCntSum[i]) / burstCnt[i] : 0.0;
                double avgDt = totalSecr[i] ? relTimeSum[i] / totalSecr[i] : 0.0;
                f << i << ' ' << avgN << ' ' << avgDt << '\n';
            }
        }

        // burst_insulin_rank_timing
        {
            std::ofstream f(pre2 + "burst_insulin_rank_timing.txt");

            int maxRanks = 0;
            for (auto& rc : rankCount)
                if ((int)rc.size() > maxRanks) maxRanks = (int)rc.size();

            f << "# node  n_bursts";
            for (int r = 0; r < maxRanks; ++r)
                f << "  avg_dt_r" << (r + 1);
            f << '\n' << std::fixed << std::setprecision(4);

            for (int i = 0; i < N; ++i) {
                f << i << ' ' << burstCnt[i];
                for (int r = 0; r < maxRanks; ++r) {
                    double avg = 0.0;
                    if (r < (int)rankCount[i].size() && rankCount[i][r] > 0)
                        avg = rankTimeSum[i][r] / rankCount[i][r];
                    f << ' ' << avg;
                }
                f << '\n';
            }
        }

        // co-activity matrices
        auto dumpCo = [&](const std::vector<std::vector<int>>& M,
            const std::vector<int>& cnt, int bins,
            const std::string& fn) {
                if (bins < 1) return;
                double TT = double(bins);
                std::ofstream f(pre2 + fn);
                for (int i = 0; i < N; ++i) {
                    double p1 = cnt[i] / TT;
                    for (int j = 0; j < N; ++j) {
                        double p2 = cnt[j] / TT;
                        double p11 = M[i][j] / TT;
                        double phi = 0.0;
                        if (p1 * (1 - p1) * p2 * (1 - p2) > 1e-12)
                            phi = p11 / std::sqrt(p1 * p2);
                        f << phi << (j < N - 1 ? ' ' : '\n');
                    }
                }
            };
        dumpCo(coSL, cntSL, binsSL, "coSL_matrix.txt");
        dumpCo(coBurst, cntBurst, binsIzh, "coIzhBurst_matrix.txt");
        dumpCo(coSpike, cntSpike, binsIzh, "coIzhSpike_matrix.txt");
    }

    // average φ for Burst/Spike + Kuramoto avgSL computed as <R>
    auto avgPhi = [&](const std::vector<std::vector<int>>& M,
        const std::vector<int>& cnt, int bins) {
            if (bins < 1) return 0.0;
            double TT = double(bins), sum = 0.0, pairs = 0.0;
            for (int i = 0; i < N; ++i) {
                double p1 = cnt[i] / TT;
                for (int j = i + 1; j < N; ++j) {
                    double p2 = cnt[j] / TT;
                    double p11 = M[i][j] / TT;
                    double phi = 0.0;
                    if (p1 * (1 - p1) * p2 * (1 - p2) > 1e-12)
                        phi = p11 / std::sqrt(p1 * p2);
                    sum += phi; pairs += 1.0;
                }
            }
            return pairs ? sum / pairs : 0.0;
        };

    double avgR = (Rcnt > 0) ? (Rsum / double(Rcnt)) : 0.0;

    return { avgR,
             avgPhi(coBurst, cntBurst, binsIzh),
             avgPhi(coSpike, cntSpike, binsIzh) };
}


// main
int main()
{
    //parameters
    const int    N = 200;
    const double T = 200000.0, dt = 0.01, T_tr = 100000.0;
    const double avg_deg = 8.0;
    const double mu = 1.0; // unused
    const double w0_mean = 0.007, w0_rng = 0.15 * w0_mean;

    const double K_sl0 = 0.001;
    const double K_izh0 = 20.0;

    const double a = 0.02, b = 0.32, d = 5.0;

    const bool   save = true;
    const int    iter_begin = 0, iter_end = 600;

    omp_set_num_threads(18);

    std::atomic<bool> error_flag{ false };

#pragma omp parallel for schedule(static) shared(error_flag)
    for (int iter = iter_begin; iter < iter_end; ++iter)
    {
        // block size 100, scale drops by 10% per block
        int    group = (iter - iter_begin) / 100;
        double scale = std::max(0.0, 1.0 - 0.1 * (group));
        double scale1 = std::max(0.0, 1.0 - 0.1 * (group));

        double K_sl = K_sl0 * scale1;
        double K_izh = K_izh0 * scale;

        std::string dir = "results/output_iteration_" + std::to_string(iter);
        if (!create_directory(dir)) {
            error_flag.store(true, std::memory_order_relaxed);
#pragma omp critical
            std::cerr << "cannot create " << dir << '\n';
            continue;
        }

        auto R = simulate(N, T, dt, T_tr, avg_deg,
            mu, w0_mean, w0_rng, K_sl,
            a, b, d, K_izh,
            save, dir);

#pragma omp critical
        std::cout << "Iter " << iter
            << "  K_sl=" << K_sl
            << "  K_izh=" << K_izh
            << "  SL=" << R.avgSL       // avg Kuramoto order parameter <R>
            << "  Burst=" << R.avgBurst
            << "  Spike=" << R.avgSpike
            << '\n';
    }

    return error_flag.load(std::memory_order_relaxed) ? 1 : 0;
}