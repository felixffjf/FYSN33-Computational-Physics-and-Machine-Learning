import math
import numpy as np
import matplotlib.pyplot as plt

plt.rcParams["figure.figsize"] = (5, 4)
plt.rcParams["axes.grid"] = True

# -----------------------------
# Basic data structures
# -----------------------------
class Particle:
    def __init__(self, event_id, pid, pt, eta, phi, e, m, truth=False):
        self.event_id = event_id
        self.pid = pid
        self.pt = pt
        self.eta = eta
        self.phi = phi
        self.e = e
        self.m = m
        self.truth = truth  # only meaningful for jets in MC

class Event:
    def __init__(self, eid, particles=None):
        self.event_id = eid
        self.particles = [] if particles is None else particles
    def add(self, p: Particle):
        assert p.event_id == self.event_id
        self.particles.append(p)
    def jets(self):
        return [p for p in self.particles if abs(p.pid) == 90]
    def leptons(self):
        return [p for p in self.particles if abs(p.pid) != 90]
    def leading_jet(self):
        js = self.jets()
        return max(js, key=lambda p: p.pt) if js else None
    def leading_lepton(self):
        ls = self.leptons()
        return max(ls, key=lambda p: p.pt) if ls else None

# -----------------------------
# IO: load CSV (MC or data)
# -----------------------------
def load_events_csv(path):
    events = {}
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"): 
                continue
            toks = line.split(",")
            if len(toks) == 8:
                eid_s, pid_s, pt_s, eta_s, phi_s, e_s, m_s, truth_s = toks
                truth = bool(int(truth_s))
            elif len(toks) == 7:
                eid_s, pid_s, pt_s, eta_s, phi_s, e_s, m_s = toks
                truth = False
            else:
                continue
            eid  = int(eid_s); pid = int(pid_s)
            pt   = float(pt_s); eta = float(eta_s); phi = float(phi_s)
            e    = float(e_s);  m   = float(m_s)
            if eid not in events:
                events[eid] = Event(eid)
            events[eid].add(Particle(eid, pid, pt, eta, phi, e, m, truth))
    return [events[k] for k in sorted(events)]

# -----------------------------
# Helpers
# -----------------------------
def dphi(a, b):
    d = a - b
    return abs((d + math.pi) % (2*math.pi) - math.pi)

# -----------------------------
# Main: cut-based selection and plots
# -----------------------------
if __name__ == "__main__":
    # Files
    DATA_FILE = "jets.csv"
    MC_FILE   = "pythia.csv"

    # Cuts
    min_pt_j  = 250.0 # min pT for jet
    min_pt_l  = 50.0 # min pT for lepton
    min_dphi  = 2.4  # radians
    eta_j_max = 2.0  # max eta for jet

    def pass_cuts(e: Event):
        j = e.leading_jet()
        l = e.leading_lepton()
        if (j is None) or (l is None):
            return False
        return (
            (j.pt >= min_pt_j) and
            (l.pt >= min_pt_l) and
            (dphi(j.phi, l.phi) >= min_dphi) and
            (abs(j.eta) <= eta_j_max)
        )

    # ---- Data: mass spectra before/after cuts
    data_events = load_events_csv(DATA_FILE)

    all_masses = []
    sel_masses_cuts = []
    seen = 0; kept = 0
    for e in data_events:
        j = e.leading_jet(); l = e.leading_lepton()
        if (j is None) or (l is None):
            continue
        seen += 1
        all_masses.append(j.m)
        if pass_cuts(e):
            kept += 1
            sel_masses_cuts.append(j.m)

    print(f"Data (cuts): selected {kept}/{seen} = {100*kept/max(1,seen):.1f}%.")

    bins = 40
    rng  = (60, 140)
    plt.figure(figsize=(5.8,4.2))
    plt.hist(all_masses,      bins=bins, range=rng, density=True, histtype="step", label="All data")
    plt.hist(sel_masses_cuts, bins=bins, range=rng, density=True, histtype="step",
             label=f"Cuts")
    plt.xlabel("Large-R jet mass [GeV]"); plt.ylabel("Density")
    plt.title("Data: jet mass before/after cuts"); plt.legend(); plt.tight_layout()

    # ---- MC: purity and efficiencies for the same cuts
    mc_events = load_events_csv(MC_FILE)

    # Totals before selection (for efficiencies)
    S0 = B0 = 0
    for e in mc_events:
        j = e.leading_jet()
        if j is None: 
            continue
        S0 += int(j.truth)
        B0 += int(not j.truth)

    S = B = 0
    for e in mc_events:
        j = e.leading_jet(); l = e.leading_lepton()
        if (j is None) or (l is None):
            continue
        if pass_cuts(e):
            if j.truth: S += 1
            else:       B += 1

    N_sel = S + B
    purity_mc = (S / N_sel) if N_sel > 0 else float('nan')
    eps_S = S / S0 if S0 > 0 else float('nan')
    eps_B = B / B0 if B0 > 0 else float('nan')
    purity0 = S0 / max(1, (S0 + B0))
    print(f"Baseline purity (no cuts) = {purity0:.3f}  with S0={S0}, B0={B0}")
    print(f"MC (cuts): purity S/(S+B) = {purity_mc:.3f}  "
          f"with S={S}, B={B}, N={N_sel}  |  eps_S={eps_S:.3f}, eps_B={eps_B:.3f}")
    plt.show()

