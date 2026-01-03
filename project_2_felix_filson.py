import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.metrics import roc_curve as sk_roc_curve, roc_auc_score
from torch.utils.data import TensorDataset, DataLoader
from sklearn.preprocessing import StandardScaler
import copy
from sklearn.model_selection import GroupShuffleSplit
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
    return abs((d + np.pi) % (2*np.pi) - np.pi)

data_events_mc = load_events_csv("pythia.csv")

class JetData:
    def __init__(self, data):
        self.data = data

    # ---------- low-level: build the 10 physics features ----------
    @staticmethod
    def _build_10(j, l):
        ptj, etaj, phij = j.pt, j.eta, j.phi
        ptl, etal, phil = l.pt, l.eta, l.phi
        dphi_val = dphi(phij, phil)                # in [0, π]
        deltaR   = np.sqrt((etaj - etal)**2 + dphi_val**2)
        pt_ratio = ptj / ptl if ptl != 0 else 0.0
        s = ptj + ptl
        pt_asym  = (ptj - ptl) / s if s != 0 else 0.0
        return np.array([
            ptj, etaj, phij,
            ptl, etal, phil,
            dphi_val, deltaR,
            pt_ratio, pt_asym
        ], dtype=float)

    # ---------- MC: features WITH truth (returns shape (n, 11)) ----------
    def features(self):
        rows = []
        for ev in self.data:
            j = ev.leading_jet(); l = ev.leading_lepton()
            if (j is None) or (l is None) or (not hasattr(j, "truth")):
                continue
            x10 = self._build_10(j, l)
            row = np.concatenate([x10, [float(j.truth)]])
            rows.append(row)
        return np.vstack(rows) if rows else np.empty((0, 11), dtype=float)

    
    
    # ---------- DATA: features WITHOUT truth (returns shape (n, 10)) ----------
    def features_data_with_mass(self):
        rows, masses = [], []
        for ev in self.data:
            j = ev.leading_jet(); l = ev.leading_lepton()
            if (j is None) or (l is None):
                continue
            rows.append(self._build_10(j, l))
            masses.append(j.m)
        X = np.vstack(rows) if rows else np.empty((0, 10), dtype=float)
        m = np.asarray(masses, dtype=float) if masses else np.empty((0,), dtype=float)
        return X, m

def feature_vector_mc(data):
    return JetData(data).features()
def feature_vector_data(data):
    return JetData(data).features_data_with_mass()
# -----------------------------
# Data → splits → standardize
# -----------------------------

def make_splits_and_scale(events, rs1=1996, rs2=2025):
    """
    Build 10-D features (no mass/energy), labels, and do a grouped 60/20/20 split by event_id.
    Returns standardized arrays + scaler + event refs + explicit indices.
    """
    rows, gids, ev_refs = [], [], []
    for ev in events:
        j = ev.leading_jet(); l = ev.leading_lepton()
        if (j is None) or (l is None) or (not hasattr(j, "truth")):
            continue
        x10 = JetData._build_10(j, l)              # 10 inputs (no mass/energy)
        rows.append(np.concatenate([x10, [float(j.truth)]]))
        gids.append(ev.event_id)
        ev_refs.append(ev)                          # keep pointer back to the Event for comparison to cut based

    if not rows:
        raise RuntimeError("No selectable events with (jet, lepton, truth) found.")

    Xy      = np.asarray(rows, dtype=np.float32)
    groups  = np.asarray(gids, dtype=np.int64)
    X       = Xy[:, :-1].astype(np.float32)
    y       = Xy[:, -1].astype(np.int64)

    # 60% train, 20% val, 20% test — grouped by event_id
    gss1 = GroupShuffleSplit(n_splits=1, test_size=0.40, random_state=rs1)
    idx_train, idx_hold = next(gss1.split(X, y, groups=groups))
    gss2 = GroupShuffleSplit(n_splits=1, test_size=0.50, random_state=rs2)
    rel_val, rel_test   = next(gss2.split(X[idx_hold], y[idx_hold], groups=groups[idx_hold]))
    idx_val  = idx_hold[rel_val]
    idx_test = idx_hold[rel_test]

    # Standardize with train-only stats
    scaler  = StandardScaler().fit(X[idx_train])
    X_train = scaler.transform(X[idx_train]).astype(np.float32)
    X_val   = scaler.transform(X[idx_val]).astype(np.float32)
    X_test  = scaler.transform(X[idx_test]).astype(np.float32)
    y_train, y_val, y_test = y[idx_train], y[idx_val], y[idx_test]

    return (X_train, X_val, X_test,
            y_train, y_val, y_test,
            scaler, ev_refs, idx_train, idx_val, idx_test)

(X_train, X_val, X_test,
 y_train, y_val, y_test,
 scaler, ev_refs, train_idx, val_idx, test_idx) = make_splits_and_scale(data_events_mc)

# -----------------------------
# Build DataLoaders 
# -----------------------------
def make_loader(Xn, yn, batch=64, shuffle=True):
    ds = TensorDataset(torch.from_numpy(Xn), torch.from_numpy(yn.astype(np.float32)))
    return DataLoader(ds, batch_size=batch, shuffle=shuffle, drop_last=False)

train_loader = make_loader(X_train, y_train, batch=64,   shuffle=True)
val_loader   = make_loader(X_val,   y_val,   batch=1024, shuffle=False)
test_loader  = make_loader(X_test,  y_test,  batch=1024, shuffle=False)

# -----------------------------
# Model
# -----------------------------
in_dim  = 10
h1, h2  = 32, 16
dropout_ratio = 0.1

model = nn.Sequential(
    nn.Linear(in_dim, h1),
    nn.ReLU(),
    nn.Dropout(dropout_ratio),
    nn.Linear(h1, h2),
    nn.ReLU(),
    nn.Linear(h2, 1)
)

print(model)
print("Number of trainable parameters:", sum(p.numel() for p in model.parameters() if p.requires_grad))

# -----------------------------
# Loss (with class weights) & optimizer
# -----------------------------
pos = int((y_train == 1).sum())
neg = int((y_train == 0).sum())
pos_weight = torch.tensor([neg / max(1, pos)], dtype=torch.float32)   #Weight rare positives

loss_function = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
optimizer = optim.AdamW(model.parameters(), lr=1e-4, weight_decay=1e-6)

# -----------------------------
# Eval helpers (use loaders)
# -----------------------------
def eval_val_loss(model, loader, criterion):
    model.eval()
    tot, n = 0.0, 0
    with torch.no_grad():
        for xb, yb in loader:
            logits = model(xb)
            loss = criterion(logits, yb.unsqueeze(1))
            tot += loss.item() * xb.size(0)   # sample-weighted
            n += xb.size(0)
    model.train()
    return tot / max(n, 1)

def eval_auc(model, loader):
    model.eval()
    ys, ps = [], []
    with torch.no_grad():
        for xb, yb in loader:
            p = torch.sigmoid(model(xb).squeeze(1))
            ys.append(yb.numpy()); ps.append(p.numpy())
    y = np.concatenate(ys); p = np.concatenate(ps)
    model.train()
    return roc_auc_score(y, p) if len(np.unique(y)) > 1 else float("nan")

def eval_val_acc(model, val_loader):
    model.eval()
    correct, total = 0, 0
    with torch.no_grad():
        for x, y in val_loader:
            probs = torch.sigmoid(model(x)).squeeze(1)
            preds = (probs > 0.5)
            correct += preds.eq(y.bool()).sum().item()
            total += y.size(0)
    model.train()
    return correct / total

# -----------------------------
# Training with early stopping
# -----------------------------
epochs = 4000
patience = 200

train_losses = np.zeros(epochs)
val_losses   = np.zeros(epochs)
train_accs   = np.zeros(epochs)
val_accs     = np.zeros(epochs)
val_aucs     = np.zeros(epochs)

best_val = float("inf")
best_epoch = -1
best_state = None
epochs_since_improve = 0

for epoch in range(epochs):
    running_loss, num_correct, total = 0.0, 0, 0
    for inputs, targets in train_loader:
        optimizer.zero_grad()
        logits = model(inputs)
        loss = loss_function(logits, targets.unsqueeze(1))
        probs = torch.sigmoid(logits).squeeze(1)
        preds = (probs > 0.5)

        loss.backward()
        optimizer.step()

        running_loss += loss.item()
        num_correct += preds.eq(targets.bool()).sum().item()
        total += targets.size(0)

    train_losses[epoch] = running_loss / len(train_loader)
    train_accs[epoch]   = num_correct / total

    val_losses[epoch] = eval_val_loss(model, val_loader, loss_function)
    val_accs[epoch]   = eval_val_acc(model, val_loader)
    val_aucs[epoch]   = eval_auc(model, val_loader)

    # Early stopping
    improved = val_losses[epoch] < best_val - 1e-6
    if improved:
        best_val = val_losses[epoch]
        best_epoch = epoch
        best_state = copy.deepcopy(model.state_dict())
        epochs_since_improve = 0
    else:
        epochs_since_improve += 1
        if epochs_since_improve >= patience:
            print(f"Early stopping at epoch {epoch+1}; best was {best_epoch+1} "
                  f"with val loss {best_val:.5f}, val acc {val_accs[best_epoch]:.3f}, "
                  f"val AUC {val_aucs[best_epoch]:.3f}")
            break

   

# restore best weights
if best_state is not None:
    model.load_state_dict(best_state)

# --- plot & mark the best epoch ---
fig, (ax1, ax2) = plt.subplots(1,2)
ax1.plot(range(1, epoch+2), train_losses[:epoch+1], label="Train loss")
ax1.plot(range(1, epoch+2), val_losses[:epoch+1],   label="Val loss")
ax1.axvline(best_epoch+1, linestyle="--", label="Best epoch")
ax1.set_xlabel("Epoch"); ax1.set_ylabel("BCE loss"); ax1.legend()

ax2.plot(range(1, epoch+2), train_accs[:epoch+1], label="Train acc")
ax2.plot(range(1, epoch+2), val_accs[:epoch+1], label="Val acc")
ax2.set_xlabel("Epoch"); ax2.set_ylabel("Accuracy"); ax2.legend()
plt.show()

print(f"Best epoch: {best_epoch+1} | Val loss: {best_val:.5f} | "
      f"Val AUC: {val_aucs[best_epoch]:.3f}")



model.eval()
with torch.no_grad():
    xb = torch.from_numpy(X_test).float().to(next(model.parameters()).device)
    logits = model(xb)
    probs = torch.sigmoid(logits).cpu().numpy().squeeze()

y_true = y_test.astype(int)

print("mean P(signal) | signal:", probs[y_true==1].mean(),
      "| background:", probs[y_true==0].mean())

# Ensure labels are 0/1 integers
y_true = np.asarray(y_true).astype(int)


# Plot score distributions by label
bins = np.linspace(0, 1, 41)
plt.figure(figsize=(5.5,4))
plt.hist(probs[y_true==1], bins=bins, density=True, alpha=0.6, label="Signal")
plt.hist(probs[y_true==0], bins=bins, density=True, alpha=0.6, label="Background")
plt.xlabel("Predicted probability P(signal)")
plt.ylabel("Density")
plt.title("Distribution of P(NN) by class")
plt.legend()
plt.tight_layout()
plt.show()




# ROC + AUC with scikit-learn
fpr, tpr, thr = sk_roc_curve(y_true, probs)       
auc_val = roc_auc_score(y_true, probs)

plt.figure()
plt.plot(fpr, tpr, label=f"AUC = {auc_val:.3f}")
plt.plot([0, 1], [0, 1], "--", lw=1)
plt.xlabel("False Positive Rate (FPR)")
plt.ylabel("True Positive Rate (TPR)")
plt.title("ROC")
plt.legend(); plt.grid(); plt.show()

# Baseline class prevalence on the test split
S0 = (y_true == 1).sum()
B0 = (y_true == 0).sum()
prevalence = S0 / (S0 + B0)

# Sweep thresholds
ts = np.linspace(0, 1, 501)
purity = np.zeros_like(ts)
eps_S  = np.zeros_like(ts)
eps_B  = np.zeros_like(ts)

for i, t in enumerate(ts):
    sel = probs >= t
    S = np.sum(sel & (y_true == 1))
    B = np.sum(sel & (y_true == 0))

    purity[i] = (S / (S + B)) if (S + B) > 0 else np.nan
    eps_S[i]  = S / S0 if S0 > 0 else np.nan
    eps_B[i]  = B / B0 if B0 > 0 else np.nan

# Choose t* = argmax purity s.t. ε_S ≥ 0.30
mask = eps_S >= 0.50
if np.any(mask):
    i_star = np.nanargmax(purity[mask])
    i_star = np.where(mask)[0][i_star]
    t_star = ts[i_star]
    pur_star = purity[i_star]
    epsS_star = eps_S[i_star]
    epsB_star = eps_B[i_star]
else:
    t_star = np.nan; pur_star = np.nan; epsS_star = np.nan; epsB_star = np.nan

# Plot purity vs threshold with baseline and t*
plt.figure()
plt.plot(ts, purity, label="Purity vs threshold")
plt.axhline(prevalence, linestyle="--", label=f"Class prevalence = {prevalence:.3f}")
if not np.isnan(t_star):
    plt.axvline(t_star, linestyle=":", label=f"t* = {t_star:.3f}")
plt.xlabel("Threshold t")
plt.ylabel("Purity (precision)")
plt.title("Purity vs threshold (test split)")
plt.legend()
plt.grid(True)
plt.show()


if not np.isnan(t_star):
    print(f"Working point t*: {t_star:.3f}")
    print(f"  Purity(t*):    {pur_star:.3f}")
    print(f"  ε_S(t*) (TPR): {epsS_star:.3f}  (constraint ≥ 0.50)")
    print(f"  ε_B(t*):       {epsB_star:.3f}")
else:
    print("No threshold satisfies ε_S ≥ 0.50 on this test split.")



#Problems C1-3:

events_atlas = load_events_csv("jets.csv")
Xd_unscaled, masses_all = feature_vector_data(events_atlas)
Xd = scaler.transform(Xd_unscaled).astype(np.float32)

model.eval()
with torch.no_grad():
    logits_d = model(torch.from_numpy(Xd))
    scores_d = torch.sigmoid(logits_d).cpu().numpy().ravel()  # s in (0,1)
    
# Use the t* already computed on the MC test split
t_star = float(t_star)  

sel_nn = scores_d >= t_star
masses_after_nn = masses_all[sel_nn]

# Plot density-normalized mass spectra (before vs after NN selection)
plt.figure(figsize=(5.8,4.2))
bins, rng = 40, (60, 140)
plt.hist(masses_all,      bins=bins, range=rng, density=True, histtype="step", label="All data")
plt.hist(masses_after_nn, bins=bins, range=rng, density=True, histtype="step", label=f"NN: s ≥ t*={t_star:.3f}")
plt.xlabel("Leading large-R jet mass [GeV]")
plt.ylabel("Density")
plt.title("Data: mass before/after NN selection")
plt.legend(); plt.grid(True); plt.tight_layout(); plt.show()

# Example cuts (reuse what you had)
min_pt_j  = 250.0
min_pt_l  = 50.0
min_dphi  = 2.4
eta_j_max = 2.0

def pass_cuts(e: Event):
    j = e.leading_jet(); l = e.leading_lepton()
    if (j is None) or (l is None):
        return False
    return (
        (j.pt >= min_pt_j) and
        (l.pt >= min_pt_l) and
        (dphi(j.phi, l.phi) >= min_dphi) and
        (abs(j.eta) <= eta_j_max)
    )

# Build the cut-selected mass spectrum on DATA
masses_after_cuts = []
for ev in events_atlas:
    j = ev.leading_jet(); l = ev.leading_lepton()
    if (j is None) or (l is None): 
        continue
    if pass_cuts(ev):
        masses_after_cuts.append(j.m)
masses_after_cuts = np.asarray(masses_after_cuts, dtype=np.float32)

# Overlay the three spectra on DATA
plt.figure(figsize=(6.2,4.4))
plt.hist(masses_all,           bins=bins, range=rng, density=True, histtype="step", label="All data")
plt.hist(masses_after_cuts,    bins=bins, range=rng, density=True, histtype="step", label="Cut-based")
plt.hist(masses_after_nn,      bins=bins, range=rng, density=True, histtype="step", label=f"NN (s ≥ t*={t_star:.3f})")
plt.xlabel("Leading large-R jet mass [GeV]")
plt.ylabel("Density")
plt.title("Data: mass spectra comparison (all vs cuts vs NN)")
plt.legend(); plt.grid(True); plt.tight_layout(); plt.show()

# --- Purity for CUTS on MC TEST ---
S0 = (y_test == 1).sum()
B0 = (y_test == 0).sum()

# Recompute S,B after applying cuts to MC TEST events
mc_test_events = data_events_mc[int(.8*len(data_events_mc)):]  # same split logic
S_cuts = B_cuts = 0
for ridx in test_idx:
    ev = ev_refs[ridx]
    j = ev.leading_jet(); l = ev.leading_lepton()
    if (j is None) or (l is None):
        continue
    if pass_cuts(ev):                 # same baseline cuts as before
        if j.truth: S_cuts += 1
        else:       B_cuts += 1

purity_cuts = S_cuts / (S_cuts + B_cuts) if (S_cuts + B_cuts) > 0 else float("nan")

# --- Purity for NN(t*) on MC TEST ---
# You already have y_true (y_test) and probs (scores on X_test)
sel_star = probs >= t_star
S_nn = np.sum(sel_star & (y_true == 1))
B_nn = np.sum(sel_star & (y_true == 0))
purity_nn = S_nn / (S_nn + B_nn) if (S_nn + B_nn) > 0 else float("nan")

print("MC test split purities:")
print(f"  CUTS purity     = {purity_cuts:.3f}  (S={S_cuts}, B={B_cuts})")
print(f"  NN(t*) purity   = {purity_nn:.3f}    (S={S_nn}, B={B_nn})")

