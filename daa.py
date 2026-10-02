import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt

# Reproducibility
torch.manual_seed(42)
np.random.seed(42)

# =====================================================================
# 1. ARCHITECTURES: HGCN vs. PAIRWISE GCN
# =====================================================================
class HypergraphConv(nn.Module):
    """Sparse Spectral Hypergraph Convolution: Node -> Hyperedge -> Node"""
    def __init__(self, in_features, out_features):
        super(HypergraphConv, self).__init__()
        self.theta = nn.Parameter(torch.FloatTensor(in_features, out_features))
        nn.init.xavier_uniform_(self.theta)

    def forward(self, X, H):
        # Ensure H is represented as a sparse COO tensor (torch.sparse_coo_tensor)
        if not H.is_sparse:
            indices = torch.nonzero(H).t()
            values = H[indices[0], indices[1]]
            H_sparse = torch.sparse_coo_tensor(indices, values, size=H.shape, device=H.device, dtype=H.dtype).coalesce()
        else:
            H_sparse = H.coalesce() if not H.is_coalesced() else H

        N, M = H_sparse.shape
        indices = H_sparse.indices()
        values = H_sparse.values()

        # Compute degrees of nodes (Dv) and hyperedges (De) in O(nnz(H))
        dv = torch.zeros(N, device=H_sparse.device, dtype=H_sparse.dtype)
        dv.scatter_add_(0, indices[0], values)

        de = torch.zeros(M, device=H_sparse.device, dtype=H_sparse.dtype)
        de.scatter_add_(0, indices[1], values)

        # Inverse diagonal scaling factors
        dv_inv_sqrt = torch.pow(torch.clamp(dv, min=1.0), -0.5).unsqueeze(1)  # Shape: [N, 1]
        de_inv = torch.pow(torch.clamp(de, min=1.0), -1.0).unsqueeze(1)       # Shape: [M, 1]

        # Refactored message passing pipeline using sparse matrix operations:
        # Drops time complexity from O(|V|^2) to O(nnz(H))
        # 1. Node feature pre-scaling by Dv^(-1/2)
        X_scaled = dv_inv_sqrt * X
        # 2. Node -> Hyperedge aggregation: H^T @ X_scaled -> O(nnz(H) * F_in)
        Y = torch.sparse.mm(H_sparse.t(), X_scaled)
        # 3. Hyperedge feature scaling by De^(-1)
        Y_scaled = de_inv * Y
        # 4. Hyperedge -> Node aggregation: H @ Y_scaled -> O(nnz(H) * F_in)
        Z = torch.sparse.mm(H_sparse, Y_scaled)
        # 5. Final node output scaling by Dv^(-1/2)
        Z_scaled = dv_inv_sqrt * Z

        # Linear projection and activation
        return torch.relu(Z_scaled @ self.theta)

class PairwiseGCNConv(nn.Module):
    """Standard Pairwise GCN: Requires clique expansion for shared dependencies"""
    def __init__(self, in_features, out_features):
        super(PairwiseGCNConv, self).__init__()
        self.weight = nn.Parameter(torch.FloatTensor(in_features, out_features))
        nn.init.xavier_uniform_(self.weight)

    def forward(self, X, A):
        # Standard symmetric Laplacian normalization D^(-1/2) * (A + I) * D^(-1/2)
        A_hat = A + torch.eye(A.size(0))
        d = torch.sum(A_hat, dim=1)
        d_inv_sqrt = torch.diag(torch.pow(torch.clamp(d, min=1.0), -0.5))
        A_norm = d_inv_sqrt @ A_hat @ d_inv_sqrt
        return torch.relu(A_norm @ X @ self.weight)

class OffloadingPolicy(nn.Module):
    def __init__(self, model_type="HGNN", in_dim=3, hidden_dim=32, num_tiers=3):
        super(OffloadingPolicy, self).__init__()
        self.model_type = model_type
        if model_type == "HGNN":
            self.conv = HypergraphConv(in_dim, hidden_dim)
        else:
            self.conv = PairwiseGCNConv(in_dim, hidden_dim)
        
        self.actor = nn.Sequential(
            nn.Linear(hidden_dim, 16),
            nn.ReLU(),
            nn.Linear(16, num_tiers),
            nn.Softmax(dim=-1)
        )

    def forward(self, X, graph_repr):
        h = self.conv(X, graph_repr)
        return self.actor(h)

# =====================================================================
# 2. MIST-FOG-CLOUD ENVIRONMENT
# =====================================================================
class IndustrialEnvironment:
    def __init__(self):
        # Tier specs: [Processing MIPS, Power in Watts, Uplink Latency in ms]
        self.tiers = {
            0: {"speed": 1200.0, "power": 1.0, "latency": 2.0},    # Mist (Local edge)
            1: {"speed": 4500.0, "power": 3.8, "latency": 20.0},   # Fog (Gateway/On-prem)
            2: {"speed": 16000.0, "power": 14.0, "latency": 75.0}  # Cloud (Remote)
        }

    def generate_workload(self, num_tasks, num_hyperedges):
        # Node features: [CPU req (MI), Payload (MB), Deadline (ms)]
        cpu = np.random.uniform(500, 3500, size=(num_tasks, 1))
        payload = np.random.uniform(2, 40, size=(num_tasks, 1))
        deadline = np.random.uniform(30, 160, size=(num_tasks, 1))
        X = np.hstack([cpu, payload, deadline])

        # Hypergraph Incidence Matrix H: [N, M]
        H = np.zeros((num_tasks, num_hyperedges))
        for i in range(num_tasks):
            edge = np.random.choice(num_hyperedges)
            H[i, edge] = 1.0
            if np.random.rand() > 0.65:
                H[i, np.random.choice(num_hyperedges)] = 1.0

        # Pairwise Clique Adjacency Matrix A: [N, N] (for standard GCN comparison)
        # Any tasks sharing a hyperedge form a pairwise clique: A = H * H^T
        A = np.dot(H, H.T)
        np.fill_diagonal(A, 0)
        A = (A > 0).astype(float)

        return torch.FloatTensor(X), torch.FloatTensor(H), torch.FloatTensor(A)

    def compute_metrics(self, X, actions):
        total_lat = 0.0
        total_eng = 0.0
        violations = 0

        for i, act in enumerate(actions):
            act = int(act)
            cpu, payload, deadline = X[i, 0].item(), X[i, 1].item(), X[i, 2].item()
            spec = self.tiers[act]

            exec_time = (cpu / spec["speed"]) * 1000.0  # ms
            comm_time = spec["latency"] * (payload / 10.0)
            latency = exec_time + comm_time
            energy = (exec_time / 1000.0) * spec["power"]

            if latency > deadline:
                violations += 1
            total_lat += latency
            total_eng += energy

        avg_lat = total_lat / len(actions)
        reward = -(0.01 * total_lat + 0.5 * total_eng + 2.0 * violations)
        return avg_lat, total_eng, reward

    def run_greedy_baseline(self, X):
        """Greedy heuristic: place on Mist; if deadline breached, place on Fog; then Cloud."""
        actions = []
        for i in range(len(X)):
            cpu, payload, deadline = X[i, 0].item(), X[i, 1].item(), X[i, 2].item()
            assigned = 2 # default Cloud
            for tier in [0, 1]:
                spec = self.tiers[tier]
                lat = (cpu / spec["speed"]) * 1000.0 + spec["latency"] * (payload / 10.0)
                if lat <= deadline:
                    assigned = tier
                    break
            actions.append(assigned)
        return actions

# =====================================================================
# 3. TRAINING & BENCHMARKING ENGINE
# =====================================================================
def train_policy(model, env, graph_type, epochs=120):
    optimizer = optim.Adam(model.parameters(), lr=0.005)
    for _ in range(epochs):
        X, H, A = env.generate_workload(num_tasks=20, num_hyperedges=5)
        graph_repr = H if graph_type == "HGNN" else A
        probs = model(X, graph_repr)
        dist = torch.distributions.Categorical(probs)
        actions = dist.sample()
        
        _, _, reward = env.compute_metrics(X, actions)
        loss = -(dist.log_prob(actions).mean() * reward)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

def evaluate_benchmarks():
    print("[1/3] Initializing environment and training HGNN and GCN agents...")
    env = IndustrialEnvironment()
    hgnn = OffloadingPolicy("HGNN")
    gcn = OffloadingPolicy("GCN")

    train_policy(hgnn, env, "HGNN")
    train_policy(gcn, env, "GCN")

    task_sizes = [10, 20, 30, 40, 50]
    results = {
        "HGNN": {"latency": [], "energy": [], "edges": []},
        "GCN": {"latency": [], "energy": [], "edges": []},
        "Greedy": {"latency": [], "energy": []}
    }

    print("[2/3] Running comparative sweeps across task densities (N=10 to 50)...")
    for N in task_sizes:
        num_hedges = max(3, N // 4)
        X, H, A = env.generate_workload(N, num_hedges)

        # 1. HGNN Evaluation
        with torch.no_grad():
            act_hgnn = torch.argmax(hgnn(X, H), dim=-1)
        lat_h, eng_h, _ = env.compute_metrics(X, act_hgnn)
        results["HGNN"]["latency"].append(lat_h)
        results["HGNN"]["energy"].append(eng_h)
        results["HGNN"]["edges"].append(int(torch.sum(H > 0).item())) # Non-zero incidence entries

        # 2. Pairwise GCN Evaluation
        with torch.no_grad():
            act_gcn = torch.argmax(gcn(X, A), dim=-1)
        lat_g, eng_g, _ = env.compute_metrics(X, act_gcn)
        results["GCN"]["latency"].append(lat_g)
        results["GCN"]["energy"].append(eng_g)
        results["GCN"]["edges"].append(int(torch.sum(A > 0).item())) # Pairwise clique edges

        # 3. Greedy Baseline
        act_greedy = env.run_greedy_baseline(X)
        lat_gr, eng_gr, _ = env.compute_metrics(X, act_greedy)
        results["Greedy"]["latency"].append(lat_gr)
        results["Greedy"]["energy"].append(eng_gr)

    print("[3/3] Generating and exporting publication-quality DAA comparison plots...")
    generate_plots(task_sizes, results)

def generate_plots(task_sizes, results):
    plt.style.use('seaborn-v0_8-whitegrid' if 'seaborn-v0_8-whitegrid' in plt.style.available else 'default')

    # Plot 1: Average Latency
    plt.figure(figsize=(7, 4.5))
    plt.plot(task_sizes, results["HGNN"]["latency"], marker='o', linewidth=2, label="HGNN-DRL (Proposed)")
    plt.plot(task_sizes, results["GCN"]["latency"], marker='s', linewidth=2, linestyle='--', label="Pairwise GCN")
    plt.plot(task_sizes, results["Greedy"]["latency"], marker='^', linewidth=2, linestyle=':', label="Greedy Heuristic")
    plt.title("DAA Benchmark: Average Task Latency vs. Workload Scale", fontsize=12, fontweight='bold')
    plt.xlabel("Number of Dependent Tasks (N)", fontsize=11)
    plt.ylabel("Average Turnaround Latency (ms)", fontsize=11)
    plt.legend()
    plt.tight_layout()
    plt.savefig("plot_latency_comparison.png", dpi=300)
    plt.close()

    # Plot 2: Total Energy Consumption
    plt.figure(figsize=(7, 4.5))
    plt.plot(task_sizes, results["HGNN"]["energy"], marker='o', color='forestgreen', linewidth=2, label="HGNN-DRL")
    plt.plot(task_sizes, results["GCN"]["energy"], marker='s', color='darkorange', linewidth=2, linestyle='--', label="Pairwise GCN")
    plt.plot(task_sizes, results["Greedy"]["energy"], marker='^', color='crimson', linewidth=2, linestyle=':', label="Greedy Heuristic")
    plt.title("DAA Benchmark: Total Energy Consumption vs. Workload Scale", fontsize=12, fontweight='bold')
    plt.xlabel("Number of Dependent Tasks (N)", fontsize=11)
    plt.ylabel("Total Energy Consumption (Joules)", fontsize=11)
    plt.legend()
    plt.tight_layout()
    plt.savefig("plot_energy_comparison.png", dpi=300)
    plt.close()

    # Plot 3: Space Complexity (Hyperedges vs Pairwise Edges)
    plt.figure(figsize=(7, 4.5))
    plt.bar([n - 1 for n in task_sizes], results["HGNN"]["edges"], width=2, label="Hypergraph Incidence Elements (nnz(H))", color='royalblue')
    plt.bar([n + 1 for n in task_sizes], results["GCN"]["edges"], width=2, label="Pairwise Clique Edges (|E|)", color='salmon')
    plt.title("DAA Space Complexity: Edge Representation Overhead", fontsize=12, fontweight='bold')
    plt.xlabel("Number of Dependent Tasks (N)", fontsize=11)
    plt.ylabel("Number of Non-Zero Graph Entries", fontsize=11)
    plt.legend()
    plt.tight_layout()
    plt.savefig("plot_space_complexity.png", dpi=300)
    plt.close()

    print("\n Success! Generated 3 benchmark plots in your current folder:")
    print("  -> plot_latency_comparison.png")
    print("  -> plot_energy_comparison.png")
    print("  -> plot_space_complexity.png")

if __name__ == "__main__":
    evaluate_benchmarks()