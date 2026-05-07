import numpy as np
import pandas as pd
import opendssdirect as dss

class RadialNetwork:

    def __init__(self, name, dss_filepath='master.dss', B=5e3, epsilon=1000):

        self.name = name
        self.dss_filepath = dss_filepath

        self.B=5e3
        self.epsilon=1000

        self.nodes = []
        self.P = {}
        self.P_tilde = {}
        self.Q = {}
        self.Q_tilde = {}
        self.children = {}
        self.parent = {}

        self.build_from_dss()

        # True Power Flow Results
        self.p = {}  # {(i,j,t): P_ij(t)} Branch power flow
        self.V = {}  # {(i,t): V_i(t)} Node Voltage Magnitude

        # Noisy Power Flow Results
        self.p_tilde = {}  # {(i,j,t): P_ij(t)} Branch power flow
        self.V_tilde = {}  # {(i,t): V_i(t)} Node Voltage Magnitude

        # Empirical Accuracy Results
        self.e_p = {}
        self.e_p_norm = {}
        self.e_V = {}
        self.e_V_norm = {}
        self.p_acc = {}
        self.V_acc = {}

    def build_from_dss(self):

        dss.Text.Command("Clear")
        dss.Text.Command(f"Compile [{self.dss_filepath}]")

        # Get Base Voltage of the Source
        dss.Text.Command("CalcVoltageBases")
        dss.Vsources.First()
        bus_full = dss.CktElement.BusNames()[0]
        bus = bus_full.split('.')[0]
        dss.Circuit.SetActiveBus(bus)
        self.V0 = dss.Bus.kVBase() * 1e3

        # Map buses to indices
        bus_names = dss.Circuit.AllBusNames()
        bus_map = {name: idx for idx, name in enumerate(bus_names)}

        # Determine number of timesteps from first Loadshape
        T = 0
        dss.Loads.First()
        if dss.Loads.Count() > 0:
            shape_name = dss.Loads.Daily()
            if shape_name:
                dss.LoadShape.Name(shape_name)
                T = dss.LoadShape.Npts()

        self.T = T

        # Initialize nodes with zero time-series
        nodes = {
            i: {"P": [0.0] * T, "Q": [0.0] * T}
            for i in bus_map.values()
        }

        # Extract Loads - Full time-series
        dss.Loads.First()
        while True:
            bus = dss.CktElement.BusNames()[0].split(".")[0]
            i = bus_map[bus]

            load_peak_kw = dss.Loads.kW()
            load_peak_kvar = dss.Loads.kvar()
            shape_name = dss.Loads.Daily()
            dss.LoadShape.Name(shape_name)

            kw = [load_peak_kw * s for s in dss.LoadShape.PMult()]
            kvar = [load_peak_kvar * s for s in dss.LoadShape.PMult()]

            for t in range(T):
                nodes[i]["P"] = [p * 1e3 for p in kw]  # kW to W
                nodes[i]["Q"] = [q * 1e3 for q in kvar]  # kvar to Var

            if not dss.Loads.Next():
                break

        # Extract Lines - Edges
        edges = []

        dss.Lines.First()
        while True:
            bus1 = dss.Lines.Bus1().split(".")[0]
            bus2 = dss.Lines.Bus2().split(".")[0]

            i = bus_map[bus1]
            j = bus_map[bus2]

            length = dss.Lines.Length()
            r = dss.Lines.R1() * length
            x = dss.Lines.X1() * length

            edges.append((i, j, r, x))

            if not dss.Lines.Next():
                break

        self.nodes = list(nodes.keys())
        self.T = len(nodes[1]["P"])  # Number of Timesteps
        self.P = {i: data["P"] for i, data in nodes.items()}  # Copy True Injections
        self.Q = {i: data["Q"] for i, data in nodes.items()}  # Copy True Injections
        self.P_tilde = {i: [0.0] * self.T for i, data in nodes.items()}  # Initialize Noisy Injections
        self.Q_tilde = {i: [0.0] * self.T for i, data in nodes.items()} # Initialize Noisy Injections

        # Tree Structure
        self.children = {i: [] for i in self.nodes}  # Initialize Dict
        self.parent = {}
        self.lines = []

        # Line Parameters
        self.r = {}  # Unit Ohms
        self.x = {}  # Unit Ohms

        for i, j, r_ij, x_ij in edges:
            self.children[i].append(j)
            self.parent[j] = i
            self.lines.append((i, j))
            self.r[(i, j)] = r_ij
            self.x[(i, j)] = x_ij

    def export_to_dss(self, tilde=False):
        with open(self.dss_filepath, 'w') as f:

            # Circuit Definition
            f.write(f"Clear\n")
            f.write(f"New Circuit.{self.name} basekv={self.V0/1e3} pu=1.0\n")
            f.write(f"Edit Vsource.Source bus1=bus0\n") # Make sure root node is index 0.

            # Lines
            for (i, j) in self.lines:
                r = self.r[(i, j)]
                x = self.x[(i, j)]
                f.write(
                    f"New Line.L_{i}_{j} "
                    f"bus1=bus{i} bus2=bus{j} "
                    f"r1={r} x1={x} r0={r} x0={x} "
                    f"length=1 units=km\n"  # Ohms per Unit Length
                )

            f.write("\n")

            load_kw = []
            load_kvar = []

            # Load-Shapes
            for i in self.nodes:
                if i == 0:
                    continue

                if tilde:
                    P_series = self.P_tilde[i]
                    Q_series = self.Q_tilde[i]
                else:
                    P_series = self.P[i]
                    Q_series = self.Q[i]

                max_kw = np.max(P_series) / 1e3
                load_kw.append(max_kw)
                P_series_scaled = P_series / np.max(P_series) if np.max(P_series) != 0 else np.zeros_like(P_series)

                max_kvar = np.max(Q_series) / 1e3
                load_kvar.append(max_kvar)
                Q_series_scaled = Q_series / np.max(Q_series) if np.max(Q_series) != 0 else np.zeros_like(Q_series)

                P_mult_str = " ".join(str(p) for p in P_series_scaled)
                Q_mult_str = " ".join(str(q) for q in Q_series_scaled)

                f.write(
                    f"New LoadShape.LS_{i} "
                    f"npts={self.T} "
                    f"interval=0.000833 " # For 3s Resolution Data
                    f"Pmult=({P_mult_str})\n"
                    f"Qmult=({Q_mult_str})\n"
                )

            f.write("\n")

            # Loads
            for i in self.nodes:
                if i == 0: continue

                f.write(
                    f"New Load.Load_{i} "
                    f"bus1=bus{i} "
                    f"phases=1 "
                    f"conn=wye "
                    f"model=1 "
                    f"kV={self.V0/1e3} "
                    f"kW={load_kw[i-1]} "
                    f'kvar={load_kvar[i-1]} '
                    f"Daily=LS_{i}\n"
                )

            f.write("\n")

            # Simulation Setup
            f.write(f"Set mode=Daily\n")
            f.write(f"Set number={self.T}\n")
            f.write(f"Set stepsize=3s\n") # Adjust if needed (should equal time resolution of load data).
            f.write(f"\nSolve\n")

    # Set of all nodes along path from root to node
    def C(self, i):
        path = []
        current = i

        # Walk to root using parent pointers
        while True:
            path.append(current)
            if current == 0:
                break
            current = self.parent[current]

        path.reverse()
        return path

    # Set of all nodes downstream of node i
    def D(self, i):
        stack = [i]
        downstream = []
        while stack:
            node = stack.pop()
            downstream.append(node)
            stack.extend(self.children.get(node, []))
        return downstream

    # Set of all lines along path from root to node
    def L(self, i):
        path = self.C(i)
        return [(path[k], path[k + 1]) for k in range(len(path) - 1)]

    def distance_to_root(self, i):
        """Returns number of edges from node i to root (node 0)."""
        dist = 0
        current = i
        while current != 0:
            current = self.parent[current]
            dist += 1
        return dist

    def lin_dist_flow(self, tilde=False):

        if tilde:
            P_src = self.P_tilde
            Q_src = self.Q_tilde
        else:
            P_src = self.P
            Q_src = self.Q

        v_drop = {} # |V_j|^2 - |V_i|^2
        v_dst = {}
        p_dst = {}
        q_dst = {}

        for t in range(self.T):

            for (i,j) in self.lines:

                p_ij = sum(P_src[h][t] for h in self.D(j))
                p_dst[(i, j, t)] = p_ij

                q_ij = sum(Q_src[h][t] for h in self.D(i))
                q_dst[(i,j,t)] = q_ij

                r_ij = self.r[(i, j)]
                x_ij = self.x[(i, j)]

                v_drop[(i, j, t)] = - 2 * (r_ij * p_ij + x_ij * q_ij) # |V_j|^2 - |V_i|^2

            for i in self.nodes:

                v_dst[(i,t)] = self.V0**2 - sum(v_drop[(h, k, t)] for h,k in self.L(i))

        if tilde:
            self.V_tilde = {k: np.sqrt(v) for k, v in v_dst.items()}
            self.p_tilde = p_dst
            self.q_tilde = q_dst
        else:
            self.V = {k: np.sqrt(v) for k, v in v_dst.items()}
            self.p = p_dst
            self.q = q_dst

    def solve_dss(self, tilde=False):

        V_dst = {}
        p_dst = {}
        q_dst = {}

        self.export_to_dss(tilde=tilde)

        # Compile
        dss.Text.Command("Clear")
        dss.Text.Command(f"compile [{self.dss_filepath}]")

        # Root Node Monitor
        (i,j) = self.lines[0]
        dss.Text.Command(
            f"New Monitor.V_root element=Line.L_{i}_{j} mode=0 terminal=1" # Current Voltage Mode # Terminal 0 - From Bus
        )

        # Add Monitors
        for (i,j) in self.lines:
            dss.Text.Command(
                f"New Monitor.P_{i}_{j} element=Line.L_{i}_{j} mode=1 terminal=2"  # Power Mode # Terminal 1 - To Bus
            )
            dss.Text.Command(
                f"New Monitor.V_{i}_{j} element=Line.L_{i}_{j} mode=0 terminal=2"  # Voltage Current Mode # Terminal 1 - To Bus
            )

        dss.Text.Command("Solve")

        # Root Node Monitor Data
        dss.Monitors.Name("V_root")
        v_data = dss.Monitors.Channel(1)
        for t, v in enumerate(v_data):
            V_dst[(0,t)] = v

        # Monitor Data
        for (i,j) in self.lines:

            dss.Monitors.Name(f"P_{i}_{j}")
            p_data = dss.Monitors.Channel(1)
            q_data = dss.Monitors.Channel(2)
            for t, p in enumerate(p_data):
                p_dst[(i, j, t)] = p
            for t, q in enumerate(q_data):
                q_dst[(i, j, t)] = - q

            dss.Monitors.Name(f"V_{i}_{j}")
            V_data = dss.Monitors.Channel(1)
            for t, v in enumerate(V_data):
                V_dst[(j, t)] = v # Using To Node

        if tilde:
            self.V_tilde = V_dst
            self.p_tilde = p_dst
            self.q_tilde = q_dst
        else:
            self.V = V_dst
            self.p = p_dst
            self.q = q_dst

    def power_flow_results(self, t=0, return_results=False, show=False, csv_folderpath=None, tilde=False):

        if tilde:
            V_src = self.V_tilde
            P_src = self.P_tilde
            Q_src = self.Q_tilde
            p_src = self.p_tilde
            q_src = self.q_tilde
        else:
            V_src = self.V_tilde
            P_src = self.P_tilde
            Q_src = self.Q_tilde
            p_src = self.p_tilde
            q_src = self.q_tilde

        # Node Table
        node_data = []
        for i in self.nodes:
            node_data.append({
                "node": i,
                "V": V_src[(i, t)],
                "P": P_src[i][t],
                "Q": Q_src[i][t],
            })
        df_nodes = pd.DataFrame(node_data).sort_values(by="node")

        # Line Table
        line_data = []
        for (i, j) in self.lines:
            line_data.append({
                "from": i,
                "to": j,
                "r": self.r[(i, j)],
                "x": self.x[(i, j)],
                "p_flow": p_src[(i, j, t)],
                "q_flow": q_src[(i, j, t)]
            })
        df_lines = pd.DataFrame(line_data).sort_values(by=["from", "to"])

        # Display
        if show:
            print(f"\n=== NODE STATES (t={t}) ===")
            print(df_nodes.to_string(index=False))
            print(f"\n=== LINE STATES (t={t}) ===")
            print(df_lines.to_string(index=False))

        # Save to CSV
        if csv_folderpath:
            nodes_csv_filepath = csv_folderpath + f"{self.name}_nodes.csv"
            lines_csv_filepath = csv_folderpath + f"{self.name}_lines.csv"
            df_nodes.to_csv(nodes_csv_filepath, index=False)
            df_lines.to_csv(lines_csv_filepath, index=False)

        if return_results: return df_nodes, df_lines

    def empirical_accuracy(self):

        # --- Elementwise errors ---
        self.e_p = {k: np.abs(self.p_tilde[k] - self.p[k]) for k in self.p}
        self.e_V = {k: np.abs(self.V_tilde[k] - self.V[k]) for k in self.V}

        # --- Normalized errors ---
        self.e_p_norm = {
            k: (np.zeros_like(self.e_p[k]) if np.max(self.p[k]) == 0 else self.e_p[k] / np.max(self.p[k]))
            for k in self.e_p
        }

        self.e_V_norm = {
            k: (np.zeros_like(self.e_V[k]) if np.max(self.V[k]) == 0 else self.e_V[k] / np.max(self.V[k]))
            for k in self.e_V
        }

        # --- System-wide accuracy ---
        total_e_p = sum(self.e_p.values())
        total_p = sum(self.p.values())
        p_acc = 0 if total_p == 0 else 1 - (total_e_p / (2 * total_p))

        total_e_V = sum(self.e_V.values())
        total_V = sum(self.V.values())
        V_acc = 1 - (total_e_V / (2 * total_V))

        # --- Per-line power accuracy ---
        p_acc_line = {}
        for (i, j) in self.lines:
            num = sum(self.e_p.get((i, j, t), 0.0) for t in range(self.T))
            den = sum(self.p.get((i, j, t), 0.0) for t in range(self.T))
            p_acc_line[(i, j)] = 1 - (num / (2 * den)) if den != 0 else np.nan

        # --- Per-node voltage accuracy ---
        V_acc_node = {}
        for i in self.nodes:
            num = sum(self.e_V.get((i, t), 0.0) for t in range(self.T))
            den = sum(self.V.get((i, t), 0.0) for t in range(self.T))
            V_acc_node[i] = 1 - (num / (2 * den)) if den != 0 else np.nan

        return p_acc_line, V_acc_node, p_acc, V_acc,

    def theoretical_accuracy(self, B, epsilon):

        sigma_p = {}
        for t in range(self.T):
            for (i, j) in self.lines:
                K = len(self.D(j))
                sigma_p[(i, j, t)] = sigma_p.get((i, j, t), 0) + 2 * np.sqrt(2 * K)  * B / epsilon

        sigma_v = {}

        alpha = {}
        for j in self.nodes:
            for (h, k) in self.L(j):
                for n in self.D(k):
                    alpha[(j, n)] = alpha.get((j, n), 0) + self.r[(h, k)]

        # Each pair (j, n) gives the total resistance of lines on the path to j that are also upstream of n
        for j in self.nodes:
            total = 0.0
            for n in self.nodes:
                total += alpha.get((j, n), 0.0)**2
            sigma_v[(j, t)] = ((4 * np.sqrt(2) * B) / epsilon) * np.sqrt(total)

        p_acc_lower_bound = 1 - (sum(sigma_p.values()) / (2 * sum(self.p.values())))
        v_acc_lower_bound = 1 - (sum(sigma_v.values()) / (2 * sum(v ** 2 for v in self.V.values())))

        # --- Per-line accuracy ---
        p_acc_line = {}
        for (i, j) in self.lines:
            num = sum(sigma_p.get((i, j, t), 0.0) for t in range(self.T))
            den = sum(self.p.get((i, j, t), 0.0) for t in range(self.T))
            val = 1 - (num / (2 * den)) if den != 0 else np.nan
            p_acc_line[(i, j)] = max(0.0, min(1.0, val)) # Min Accuracy below 0 means nothing

        # --- Per-node accuracy ---
        v_acc_node = {}
        for j in self.nodes:
            num = sum(sigma_v.get((j, t), 0.0) for t in range(self.T))
            den = sum((self.V.get((j, t), 0.0)) ** 2 for t in range(self.T))

            v_acc_node[j] = 1 - (num / (2 * den)) if den != 0 else 0.0

        return p_acc_line, v_acc_node, p_acc_lower_bound, v_acc_lower_bound
