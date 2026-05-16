import numpy as np
import pandas as pd
import opendssdirect as dss

class RadialNetwork:

    def __init__(self, name, dss_filepath='Master.dss'):

        self.name = name
        self.dss_filepath = dss_filepath

        self.nodes = []
        self.V0 = 0.0
        self.P = {}
        self.Q = {}
        self.children = {}
        self.parent = {}

        self.build_from_dss()

        # Power Flow Results - Nodes
        self.V = {}

        # Power Flow Results - Branches
        self.p = {}
        self.v = {}
        self.i = {}

    def build_from_dss(self):

        dss.Text.Command("Clear")
        dss.Text.Command(f"Compile [{self.dss_filepath}]")

        # Get Base Voltage of the Source
        dss.Text.Command("CalcVoltageBases")
        dss.Vsources.First()
        bus_full = dss.CktElement.BusNames()[0]
        bus = bus_full.split('.')[0]
        dss.Circuit.SetActiveBus(bus)
        V = dss.Bus.VMagAngle()
        self.V0 = V[0]
        # self.V0 = (dss.Bus.kVBase() / np.sqrt(3)) * 1e3 # Go from line-to-line to single phase

        # Map Buses to Indices
        bus_names = dss.Circuit.AllBusNames()
        bus_map = {name: idx for idx, name in enumerate(bus_names)}

        # Determine number of timesteps from first loadshape
        T = 0
        dss.Loads.First()
        if dss.Loads.Count() > 0:
            shape_name = dss.Loads.Daily()
            if shape_name:
                dss.LoadShape.Name(shape_name)
                T = dss.LoadShape.Npts()
                interval = dss.LoadShape.SInterval() # Interval in Seconds
        self.T = T
        self.interval = interval # Interval in Seconds

        # Initialize nodes with zero time-series
        nodes = {
            i: {"P": [0.0] * T, "Q": [0.0] * T}
            for i in bus_map.values()
        }

        # Extract Loads - Full time-series
        dss.Loads.First()
        while True:
            name = dss.Loads.Name()
            dss.Circuit.SetActiveElement(name)
            bus = dss.CktElement.BusNames()[0].split(".")[0]
            i = bus_map[bus]

            peak_kw = dss.Loads.kW()
            peak_kvar = dss.Loads.kvar()
            shape_name = dss.Loads.Daily()
            dss.LoadShape.Name(shape_name)

            kw = [peak_kw * s for s in dss.LoadShape.PMult()]
            Qmult = dss.LoadShape.QMult() if dss.LoadShape.QMult() != [0.0] else dss.LoadShape.PMult()
            kvar = [peak_kvar * s for s in Qmult] # TODO: Fix this in other class as well

            for t in range(T):
                nodes[i]["P"] = [p * 1e3 for p in kw] # kW to W
                nodes[i]["Q"] = [q * 1e3 for q in kvar] # kVar to Var
            if not dss.Loads.Next():
                break

        # Extract Lines
        edges = []

        dss.Lines.First()
        while True:
            bus1 = dss.Lines.Bus1().split('.')[0]
            bus2 = dss.Lines.Bus2().split('.')[0]
            i = bus_map[bus1]
            j = bus_map[bus2]
            length = dss.Lines.Length()
            r = dss.Lines.R1() * length
            x = dss.Lines.X1() * length
            edges.append((i, j, r, x))
            if not dss.Lines.Next():
                break

        self.nodes = list(nodes.keys())
        self.T = len(nodes[1]["P"]) # Number of Timesteps
        self.P = {i: data["P"] for i, data in nodes.items()}
        self.Q = {i: data["Q"] for i, data in nodes.items()}

        # Tree Structure
        self.children = {i: [] for i in self.nodes} # Initialize Dict
        self.parent = {}
        self.lines = []

        # Line Parameters
        self.r = {} # Ohms
        self.x = {} # Ohms

        for i, j, r_ij, x_ij in edges:
            self.children[i].append(j)
            self.parent[j] = i
            self.lines.append((i,j))
            self.r[(i,j)] = r_ij
            self.x[(i, j)] = x_ij

    def export_to_dss(self):
        with open(self.dss_filepath, "w") as f:
            # Circuit Definition
            f.write("Clear\n")
            f.write(f"New Circuit.{self.name} phases=1 basekv={(self.V0/1e3)*np.sqrt(3)} pu=1.0\n")
            f.write(f"Edit Vsource.Source bus1=bus0\n")  # Make sure root node is index 0.
            f.write("\n")
            # Lines
            for (i, j) in self.lines:
                r = self.r[(i, j)]
                x = self.x[(i, j)]
                f.write(
                    f"New Line.L_{i}_{j} "
                    f"phases=1 "
                    f"bus1=bus{i} bus2=bus{j} "
                    f"r1={r} x1={x} r0={r} x0={x} " # TODO: Is this correct, or should zero-sequence impedance be zero?
                    f"length=1 units=km\n"
                )
            f.write("\n")

            load_kw = {}
            load_kvar = {}

            # Cache Unique Load-Shapes
            shape_map = {}
            shape_counter = 0
            node_to_shape = {}

            # Load-Shapes
            for i in self.nodes:
                if i == 0: continue
                max_p = np.max(self.P[i])
                max_q = np.max(self.Q[i])
                load_kw[i] = max_p / 1e3
                load_kvar[i] = max_q / 1e3
                P_series_scaled = self.P[i] / max_p if max_p != 0 else np.zeros_like(self.P[i])
                Q_series_scaled = self.Q[i] / max_q if max_q != 0 else np.zeros_like(self.Q[i])
                key = (
                    tuple(np.round(P_series_scaled, 6)),
                    tuple(np.round(Q_series_scaled, 6))
                )
                if key not in shape_map:
                    shape_name = f"LS_{shape_counter}"
                    shape_map[key] = shape_name
                    shape_counter += 1
                    P_mult_str = " ".join(str(p) for p in P_series_scaled)
                    Q_mult_str = " ".join(str(q) for q in Q_series_scaled)
                    f.write(
                        f"New LoadShape.{shape_name} "
                        f"npts={self.T} "
                        f"Sinterval={self.interval} "
                        f"Pmult=({P_mult_str})\n"
                        f"Qmult=({Q_mult_str})\n"
                    )
                else: shape_name = shape_map[key]
                node_to_shape[i] = shape_name
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
                    f"kW={load_kw[i]} "
                    f"kVar={load_kvar[i]} "
                    f"Daily={node_to_shape[i]}\n"
                )
            f.write("\n")

            # Simulation Setup
            f.write(f'Set VoltageBases=[{self.V0 / 1e3}]\n')
            f.write("Set mode=Daily\n")
            f.write(f"Set number={self.T}\n")
            f.write(f"Set stepsize={self.interval}\n")

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
        return [(path[k], path[k+1]) for k in range(len(path) - 1)]

    # Set of all lines connected to node i
    def M(self, i):
        return [(u,v) for (u,v) in self.lines if u == i or v == i]

    def distance_to_root(self, i):
        """Returns number of edges from node i to root (node 0)."""
        dist = 0
        current = i
        while current != 0:
            current = self.parent[current]
            dist += 1
        return dist

    def lin_dist_flow(self):

        v_squared_drop = {} # |V_j|^2 - |V_i|^2

        # Node Results
        V = {} # Node Voltage Magnitude

        # Branch Results
        v = {} # Branch Voltage Drop Magnitude
        i_branch = {} # Branch Current Flow Magnitude
        p = {} # Branch Real Power Flow
        q = {} # Branch Reactive Power Flow

        for t in range(self.T):
            for (i,j) in self.lines:
                p_ij = sum(self.P[h][t] for h in self.D(j)) # Branch power is sum of downstream injections.
                p[(i,j,t)] = p_ij
                q_ij = sum(self.Q[h][t] for h in self.D(i)) # Branch power is sum of downstream injections.
                q[(i,j,t)] = q_ij
                r_ij = self.r[(i,j)]
                x_ij = self.x[(i,j)]
                v_squared_drop[(i,j,t)] = 2 * (r_ij * p_ij + x_ij * q_ij) # |V_j|^2 - |V_i|^2
            for i in self.nodes:
                V_squared = self.V0**2 - sum(v_squared_drop[(h,k,t)] for h,k in self.L(i)) # Squared Nodal Voltage is sum of upstream squared voltage drops
                V[(i,t)] = np.sqrt(V_squared) # Nodal Voltage
            for (i,j) in self.lines:
                v[(i,j,t)] = V[(j,t)] - V[(i,t)] # Line Voltage Drops
                i_ij = np.sqrt(p[(i,j,t)]**2 + q[(i,j,t)]**2) / V[(i,t)] # Line Current Flow
                i_branch[(i,j,t)] = i_ij

        V_formatted = {}
        for (i, t), value in V.items():
            if i not in V_formatted:
                V_formatted[i] = []
            V_formatted[i].append(value)

        # convert lists to numpy arrays
        for i in V_formatted:
            V_formatted[i] = np.array(V_formatted[i])

        v_formatted = {}
        for (i,j,t), value in v.items():
            if (i,j) not in v_formatted:
                v_formatted[(i,j)] = []
            v_formatted[(i,j)].append(value)

        for (i,j) in v_formatted:
            v_formatted[(i,j)] = np.array(v_formatted[(i,j)])

        i_formatted = {}
        for (i, j, t), value in i_branch.items():
            if i not in i_formatted:
                i_formatted[(i,j)] = []
            i_formatted[(i,j)].append(value)

        for (i,j) in i_formatted:
            i_formatted[(i,j)] = np.array(i_formatted[(i,j)])

        p_formatted = {}
        for (i,j,t), value in p.items():
            if (i,j) not in p_formatted:
                p_formatted[(i,j)] = []
            p_formatted[(i,j)].append(value)

        for (i,j) in p_formatted:
            p_formatted[(i,j)] = np.array(p_formatted[(i,j)])

        q_formatted = {}
        for (i,j,t), value in q.items():
            if (i,j) not in q_formatted:
                q_formatted[(i,j)] = []
            q_formatted[(i,j)].append(value)

        for (i,j) in q_formatted:
            q_formatted[(i,j)] = np.array(q_formatted[(i,j)])

        self.V = V_formatted
        self.v = v_formatted
        self.i = i_formatted
        self.p = p_formatted
        self.q = q_formatted

    def solve_dss(self):

        # Node Results
        V = {}

        # Branch Results
        v = {}
        i_flow = {}
        p = {}
        q = {}

        self.export_to_dss()

        # Compile
        dss.Text.Command("Clear")
        dss.Text.Command(f"Compile [{self.dss_filepath}]")

        # Root Node Monitor
        (i,j) = self.lines[0]
        dss.Text.Command(
            f"New Monitor.V_root element=Line.L_{i}_{j} mode=0 terminal=1" # Current Voltage Mode From Bus
        )
        dss.Text.Command(
            f"New Monitor.P_root element=Line.L_{i}_{j} mode=1 terminal=1" # Power Mode From Bus
        )

        # Add Monitors
        for (i,j) in self.lines:
            dss.Text.Command(
                f"New Monitor.P_{i}_{j} element=Line.L_{i}_{j} mode=1 terminal=2" # Power Mode To Bus
            )
            dss.Text.Command(
                f"New Monitor.V_{i}_{j} element=Line.L_{i}_{j} mode=0 terminal=2" # Voltage Current Mode To Bus
            )

        dss.Text.Command("CalcVoltageBases")
        dss.Text.Command("Solve")

        # Root Node Monitor Data
        dss.Monitors.Name("V_root")
        V_data = dss.Monitors.Channel(1) # Phase 1 Voltage Magnitude
        V[0] = V_data

        # Monitor Data
        for (i,j) in self.lines:

            dss.Monitors.Name(f"V_{i}_{j}")
            V_data = dss.Monitors.Channel(1)
            V[j] = V_data
            i_data = dss.Monitors.Channel(3)
            i_flow[(i,j)] = i_data

            dss.Monitors.Name(f"P_{i}_{j}")
            s_data = dss.Monitors.Channel(1)
            theta_data = dss.Monitors.Channel(2)
            p_data = s_data * np.cos(theta_data) # elementwise for two np 1d arrays
            q_data = s_data * np.sin(theta_data)
            p[(i,j)] = p_data * 1e3
            q[(i,j)] = q_data * 1e3

        for (i,j) in self.lines:
            v[(i,j)] = (V[i] - V[j])

        self.V = V
        self.v = v
        self.i = i_flow
        self.p = p
        self.q = q

        print('')

    def power_flow_results(self, t=0, return_results=False, show=False, csv_folderpath=None):

        # Node Table
        node_data = []
        for i in self.nodes:
            node_data.append(
                {
                    "node": i,
                    "V": self.V[i][t],
                    "P": self.P[i][t],
                    "Q": self.Q[i][t]
                })
        df_nodes = pd.DataFrame(node_data).sort_values(by="node")

        # Line Table
        line_data = []
        for (i,j) in self.lines:
            line_data.append({
                "from": i,
                "to": j,
                "r": self.r[(i,j)],
                "x": self.x[(i,j)],
                "p_flow": self.p[(i,j)][t],
                "q_flow": self.q[(i,j)][t],
                "i_flow": self.i[(i,j)][t],
                "v_drop": self.v[(i,j)][t],
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

    def compute_error(self, node_results_1, node_results_2, line_results_1, line_results_2):
        def bounded_error(a, b):
            return np.linalg.norm(a - b) / (np.linalg.norm(a - b) + np.linalg.norm(b))

        node_errors = {}
        for key in node_results_1.keys():
            if key in node_results_2.keys():
                node_errors[key] = bounded_error(node_results_1[key], node_results_2[key])

        line_errors = {}
        for key in line_results_1.keys():
            if key in line_results_2.keys():
                line_errors[key] = bounded_error(line_results_1[key], line_results_2[key])

        return node_errors, line_errors