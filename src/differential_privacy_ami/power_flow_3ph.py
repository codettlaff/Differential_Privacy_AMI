import numpy as np
import pandas as pd
import opendssdirect as dss

class RadialNetwork:

    def __init__(self, name, dss_filepath='Master.dss'):

        self.name = name
        self.dss_filepath = dss_filepath

        self.nodes = []
        self.V0 = 0.0

        self.P = {} # {i: np.array shape (T, 3)}
        self.Q = {} # {i: np.array shape (T, 3)}
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
        self.V0 = dss.Bus.kVBase() * 1e3

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
        self.T = T

        # Initialize nodes with zero time-series (3-phase)
        nodes = {
            i: {
                "P": np.zeros((T,3)), # Columns = Phases A, B, C
                "Q": np.zeros((T,3))
            }
            for i in bus_map.values()
        }

        # Extract Loads - Full Time-Series
        # Open-DSS applies the same Load Shape to all three phases.
        dss.Loads.First()
        while True:
            bus_full = dss.CktElement.BusNames()[0]
            parts = bus_full.split('.')
            bus = parts[0]
            phases = [int(p) for p in parts[1:]]
            i = bus_map[bus]

            peak_kw = dss.Loads.kW()
            peak_kvar = dss.Loads.kvar()
            shape_name = dss.Loads.Daily()
            dss.LoadShape.Name(shape_name)

            kw = [peak_kw * s for s in dss.LoadShape.PMult()]
            Qmult = dss.LoadShape.QMult() if dss.LoadShape.QMult != [0.0] else dss.LoadShape.PMult()
            kvar = [peak_kvar * s for s in Qmult]

            # Initialize 3-Phase Nodes
            nodes[i]["P"] = np.zeros((T, 3))
            nodes[i]["Q"] = np.zeros((T, 3))

            for phase in phases:
                idx = phase - 1
                nodes[i]["P"][:, idx] = np.array(kw) * 1e3 # kW -> W
                nodes[i]["Q"][:, idx] = np.array(kvar) * 1e3 # kVar -> Var
            if not dss.Loads.Next():
                break

        # Extract Lines
        # Assuming Balanced Three Phase System with No Zero Sequence
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
        self.T = len(nodes[1]["P"])  # Number of Timesteps
        self.P = {i: data["P"] for i, data in nodes.items()}
        self.Q = {i: data["Q"] for i, data in nodes.items()}

        # Tree Structure
        self.children = {i: [] for i in self.nodes}  # Initialize Dict
        self.parent = {}
        self.lines = []

        # Line Parameters
        self.r = {}  # Ohms
        self.x = {}  # Ohms

        for i, j, r_ij, x_ij in edges:
            self.children[i].append(j)
            self.parent[j] = i
            self.lines.append((i, j))
            self.r[(i, j)] = r_ij
            self.x[(i, j)] = x_ij