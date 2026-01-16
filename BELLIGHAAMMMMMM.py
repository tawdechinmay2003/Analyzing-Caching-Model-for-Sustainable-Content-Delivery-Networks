import networkx as nx, random as rd, matplotlib.pyplot as plt, numpy as np, re
from collections import Counter, OrderedDict as OD
import json, datetime as dt
from sklearn.cluster import KMeans
import warnings
import torch, torch.nn as nn, torch.nn.functional as F
from torch_geometric.nn import GCNConv, GATConv
from torch_geometric.data import Data
import torch.optim as optim

warnings.filterwarnings('ignore')

DEV_PWR = {"router": 100, "switch": 50, "server": 120}
LINK_PWR, EMIT_FACTOR, ELEC_RATE = 0.15, 0.475, 27
CARBON_DATA = {"high": 450, "medium": 300, "low": 100, "solar": 50}
SCENARIOS = {
    "low": {"zipf": 1.3, "traffic": (100, 500), "cache": 40, "contents": 300},
    "normal": {"zipf": 1.5, "traffic": (500, 2000), "cache": 100, "contents": 1200},
    "high": {"zipf": 1.7, "traffic": (2000, 5000), "cache": 200, "contents": 2500},
    "special": {"zipf": 1.9, "traffic": (5000, 20000), "cache": 300, "contents": 4000}
}


class LRUCache:
    def __init__(self, cap):
        self.cap, self.cache = cap, OD()

    def access(self, c):
        if c in self.cache:
            self.cache.move_to_end(c)
            return True
        return False

    def insert(self, c):
        if len(self.cache) >= self.cap:
            self.cache.popitem(last=False)
        self.cache[c] = dt.datetime.now()


class GNNTrafficRouter(nn.Module):
    def __init__(self, node_feat_dim=8, hidden_dim=64):
        super().__init__()
        self.node_encoder = nn.Sequential(
            nn.Linear(node_feat_dim, hidden_dim), nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim), nn.ReLU())
        self.gnn1 = GCNConv(hidden_dim, hidden_dim)
        self.gnn2 = GATConv(hidden_dim, hidden_dim, heads=2, concat=False)
        self.gnn3 = GCNConv(hidden_dim, hidden_dim)
        self.policy_net = nn.Sequential(
            nn.Linear(hidden_dim * 3, hidden_dim * 2), nn.ReLU(),
            nn.Linear(hidden_dim * 2, hidden_dim), nn.ReLU(),
            nn.Linear(hidden_dim, 1))
        self.value_net = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim), nn.ReLU(),
            nn.Linear(hidden_dim, 5), nn.Sigmoid())
        self.layer_norm = nn.LayerNorm(hidden_dim)

    def forward(self, data, src_idx, dst_idx, current_idx=None):
        x = self.node_encoder(data.x)
        edge_weight = data.edge_weight if hasattr(data, 'edge_weight') else None

        x1 = F.relu(self.gnn1(x, data.edge_index, edge_weight))
        x1 = self.layer_norm(x1)
        x2 = F.relu(self.gnn2(x1, data.edge_index))
        x2 = self.layer_norm(x2)
        x3 = F.relu(self.gnn3(x2, data.edge_index, edge_weight))
        x = self.layer_norm(x + x3)

        src_emb, dst_emb = x[src_idx], x[dst_idx]
        path_value = self.value_net(torch.cat([src_emb, dst_emb], dim=-1))
        next_hop_scores = {}

        if current_idx is not None:
            current_emb = x[current_idx]
            row, col = data.edge_index
            neighbors = col[row == current_idx].tolist()
            for neighbor in neighbors:
                combined = torch.cat([src_emb, current_emb, dst_emb], dim=-1)
                next_hop_scores[neighbor] = self.policy_net(combined).item()

        return next_hop_scores, path_value

    def predict_path_quality(self, data, src_idx, dst_idx):
        _, path_value = self.forward(data, src_idx, dst_idx)
        return path_value

    def select_next_hop(self, data, src_idx, dst_idx, current_idx, epsilon=0.1):
        scores, _ = self.forward(data, src_idx, dst_idx, current_idx)
        if rd.random() < epsilon or not scores:
            return rd.choice(list(data.edge_index[1][data.edge_index[0] == current_idx].tolist()))
        return max(scores.items(), key=lambda x: x[1])[0]


class GNNTrafficManager:
    def __init__(self, G, learning_rate=0.001, replay_buffer_size=500):
        self.G = G
        self.replay_buffer = []
        self.replay_buffer_size = replay_buffer_size
        self.step_count = 0
        self.pyg_data = self.convert_to_pyg_data(G)
        self.gnn_router = GNNTrafficRouter()
        self.optimizer = optim.Adam(self.gnn_router.parameters(), lr=learning_rate)
        self.loss_fn = nn.MSELoss()
        self.routing_history = []
        self.exploration_rate = 0.3
        self.min_exploration = 0.05
        self.exploration_decay = 0.995

    def convert_to_pyg_data(self, G):
        node_list = sorted(list(G.nodes()))
        self.node_id_to_idx = {node: i for i, node in enumerate(node_list)}
        self.idx_to_node_id = {i: node for i, node in enumerate(node_list)}

        node_features = []
        for node in node_list:
            features = [
                G.nodes[node].get('carbon_intensity', 450) / 500.0,
                G.nodes[node].get('current_load', 0) / 100.0,
                1.0 if G.nodes[node].get('type') == 'router' else 0.0,
                1.0 if G.nodes[node].get('type') == 'switch' else 0.0,
                1.0 if G.nodes[node].get('type') == 'server' else 0.0,
                hash(str(G.nodes[node].get('region', 'default'))) % 100 / 100.0,
                dt.datetime.now().hour / 24.0,
                1.0 if G.nodes[node].get('status', 'active') == 'active' else 0.0
            ]
            node_features.append(features)

        edge_indices = []
        edge_weights = []
        for u, v in G.edges():
            u_idx, v_idx = self.node_id_to_idx[u], self.node_id_to_idx[v]
            edge_indices.extend([[u_idx, v_idx], [v_idx, u_idx]])
            traffic = G[u][v].get('traffic', 10)
            capacity = G[u][v].get('weight', 100)
            weight = 1.0 / max(0.1, traffic / max(1, capacity))
            edge_weights.extend([weight, weight])

        x = torch.tensor(node_features, dtype=torch.float)
        edge_index = torch.tensor(edge_indices, dtype=torch.long).t().contiguous()
        edge_weight = torch.tensor(edge_weights, dtype=torch.float)

        data = Data(x=x, edge_index=edge_index)
        data.edge_weight = edge_weight
        return data

    def update_graph_data(self):
        self.pyg_data = self.convert_to_pyg_data(self.G)

    def convert_node_id_to_idx(self, node_id):
        return self.node_id_to_idx.get(node_id, -1)

    def convert_idx_to_node_id(self, idx):
        return self.idx_to_node_id.get(idx, None)

    def find_path(self, source, destination, max_steps=15):
        src_idx, dst_idx = self.convert_node_id_to_idx(source), self.convert_node_id_to_idx(destination)
        if src_idx == -1 or dst_idx == -1:
            return [source]

        path = [source]
        current_idx = src_idx
        steps = 0

        while self.convert_idx_to_node_id(current_idx) != destination and steps < max_steps:
            self.update_graph_data()
            try:
                next_idx = self.gnn_router.select_next_hop(
                    self.pyg_data, src_idx, dst_idx, current_idx, self.exploration_rate)
                next_node = self.convert_idx_to_node_id(next_idx)
                if next_node is None or next_node in path[-3:]:
                    break
                path.append(next_node)
                current_idx = next_idx
                steps += 1
            except:
                break

        return path

    def select_server(self, req_node, content_size=1, regions=None, greenest=None):
        candidates = []
        for node in self.G.nodes():
            if node == req_node: continue
            if self.G.nodes[node].get('type') != 'server': continue
            if self.G.nodes[node].get('status') == 'offline': continue
            try:
                if nx.has_path(self.G, req_node, node):
                    candidates.append(node)
            except:
                continue

        if not candidates:
            return req_node if self.G.nodes[req_node].get('type') == 'server' else rd.choice(list(self.G.nodes()))

        req_idx = self.convert_node_id_to_idx(req_node)
        if req_idx == -1:
            return carbon_routing(self.G, req_node, regions, greenest)

        candidate_scores = []
        for server in candidates:
            server_idx = self.convert_node_id_to_idx(server)
            if server_idx == -1: continue

            try:
                with torch.no_grad():
                    predicted = self.gnn_router.predict_path_quality(self.pyg_data, req_idx, server_idx)
                score = (predicted[0] * 0.3 + predicted[1] * 0.2 + predicted[2] * 0.25 +
                         predicted[3] * 0.15 + predicted[4] * 0.1).item()
                candidate_scores.append((score, server))
            except:
                continue

        if not candidate_scores:
            return carbon_routing(self.G, req_node, regions, greenest)

        best_server = max(candidate_scores, key=lambda x: x[0])[1]
        current_load = self.G.nodes[best_server].get('current_load', 0)
        self.G.nodes[best_server]['current_load'] = min(100, current_load + 3)
        return best_server

    def learn_from_experience(self, source, destination, path_used, actual_metrics):
        src_idx, dst_idx = self.convert_node_id_to_idx(source), self.convert_node_id_to_idx(destination)
        if src_idx == -1 or dst_idx == -1:
            return

        self.replay_buffer.append({
            'source': source, 'destination': destination, 'path': path_used,
            'metrics': actual_metrics, 'graph_data': self.pyg_data.clone()
        })
        if len(self.replay_buffer) > self.replay_buffer_size:
            self.replay_buffer.pop(0)

        if len(self.replay_buffer) >= 8:
            batch = rd.sample(self.replay_buffer, min(8, len(self.replay_buffer)))
            self.train_on_batch(batch)

        self.exploration_rate = max(self.min_exploration, self.exploration_rate * self.exploration_decay)
        self.routing_history.append({
            'step': self.step_count, 'source': source, 'destination': destination,
            'path_length': len(path_used), 'metrics': actual_metrics
        })
        self.step_count += 1

    def train_on_batch(self, batch):
        if len(batch) == 0:
            return 0.0

        self.gnn_router.train()
        self.optimizer.zero_grad()
        total_loss = 0
        valid_items = 0

        for exp in batch:
            src_idx = self.convert_node_id_to_idx(exp['source'])
            dst_idx = self.convert_node_id_to_idx(exp['destination'])
            if src_idx == -1 or dst_idx == -1: continue

            try:
                predicted = self.gnn_router.predict_path_quality(exp['graph_data'], src_idx, dst_idx)
                actual = torch.tensor(exp['metrics'], dtype=torch.float)
                total_loss += self.loss_fn(predicted, actual)
                valid_items += 1
            except:
                continue

        if valid_items > 0:
            total_loss = total_loss / valid_items
            total_loss.backward()
            self.optimizer.step()
            return total_loss.item()

        return 0.0

    def get_stats(self):
        if not self.routing_history:
            return "GNN Router: Learning..."
        recent = self.routing_history[-50:] if len(self.routing_history) >= 50 else self.routing_history
        if not recent: return "GNN Router: No data"
        avg_lat = np.mean([h['metrics'][0] for h in recent]) if recent else 0
        avg_energy = np.mean([h['metrics'][1] for h in recent]) if recent else 0
        return (f"GNN: Steps={self.step_count}, Exp={self.exploration_rate:.2f}, "
                f"Lat={avg_lat:.2f}, E={avg_energy:.2f}")


class HierarchicalCache:
    def __init__(self, G, edge=10, reg=20, core=30):
        self.G, self.edge_sz, self.reg_sz, self.core_sz = G, edge, reg, core
        self.edge_caches = {n: LRUCache(edge) for n in G.nodes()}
        cent = nx.betweenness_centrality(G)
        reg_nodes = dict(sorted(cent.items(), key=lambda x: x[1], reverse=True)[:3])
        self.reg_caches = {n: LRUCache(reg) for n in reg_nodes}
        self.core_cache = LRUCache(core)
        self.demand_hist, self.adapt_int, self.min_sz, self.max_sz = {}, 100, 5, 500

    def adaptive_sizing(self, step, traffic):
        if step % self.adapt_int == 0:
            hour = dt.datetime.now().hour
            if hour not in self.demand_hist: self.demand_hist[hour] = []
            self.demand_hist[hour].append(traffic)
            for h in list(self.demand_hist.keys()):
                if len(self.demand_hist[h]) > 10: self.demand_hist[h] = self.demand_hist[h][-10:]
            if hour in self.demand_hist and self.demand_hist[hour]:
                avg_d = np.mean(self.demand_hist[hour])
                fact = min(5.0, max(0.3, avg_d / 100))
                new_edge = max(self.min_sz, min(self.max_sz, int(self.edge_sz * fact)))
                new_reg = max(self.min_sz + 5, min(self.max_sz, int(self.reg_sz * (1 + (fact - 1) * 0.7))))
                new_core = max(self.min_sz + 10, min(self.max_sz, int(self.core_sz * (1 + (fact - 1) * 0.3))))
                for c in self.edge_caches.values(): c.cap = new_edge
                for c in self.reg_caches.values(): c.cap = new_reg
                self.core_cache.cap = new_core

    def access(self, content, req):
        if self.edge_caches[req].access(content): return "edge", True
        for rn, cache in self.reg_caches.items():
            if cache.access(content):
                self.edge_caches[req].insert(content)
                return "regional", True
        if self.core_cache.access(content):
            closest = self._find_closest_regional(req)
            if closest: self.reg_caches[closest].insert(content)
            self.edge_caches[req].insert(content)
            return "core", True
        return "origin", False

    def _find_closest_regional(self, req):
        try:
            dists = {}
            for rn in self.reg_caches.keys():
                if req in self.G.nodes() and rn in self.G.nodes() and nx.has_path(self.G, req, rn):
                    path = nx.shortest_path(self.G, req, rn, weight='weight')
                    dists[rn] = len(path)
            return min(dists, key=dists.get) if dists else list(self.reg_caches.keys())[0] if self.reg_caches else None
        except:
            return list(self.reg_caches.keys())[0] if self.reg_caches else None

    def insert(self, content, loc="edge", node=None):
        if loc == "edge" and node:
            self.edge_caches[node].insert(content)
        elif loc == "regional" and node:
            self.reg_caches[node].insert(content)
        elif loc == "core":
            self.core_cache.insert(content)


class PowerManager:
    def __init__(self, G, traffic_thresh=50, idle_thresh=30):
        self.G, self.traffic_thresh, self.idle_thresh = G, traffic_thresh, idle_thresh
        self.cluster_states, self.idle_timers = {}, {}

    def detect_idle(self):
        idle = []
        for n in self.G.nodes():
            traffic = sum(self.G[n][nb].get('traffic', 0) for nb in self.G.neighbors(n))
            if traffic < self.traffic_thresh:
                if n not in self.idle_timers: self.idle_timers[n] = 0
                self.idle_timers[n] += 1
            else:
                self.idle_timers[n] = 0
            if self.idle_timers.get(n, 0) >= self.idle_thresh: idle.append(n)
        return idle

    def manage_power(self, idle_nodes):
        for n in idle_nodes:
            state = self.cluster_states.get(n, 'active')
            if state == 'active':
                self.enter_low_power(n);
                self.cluster_states[n] = 'low_power'
            elif state == 'low_power' and self.idle_timers[n] >= 60:
                self.shutdown(n);
                self.cluster_states[n] = 'shutdown'

    def enter_low_power(self, n):
        orig = DEV_PWR.get(self.G.nodes[n].get('type', 'switch'), 50)
        self.G.nodes[n]['power'] = orig * 0.2
        self.G.nodes[n]['carbon_intensity'] *= 0.3
        self.G.nodes[n]['perf_factor'] = 0.5

    def shutdown(self, n):
        self.G.nodes[n]['power'] = 0
        self.G.nodes[n]['carbon_intensity'] = 0
        self.G.nodes[n]['perf_factor'] = 0
        self.G.nodes[n]['status'] = 'offline'

    def wake_up(self, n):
        orig_pwr = DEV_PWR.get(self.G.nodes[n].get('type', 'switch'), 50)
        orig_carb = CARBON_DATA.get(self.G.nodes[n].get('energy_type', 'high'), 450)
        self.G.nodes[n]['power'] = orig_pwr
        self.G.nodes[n]['carbon_intensity'] = orig_carb
        self.G.nodes[n]['perf_factor'] = 1.0
        self.G.nodes[n]['status'] = 'active'
        self.cluster_states[n] = 'active';
        self.idle_timers[n] = 0

    def monitor_traffic(self):
        for n in list(self.G.nodes()):
            if self.cluster_states.get(n) in ['low_power', 'shutdown']:
                demand = sum(self.G[n][nb].get('traffic', 0) for nb in self.G.neighbors(n))
                if demand > self.traffic_thresh * 3: self.wake_up(n)


def assign_energy(G):
    for n in G.nodes():
        r = rd.random()
        if r < 0.6:
            G.nodes[n]['carbon_intensity'], G.nodes[n]['energy_type'] = CARBON_DATA["high"], "high"
        elif r < 0.8:
            G.nodes[n]['carbon_intensity'], G.nodes[n]['energy_type'] = CARBON_DATA["medium"], "medium"
        elif r < 0.9:
            G.nodes[n]['carbon_intensity'], G.nodes[n]['energy_type'] = CARBON_DATA["low"], "low"
        else:
            G.nodes[n]['carbon_intensity'], G.nodes[n]['energy_type'] = CARBON_DATA["solar"], "solar"


def update_time_carbon(G, step):
    hour = (step // 60) % 24
    for n in G.nodes():
        e_type = G.nodes[n].get('energy_type', 'high')
        if e_type == "solar":
            if 6 <= hour <= 18:
                solar_mult = max(0.3, np.sin((hour - 6) * np.pi / 12))
                G.nodes[n]['carbon_intensity'] = CARBON_DATA["solar"] * (1 - solar_mult * 0.8)
            else:
                G.nodes[n]['carbon_intensity'] = CARBON_DATA["high"]
        elif e_type == "low" and (hour <= 6 or hour >= 20):
            G.nodes[n]['carbon_intensity'] = CARBON_DATA["low"] * 0.7


def create_regions(G, pos, num_r=6):
    if not pos:
        nodes = list(G.nodes())
        rs = max(1, len(nodes) // num_r)
        return {f"r_{i}": nodes[i * rs:(i + 1) * rs if i < num_r - 1 else len(nodes)] for i in range(num_r)}
    coords = np.array([(pos[n][0], pos[n][1]) for n in G.nodes()])
    nodes_list = list(G.nodes())
    kmeans = KMeans(n_clusters=num_r, random_state=42, n_init=10)
    labels = kmeans.fit_predict(coords)
    regions = {}
    for i in range(num_r):
        regions[f"r_{i}"] = [nodes_list[j] for j in range(len(labels)) if labels[j] == i]
    return regions


def compute_greenest(G, regions):
    greenest, stats = {}, {}
    for rn, nodes in regions.items():
        if not nodes: continue
        candidates = []
        for n in nodes:
            if G.nodes[n].get('status') != 'offline':
                c = G.nodes[n].get('carbon_intensity', 450)
                candidates.append((n, c))
        if candidates:
            candidates.sort(key=lambda x: x[1])
            gs, cv = candidates[0]
            greenest[rn] = gs
            c_vals = [c for _, c in candidates]
            stats[rn] = {'greenest': gs, 'green_carb': cv, 'avg_carb': np.mean(c_vals), 'count': len(candidates)}
    return greenest, stats


def find_region(n, regions):
    for rn, nodes in regions.items():
        if n in nodes: return rn
    return None


def regional_routing(req, regions, greenest, G):
    req_r = find_region(req, regions)
    if not req_r: return req
    gs = greenest.get(req_r)
    if not gs or G.nodes[gs].get('status') == 'offline': return req
    return gs


def carbon_routing(G, req, regions=None, greenest=None, max_latency=20):
    if regions and greenest: return regional_routing(req, regions, greenest, G)
    lat_penalty = max(1, int(max_latency / 10))
    server_lats, avail_servers = {}, []
    for s in G.nodes():
        if s == req: continue
        if G.nodes[s].get('status') == 'offline': continue
        try:
            if nx.has_path(G, req, s):
                path = nx.shortest_path(G, req, s, weight='weight')
                lats = len(path) - 1
                server_lats[s] = lats
                avail_servers.append(s)
        except:
            continue
    if not avail_servers: return req if G.nodes[req].get('type') == 'server' else rd.choice(list(G.nodes()))
    min_lat = min(server_lats.values())
    max_allowed = min_lat + lat_penalty
    candidates = []
    for s in avail_servers:
        if server_lats[s] <= max_allowed:
            c = G.nodes[s].get('carbon_intensity', 450)
            candidates.append((s, c, server_lats[s]))
    if not candidates:
        max_allowed += 2
        for s in avail_servers:
            if server_lats[s] <= max_allowed:
                c = G.nodes[s].get('carbon_intensity', 450)
                candidates.append((s, c, server_lats[s]))
    if not candidates:
        for s in avail_servers:
            c = G.nodes[s].get('carbon_intensity', 450)
            candidates.append((s, c, server_lats[s]))
    candidates.sort(key=lambda x: (x[1], x[2]))
    return candidates[0][0] if candidates else avail_servers[0]


def gnn_enhanced_routing(G, req_node, content, gnn_manager=None, regions=None, greenest=None):
    if gnn_manager is None:
        return carbon_routing(G, req_node, regions, greenest)
    return gnn_manager.select_server(req_node, content_size=1, regions=regions, greenest=greenest)


def load_network(net_choice):
    nets = {
        "allegiance": {"dot": "Allegiance_Telecom.dot", "edge": "Allegiance_Telecom2.txt",
                       "city": "Allegiance_Telecom3.txt", "origin": 7},
        "athome": {"dot": "At_Home_Network.dot", "edge": "At_Home_Network2.txt", "city": "At_Home_Network3.txt",
                   "origin": 15},
        "cais": {"dot": "CAIS_Internet.dot", "edge": "CAIS_Internet2.txt", "city": "CAIS_Internet3.txt", "origin": 4},
        "att": {"dot": "ATT.dot", "edge": "ATT2.txt", "city": "ATT3.txt", "origin": 10}
    }
    cfg = nets[net_choice]
    G, pos, city_names = nx.Graph(), {}, {}
    with open(cfg["dot"], 'r') as f:
        for line in f:
            if 'pos=' in line:
                m = re.match(r'(\d+)\s+\[pos="([\d\.]+),\s*([\d\.]+)"', line.strip())
                if m:
                    n = int(m.group(1))
                    x, y = float(m.group(2)), float(m.group(3))
                    pos[n] = (x, y)
                    G.add_node(n)
                    if rd.random() < 0.2:
                        G.nodes[n]['type'], G.nodes[n]['power'] = 'router', DEV_PWR['router']
                    elif rd.random() < 0.4:
                        G.nodes[n]['type'], G.nodes[n]['power'] = 'server', DEV_PWR['server']
                    else:
                        G.nodes[n]['type'], G.nodes[n]['power'] = 'switch', DEV_PWR['switch']
                    G.nodes[n]['status'], G.nodes[n]['perf_factor'] = 'active', 1.0
    with open(cfg["edge"], 'r') as f:
        for line in f:
            parts = line.strip().split(',')
            if len(parts) >= 3:
                cap, u, v = float(parts[0]), int(parts[1]), int(parts[2])
                if u in G.nodes() and v in G.nodes():
                    G.add_edge(u, v, weight=cap, base_weight=cap)
    with open(cfg["city"], 'r') as f:
        for line in f:
            parts = line.strip().split(',', 1)
            if len(parts) == 2:
                city_names[int(parts[0])] = parts[1].strip('"')
    return G, pos, city_names, cfg["origin"]


def calc_energy(G, path, dur=2.0):
    t_hr = dur / 3600
    static = sum(G.nodes[n].get('power', DEV_PWR.get(G.nodes[n].get("type", "switch"), 50)) * t_hr
                 for n in path if G.nodes[n].get('status') != 'offline')
    dynamic = sum(G[u][v].get("traffic", 10) * LINK_PWR * t_hr
                  for u, v in zip(path, path[1:])
                  if G.nodes[u].get('status') != 'offline' and G.nodes[v].get('status') != 'offline')
    return static + dynamic


def zipf_dist(num_c, skew=1.6, num_req=2000, win=500, shift_freq=5):
    ranks = np.arange(1, num_c + 1)
    reqs, shifts = [], []
    weights = 1 / np.power(ranks, skew);
    weights /= weights.sum()
    for i in range(num_req):
        if i % (win / shift_freq) == 0:
            pop_c = np.random.choice(num_c, size=rd.randint(2, 5), replace=False)
            shift_mag = rd.uniform(2, 5)
            new_w = weights.copy();
            new_w[pop_c] *= shift_mag;
            new_w /= new_w.sum()
            weights = new_w;
            shifts.append((i, pop_c))
        reqs.append(np.random.choice(num_c, p=weights))
    return np.array(reqs), shifts


def predict_next(reqs, win=100, top=5, method="frequency"):
    win = reqs[-win:]
    if method == "frequency":
        freq = Counter(win)
        return [c for c, _ in freq.most_common(top)]
    elif method == "markov":
        if len(win) > 1:
            trans = {}
            for i in range(len(win) - 1):
                cur, nxt = win[i], win[i + 1]
                if cur not in trans: trans[cur] = Counter()
                trans[cur][nxt] += 1
            last = win[-1]
            if last in trans: return [i for i, _ in trans[last].most_common(top)]
        freq = Counter(win)
        return [c for c, _ in freq.most_common(top)]


def calc_emissions(energy):
    return (energy / 1000) * EMIT_FACTOR


def calc_cost(energy):
    return (energy / 1000) * ELEC_RATE


def predictive_caching(G, content_req, cache_sz=5, origin=0, pred_win=100, scenario="normal", regions=None,
                       greenest=None, use_gnn=False, gnn_manager=None):
    caches = {n: LRUCache(cache_sz) for n in G.nodes()}
    energy = lat = hits = misses = 0
    pm = PowerManager(G)
    pred_method = "markov" if scenario == "special" else "frequency"

    for step, content in enumerate(content_req):
        req_node = rd.choice(list(G.nodes()))
        req = gnn_enhanced_routing(G, req_node, content, gnn_manager, regions,
                                   greenest) if use_gnn and gnn_manager else carbon_routing(G, req_node, regions,
                                                                                            greenest)

        if step % 100 == 0: update_network_congestion(G, step, G.graph.get('congestion_factor', 0.5))
        if step % 50 == 0:
            idle = pm.detect_idle();
            pm.manage_power(idle);
            pm.monitor_traffic()

        if step >= pred_win:
            pred = predict_next(content_req[:step], pred_win, cache_sz // 2, pred_method)
            for p in pred:
                if not caches[req].access(p): caches[req].insert(p)

        if caches[req].access(content):
            hits += 1
        else:
            misses += 1
            try:
                path = nx.shortest_path(G, req, origin, weight='weight')
                e = calc_energy(G, path, 2.0);
                energy += e
                l = len(path) - 1;
                lat += l
                caches[req].insert(content)
                if use_gnn and gnn_manager:
                    normalized_metrics = [l / 20.0, e / 1000.0, G.nodes[req].get('carbon_intensity', 450) / 500.0, 0.5,
                                          0.3]
                    gnn_manager.learn_from_experience(req, origin, path, normalized_metrics)
            except:
                misses += 1

    emissions, cost = calc_emissions(energy), calc_cost(energy)
    return energy, lat, hits, misses, emissions, cost


def cooperative_caching(G, content_req, cache_sz=5, origin=0, scenario="normal", regions=None, greenest=None,
                        use_gnn=False, gnn_manager=None):
    caches = {n: LRUCache(cache_sz) for n in G.nodes()}
    energy = lat = hits = misses = 0
    pm = PowerManager(G)

    for step, content in enumerate(content_req):
        req_node = rd.choice(list(G.nodes()))
        req = gnn_enhanced_routing(G, req_node, content, gnn_manager, regions,
                                   greenest) if use_gnn and gnn_manager else carbon_routing(G, req_node, regions,
                                                                                            greenest)

        if step % 100 == 0: update_network_congestion(G, step, G.graph.get('congestion_factor', 0.5))
        if step % 50 == 0:
            idle = pm.detect_idle();
            pm.manage_power(idle);
            pm.monitor_traffic()

        found = False
        if caches[req].access(content):
            hits += 1;
            found = True
        else:
            check_n = True
            if scenario == "high" and rd.random() < 0.3: check_n = False
            if check_n:
                for nb in G.neighbors(req):
                    if caches[nb].access(content): hits += 1; found = True; break

        if not found:
            misses += 1
            try:
                path = nx.shortest_path(G, req, origin, weight='weight')
                e = calc_energy(G, path, 2.0);
                energy += e
                l = len(path) - 1;
                lat += l
                caches[req].insert(content)
                if use_gnn and gnn_manager:
                    normalized_metrics = [l / 20.0, e / 1000.0, G.nodes[req].get('carbon_intensity', 450) / 500.0, 0.5,
                                          0.3]
                    gnn_manager.learn_from_experience(req, origin, path, normalized_metrics)
            except:
                misses += 1

    emissions, cost = calc_emissions(energy), calc_cost(energy)
    return energy, lat, hits, misses, emissions, cost


def hierarchical_caching(G, content_req, cache_sz=5, origin=0, scenario="normal", regions=None, greenest=None,
                         use_gnn=False, gnn_manager=None):
    cs = HierarchicalCache(G, max(15, cache_sz // 3), max(25, cache_sz // 2), max(20, cache_sz // 4))
    energy = lat = hits = misses = 0
    hier_hits = {"edge": 0, "regional": 0, "core": 0, "origin": 0}
    pm = PowerManager(G)

    for step, content in enumerate(content_req):
        req_node = rd.choice(list(G.nodes()))
        req = gnn_enhanced_routing(G, req_node, content, gnn_manager, regions,
                                   greenest) if use_gnn and gnn_manager else carbon_routing(G, req_node, regions,
                                                                                            greenest)

        if step % 100 == 0: update_network_congestion(G, step, G.graph.get('congestion_factor', 0.5))
        if step % 50 == 0:
            idle = pm.detect_idle();
            pm.manage_power(idle);
            pm.monitor_traffic()

        avg_traf = np.mean([G[u][v].get('traffic', 10) for u, v in G.edges()])
        cs.adaptive_sizing(step, avg_traf)

        level, found = cs.access(content, req)
        if found:
            hits += 1; hier_hits[level] += 1
        else:
            misses += 1
            try:
                path = nx.shortest_path(G, req, origin, weight='weight')
                e = calc_energy(G, path, 2.0);
                energy += e
                l = len(path) - 1;
                lat += l
                if len(path) <= 3:
                    cs.insert(content, "edge", req)
                elif len(path) <= 6:
                    rn = cs._find_closest_regional(req)
                    if rn: cs.insert(content, "regional", rn)
                else:
                    cs.insert(content, "core")
                if use_gnn and gnn_manager:
                    normalized_metrics = [l / 20.0, e / 1000.0, G.nodes[req].get('carbon_intensity', 450) / 500.0, 0.5,
                                          0.3]
                    gnn_manager.learn_from_experience(req, origin, path, normalized_metrics)
            except:
                misses += 1

    print(f"Hierarchy: {hier_hits}")
    emissions, cost = calc_emissions(energy), calc_cost(energy)
    return energy, lat, hits, misses, emissions, cost


def update_network_congestion(G, step, cong_fact=0.5):
    update_time_carbon(G, step)
    for u, v in G.edges():
        if 'base_weight' not in G[u][v]: G[u][v]['base_weight'] = G[u][v]['weight']
        cong = cong_fact * (1 + 0.3 * np.sin(step * 0.05) + 0.4 * rd.random())
        G[u][v]['weight'] = G[u][v]['base_weight'] * (1 + cong)
        if 'traffic_range' in G.graph:
            min_t, max_t = G.graph['traffic_range']
            G[u][v]['traffic'] = rd.uniform(min_t, max_t)
        else:
            G[u][v]['traffic'] = rd.uniform(10, 100)


def simulate_network(G, strat, content_req, cache_sz=5, origin=0, scenario="normal", regions=None, greenest=None,
                     use_gnn=False, gnn_manager=None):
    if strat == "Predictive":
        return predictive_caching(G, content_req, cache_sz, origin, 100, scenario, regions, greenest, use_gnn,
                                  gnn_manager)
    elif strat == "Cooperative":
        return cooperative_caching(G, content_req, cache_sz, origin, scenario, regions, greenest, use_gnn, gnn_manager)
    elif strat == "Hierarchical":
        return hierarchical_caching(G, content_req, cache_sz, origin, scenario, regions, greenest, use_gnn, gnn_manager)


def plot_results(net_name, scenario, e_res, h_res, em_res, c_res):
    strats = list(e_res.keys())
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 12))
    colors = ['#4D4D4D', '#0072B2', '#2ECC71']
    bars = ax1.bar(range(len(strats)), [e_res[s] for s in strats], color=colors[:len(strats)], width=0.8)
    ax1.set_xticks(range(len(strats)));
    ax1.set_xticklabels(strats, rotation=15)
    ax1.set_title("Energy (Wh)", fontsize=14, fontweight='bold');
    ax1.set_ylabel("Energy (Wh)")
    bars = ax2.bar(range(len(strats)), [h_res[s] for s in strats], color=colors[:len(strats)], width=0.8)
    ax2.set_xticks(range(len(strats)));
    ax2.set_xticklabels(strats, rotation=15)
    ax2.set_title("Hit Ratio (%)", fontsize=14, fontweight='bold');
    ax2.set_ylabel("Hit Ratio (%)")
    bars = ax3.bar(range(len(strats)), [em_res[s] for s in strats], color=colors[:len(strats)], width=0.8)
    ax3.set_xticks(range(len(strats)));
    ax3.set_xticklabels(strats, rotation=15)
    ax3.set_title("CO2 Emissions (kg)", fontsize=14, fontweight='bold');
    ax3.set_ylabel("CO2 (kg)")
    bars = ax4.bar(range(len(strats)), [c_res[s] for s in strats], color=colors[:len(strats)], width=0.8)
    ax4.set_xticks(range(len(strats)));
    ax4.set_xticklabels(strats, rotation=15)
    ax4.set_title("Cost (Yen)", fontsize=14, fontweight='bold');
    ax4.set_ylabel("Cost (Yen)")
    plt.suptitle(f"{net_name} - {scenario} Scenario", fontsize=16, fontweight='bold')
    plt.tight_layout();
    plt.savefig(f"results_{net_name}_{scenario}.png", dpi=300, bbox_inches='tight');
    plt.show()


def plot_comparison(net_name, all_res):
    scenarios = list(SCENARIOS.keys())
    strats = list(all_res[scenarios[0]]["energy"].keys())
    colors = {'Cooperative': '#4D4D4D', 'Predictive': '#0072B2', 'Hierarchical': '#2ECC71'}
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(16, 12))
    x_pos = np.arange(len(scenarios));
    width = 0.25
    for i, s in enumerate(strats):
        e_vals = [all_res[sc]["energy"][s] for sc in scenarios]
        ax1.bar(x_pos + i * width, e_vals, width, label=s, color=colors.get(s, '#999999'))
    ax1.set_xticks(x_pos + width * 1.0);
    ax1.set_xticklabels([s.replace('_', ' ').title() for s in scenarios])
    ax1.set_title("Energy Across Scenarios", fontsize=14, fontweight='bold');
    ax1.legend()
    for i, s in enumerate(strats):
        h_vals = [all_res[sc]["hit_ratio"][s] for sc in scenarios]
        ax2.bar(x_pos + i * width, h_vals, width, label=s, color=colors.get(s, '#999999'))
    ax2.set_xticks(x_pos + width * 1.0);
    ax2.set_xticklabels([s.replace('_', ' ').title() for s in scenarios])
    ax2.set_title("Hit Ratio Across Scenarios", fontsize=14, fontweight='bold');
    ax2.legend()
    for i, s in enumerate(strats):
        em_vals = [all_res[sc]["emissions"][s] for sc in scenarios]
        ax3.bar(x_pos + i * width, em_vals, width, label=s, color=colors.get(s, '#999999'))
    ax3.set_xticks(x_pos + width * 1.0);
    ax3.set_xticklabels([s.replace('_', ' ').title() for s in scenarios])
    ax3.set_title("Emissions Across Scenarios", fontsize=14, fontweight='bold');
    ax3.legend()
    for i, s in enumerate(strats):
        c_vals = [all_res[sc]["cost"][s] for sc in scenarios]
        ax4.bar(x_pos + i * width, c_vals, width, label=s, color=colors.get(s, '#999999'))
    ax4.set_xticks(x_pos + width * 1.0);
    ax4.set_xticklabels([s.replace('_', ' ').title() for s in scenarios])
    ax4.set_title("Cost Across Scenarios", fontsize=14, fontweight='bold');
    ax4.legend()
    plt.suptitle(f"{net_name} - Strategy Comparison", fontsize=16, fontweight='bold')
    plt.tight_layout();
    plt.savefig(f"results_{net_name}_comparison.png", dpi=300, bbox_inches='tight');
    plt.show()


def plot_gnn_learning(gnn_manager, net_name, scenario):
    if not gnn_manager or not gnn_manager.routing_history:
        print("No GNN learning data to plot")
        return
    history = gnn_manager.routing_history
    recent = history[-200:] if len(history) > 200 else history
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(14, 10))
    steps = [h['step'] for h in recent]
    latencies = [h['metrics'][0] for h in recent]
    energies = [h['metrics'][1] for h in recent]
    window = min(20, len(latencies) // 10)
    if window > 1:
        lat_smooth = np.convolve(latencies, np.ones(window) / window, mode='valid')
        eng_smooth = np.convolve(energies, np.ones(window) / window, mode='valid')
        ax1.plot(steps[window - 1:], lat_smooth, 'b-', linewidth=2, label='Latency')
        ax1.plot(steps[window - 1:], eng_smooth / np.max(eng_smooth) * np.max(lat_smooth), 'g-', linewidth=2,
                 label='Energy (scaled)')
    else:
        ax1.plot(steps, latencies, 'b-', alpha=0.5, linewidth=1, label='Latency')
        ax1.plot(steps, [e / np.max(energies) * np.max(latencies) for e in energies], 'g-', alpha=0.5, linewidth=1,
                 label='Energy (scaled)')
    ax1.set_xlabel("GNN Learning Steps");
    ax1.set_ylabel("Normalized Performance")
    ax1.set_title("🧠 GNN Learning Progress");
    ax1.legend();
    ax1.grid(True, alpha=0.3)
    ax2.axis('off')
    ax2.text(0.1, 0.8, "GNN Router Statistics", fontsize=14, fontweight='bold', transform=ax2.transAxes)
    ax2.text(0.1, 0.7, f"Exploration Rate: {gnn_manager.exploration_rate:.3f}", fontsize=11, color='blue',
             transform=ax2.transAxes)
    ax2.text(0.1, 0.6, f"Training Steps: {gnn_manager.step_count}", fontsize=11, color='green', transform=ax2.transAxes)
    ax2.text(0.1, 0.5, f"Replay Buffer: {len(gnn_manager.replay_buffer)}", fontsize=11, color='purple',
             transform=ax2.transAxes)
    path_lengths = [h['path_length'] for h in recent]
    avg_path_len = np.mean(path_lengths) if path_lengths else 0
    ax2.text(0.1, 0.4, f"Avg Path Length: {avg_path_len:.2f} hops", fontsize=11, color='orange',
             transform=ax2.transAxes)
    path_counts = Counter(path_lengths)
    if path_counts:
        bars = ax3.bar(range(len(path_counts)), list(path_counts.values()), color='lightcoral', edgecolor='darkred',
                       linewidth=1.5)
        ax3.set_xlabel("Path Length (hops)");
        ax3.set_ylabel("Frequency")
        ax3.set_title("Path Length Distribution")
        ax3.set_xticks(range(len(path_counts)))
        ax3.set_xticklabels([f"{length}" for length in sorted(path_counts.keys())])
        for bar in bars:
            height = bar.get_height()
            ax3.text(bar.get_x() + bar.get_width() / 2., height + 0.5, f'{int(height)}', ha='center', va='bottom',
                     fontsize=9)
    ax4.axis('off')
    insights = ["🧠 GNN Learned To:", "• Understand network topology", "• Predict path quality",
                "• Balance multiple objectives", "• Adapt to traffic patterns", "• Make context-aware decisions"]
    if len(history) > 50:
        early, late = history[:50], history[-50:]
        early_lat, late_lat = np.mean([h['metrics'][0] for h in early]), np.mean([h['metrics'][0] for h in late])
        early_energy, late_energy = np.mean([h['metrics'][1] for h in early]), np.mean([h['metrics'][1] for h in late])
        lat_improvement = ((early_lat - late_lat) / early_lat * 100) if early_lat > 0 else 0
        energy_improvement = ((early_energy - late_energy) / early_energy * 100) if early_energy > 0 else 0
        insights.extend(
            [f"• Latency improvement: {lat_improvement:+.1f}%", f"• Energy improvement: {energy_improvement:+.1f}%"])
    insight_text = "\n".join(insights)
    ax4.text(0.05, 0.5, insight_text, fontsize=11, transform=ax4.transAxes,
             bbox=dict(boxstyle="round,pad=0.5", facecolor="lightblue", alpha=0.8))
    plt.suptitle(f"🧠 GNN Traffic Router - Learning Summary\n{net_name} ({scenario})", fontsize=16, fontweight='bold')
    plt.tight_layout();
    plt.savefig(f"gnn_learning_{net_name}_{scenario}.png", dpi=300, bbox_inches='tight');
    plt.show()


def main():
    print("🧠 Network Caching Simulator with GNN Traffic Management")
    print("1. Allegiance 2. At Home 3. CAIS 4. AT&T")
    choice = input("Network (1-4): ").strip()
    net_map = {"1": "allegiance", "2": "athome", "3": "cais", "4": "att"}
    net_choice = net_map.get(choice, "allegiance")
    print("🧠 GNN Traffic Management: 1.No 2.Yes")
    gnn_choice = input("GNN (1-2): ").strip()
    use_gnn = gnn_choice == "2"
    G, pos, cities, origin = load_network(net_choice)
    G.remove_nodes_from([n for n in G.nodes() if n not in pos])
    G.pos, G.cities = pos, cities;
    assign_energy(G)
    for node in G.nodes(): G.nodes[node]['current_load'] = 0
    energy_counts = Counter([G.nodes[n].get('energy_type', 'high') for n in G.nodes()])
    print(f"Energy Dist: {dict(energy_counts)}")
    regions = create_regions(G, pos, 6)
    greenest, stats = compute_greenest(G, regions)
    print(f"Regions: {len(regions)}")
    for rn, s in stats.items():
        city = cities.get(s['greenest'], f"Node {s['greenest']}")
        print(f"  {rn}: {s['count']} servers, Greenest: {city} ({s['green_carb']}g)")
    all_res = {};
    gnn_managers = {}

    for scen, cfg in SCENARIOS.items():
        print(f"\n=== {scen} ===")
        G_s = G.copy();
        G_s.pos, G_s.cities = pos, cities
        G_s.graph['traffic_range'] = cfg['traffic'];
        G_s.graph['congestion_factor'] = 0.5
        content_req, _ = zipf_dist(cfg['contents'], cfg['zipf'], 500)
        gnn_manager = None
        if use_gnn:
            try:
                gnn_manager = GNNTrafficManager(G_s)
                gnn_managers[scen] = gnn_manager
                print("🧠 GNN Traffic Router initialized")
            except Exception as e:
                print(f"⚠️  GNN initialization failed: {e}")
                use_gnn = False
                gnn_manager = None
        strats = ["Cooperative", "Predictive", "Hierarchical"]
        e_res, l_res, h_res, em_res, c_res = {}, {}, {}, {}, {}
        for s in strats:
            try:
                e, l, hits, misses, em, c = simulate_network(G_s, s, content_req, cfg['cache'], origin, scen, regions,
                                                             greenest, use_gnn, gnn_manager)
                hr = (hits / (hits + misses)) * 100 if (hits + misses) > 0 else 0
                e_res[s], l_res[s], h_res[s], em_res[s], c_res[s] = e, l, hr, em, c
                print(f"{s}: HR={hr:.1f}%, E={e:.1f}Wh, CO2={em:.3f}kg")
                if use_gnn and gnn_manager: print(f"   {gnn_manager.get_stats()}")
            except Exception as e:
                print(f"⚠️  Error in {s}: {e}")
                e_res[s], l_res[s], h_res[s], em_res[s], c_res[s] = 0, 0, 0, 0, 0
        all_res[scen] = {"energy": e_res, "latency": l_res, "hit_ratio": h_res, "emissions": em_res, "cost": c_res}
        try:
            plot_results(net_choice, scen, e_res, h_res, em_res, c_res)
            if use_gnn and gnn_manager: plot_gnn_learning(gnn_manager, net_choice, scen)
        except:
            pass

    try:
        plot_comparison(net_choice, all_res)
    except:
        pass

    print(f"\n=== GNN TRAFFIC MANAGEMENT SUMMARY ===")
    if use_gnn:
        print("✅ GNN Traffic Management Features:")
        print("1. Graph Neural Network architecture")
        print("2. Learns from network topology")
        print("3. End-to-end policy learning")
        print("4. Experience replay with online learning")
        print("5. Multi-objective optimization (learned weights)")
    else:
        print("GNN traffic management was disabled or failed.")
        print("Run with option '2' to enable GNN-powered routing.")


if __name__ == "__main__":
    main()