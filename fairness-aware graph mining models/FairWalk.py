import os
from collections import defaultdict

import numpy as np
import networkx as nx
import gensim
from joblib import Parallel, delayed
from tqdm import tqdm
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score
import random
from tqdm import tqdm


def parallel_generate_walks(d_graph: dict, global_walk_length: int, num_walks: int, cpu_num: int,
                            sampling_strategy: dict = None, num_walks_key: str = None, walk_length_key: str = None,
                            neighbors_key: str = None, probabilities_key: str = None,
                            first_travel_key: str = None, quiet: bool = False) -> list:
    """
    Generates the random walks which will be used as the skip-gram input.

    :return: List of walks. Each walk is a list of nodes.
    """

    walks = list()

    if not quiet:
        pbar = tqdm(total=num_walks, desc='Generating walks (CPU: {})'.format(cpu_num))

    for n_walk in range(num_walks):

        # Update progress bar
        if not quiet:
            pbar.update(1)

        # Shuffle the nodes
        shuffled_nodes = list(d_graph.keys())
        random.shuffle(shuffled_nodes)

        # Start a random walk from every node
        for source in shuffled_nodes:

            # Skip nodes with specific num_walks
            if source in sampling_strategy and \
                    num_walks_key in sampling_strategy[source] and \
                    sampling_strategy[source][num_walks_key] <= n_walk:
                continue

            # Start walk
            walk = [source]

            # Calculate walk length
            if source in sampling_strategy:
                walk_length = sampling_strategy[source].get(walk_length_key, global_walk_length)
            else:
                walk_length = global_walk_length

            # Perform walk
            while len(walk) < walk_length:

                walk_options = d_graph[walk[-1]].get(neighbors_key, None)

                # Skip dead end nodes
                if not walk_options:
                    break

                group2neighbors = d_graph[walk[-1]][neighbors_key]
                all_possible_groups = [group for group in group2neighbors if len(group2neighbors[group]) > 0]
                random_group = np.random.choice(all_possible_groups, size=1)[0]
                walk_options = walk_options[random_group]

                if len(walk) == 1:  # For the first step
                    probabilities = d_graph[walk[-1]][first_travel_key][random_group]
                    walk_to = np.random.choice(walk_options, size=1, p=probabilities)[0]
                else:
                    probabilities = d_graph[walk[-1]][probabilities_key][random_group][walk[-2]]
                    walk_to = np.random.choice(walk_options, size=1, p=probabilities)[0]

                walk.append(walk_to)

            walk = list(map(str, walk))  # Convert all to strings

            walks.append(walk)

    if not quiet:
        pbar.close()

    return walks


class FairWalk():
    FIRST_TRAVEL_KEY = 'first_travel_key'
    PROBABILITIES_KEY = 'probabilities'
    NEIGHBORS_KEY = 'neighbors'
    GROUP_KEY = 'group'
    WEIGHT_KEY = 'weight'
    NUM_WALKS_KEY = 'num_walks'
    WALK_LENGTH_KEY = 'walk_length'
    P_KEY = 'p'
    Q_KEY = 'q'

    def fit(self, adj,  labels, idx_train, dimensions: int = 64, walk_length: int = 20, num_walks: int = 5, p: float = 1,
                 q: float = 1, weight_key: str = 'weight', workers: int = 1, sampling_strategy: dict = None,
                 quiet: bool = False, temp_folder: str = None):
        """
        Initiates the FairWalk object, precomputes walking probabilities and generates the walks.

        :param graph: Input graph
        :param dimensions: Embedding dimensions (default: 128)
        :param walk_length: Number of nodes in each walk (default: 80)
        :param num_walks: Number of walks per node (default: 10)
        :param p: Return hyper parameter (default: 1)
        :param q: Inout parameter (default: 1)
        :param weight_key: On weighted graphs, this is the key for the weight attribute (default: 'weight')
        :param workers: Number of workers for parallel execution (default: 1)
        :param sampling_strategy: Node specific sampling strategies, supports setting node specific 'q', 'p', 'num_walks' and 'walk_length'.
        Use these keys exactly. If not set, will use the global ones which were passed on the object initialization
        :param temp_folder: Path to folder with enough space to hold the memory map of self.d_graph (for big graphs); to be passed joblib.Parallel.temp_folder
        """

        self.graph = nx.from_numpy_array(adj.to_dense().numpy())
        n = len(self.graph.nodes())
        node2group = {node: group for node, group in zip(self.graph.nodes(), (5 * np.random.random(n)).astype(int))}
        nx.set_node_attributes(self.graph, node2group, 'group')

        self.dimensions = dimensions
        self.walk_length = walk_length
        self.num_walks = num_walks
        self.p = p
        self.q = q
        self.weight_key = weight_key
        self.workers = workers
        self.quiet = quiet
        self.d_graph = defaultdict(dict)

        if sampling_strategy is None:
            self.sampling_strategy = {}
        else:
            self.sampling_strategy = sampling_strategy

        self.temp_folder, self.require = None, None
        if temp_folder:
            if not os.path.isdir(temp_folder):
                raise NotADirectoryError("temp_folder does not exist or is not a directory. ({})".format(temp_folder))

            self.temp_folder = temp_folder
            self.require = "sharedmem"

        self._precompute_probabilities()
        self.walks = self._generate_walks()
        self.embs=gensim.models.Word2Vec(self.walks,vector_size=self.dimensions, window=5).wv.vectors

        self.labels=labels
        self.lgreg = LogisticRegression(random_state=0, class_weight='balanced', max_iter=500).fit(
            self.embs[idx_train], labels[idx_train])


    def predict(self,idx_test):
        pred = self.lgreg.predict(self.embs[idx_test])
        score = f1_score(self.labels[idx_test], pred, average='micro')
        return score



    def _precompute_probabilities(self):
        """
        Precomputes transition probabilities for each node.
        """

        d_graph = self.d_graph

        nodes_generator = self.graph.nodes() if self.quiet \
            else tqdm(self.graph.nodes(), desc='Computing transition probabilities')

        node2groups = nx.get_node_attributes(self.graph, self.GROUP_KEY)
        groups = np.unique(list(node2groups.values()))

        # Init probabilities dict
        for node in self.graph.nodes():
            for group in groups:
                if self.PROBABILITIES_KEY not in d_graph[node]:
                    d_graph[node][self.PROBABILITIES_KEY] = dict()
                if group not in d_graph[node][self.PROBABILITIES_KEY]:
                    d_graph[node][self.PROBABILITIES_KEY][group] = dict()

        for source in nodes_generator:
            for current_node in self.graph.neighbors(source):

                unnormalized_weights = list()
                d_neighbors = list()
                neighbor_groups = list()

                # Calculate unnormalized weights
                for destination in self.graph.neighbors(current_node):

                    p = self.sampling_strategy[current_node].get(self.P_KEY,
                                                                 self.p) if current_node in self.sampling_strategy else self.p
                    q = self.sampling_strategy[current_node].get(self.Q_KEY,
                                                                 self.q) if current_node in self.sampling_strategy else self.q

                    if destination == source:  # Backwards probability
                        ss_weight = self.graph[current_node][destination].get(self.weight_key, 1) * 1 / p
                    elif destination in self.graph[source]:  # If the neighbor is connected to the source
                        ss_weight = self.graph[current_node][destination].get(self.weight_key, 1)
                    else:
                        ss_weight = self.graph[current_node][destination].get(self.weight_key, 1) * 1 / q

                    # Assign the unnormalized sampling strategy weight, normalize during random walk
                    unnormalized_weights.append(ss_weight)
                    d_neighbors.append(destination)
                    if self.GROUP_KEY not in self.graph.nodes[destination]:
                        raise Exception('no group attribute')
                    neighbor_groups.append(self.graph.nodes[destination][self.GROUP_KEY])

                unnormalized_weights = np.array(unnormalized_weights)
                d_neighbors = np.array(d_neighbors)
                neighbor_groups = np.array(neighbor_groups)

                for group in groups:
                    cur_unnormalized_weights = unnormalized_weights[neighbor_groups == group]
                    cur_d_neighbors = d_neighbors[neighbor_groups == group]

                    # Normalize
                    d_graph[current_node][self.PROBABILITIES_KEY][group][
                        source] = cur_unnormalized_weights / cur_unnormalized_weights.sum()

                    # Save neighbors
                    d_graph[current_node].setdefault(self.NEIGHBORS_KEY, {})[group] = list(cur_d_neighbors)

            # Calculate first_travel weights for source
            first_travel_weights = []
            first_travel_neighbor_groups = []
            for destination in self.graph.neighbors(source):
                first_travel_weights.append(self.graph[source][destination].get(self.weight_key, 1))
                first_travel_neighbor_groups.append(self.graph.nodes[destination][self.GROUP_KEY])

            first_travel_weights = np.array(first_travel_weights)
            first_travel_neighbor_groups = np.array(first_travel_neighbor_groups)
            d_graph[source][self.FIRST_TRAVEL_KEY] = {}
            for group in groups:
                cur_first_travel_weights = first_travel_weights[first_travel_neighbor_groups == group]
                d_graph[source][self.FIRST_TRAVEL_KEY][group] = cur_first_travel_weights / cur_first_travel_weights.sum()

    def _generate_walks(self) -> list:
        """
        Generates the random walks which will be used as the skip-gram input.
        :return: List of walks. Each walk is a list of nodes.
        """

        flatten = lambda l: [item for sublist in l for item in sublist]

        # Split num_walks for each worker
        num_walks_lists = np.array_split(range(self.num_walks), self.workers)

        walk_results = Parallel(n_jobs=self.workers, temp_folder=self.temp_folder, require=self.require)(
            delayed(parallel_generate_walks)(self.d_graph,
                                             self.walk_length,
                                             len(num_walks),
                                             idx,
                                             self.sampling_strategy,
                                             self.NUM_WALKS_KEY,
                                             self.WALK_LENGTH_KEY,
                                             self.NEIGHBORS_KEY,
                                             self.PROBABILITIES_KEY,
                                             self.FIRST_TRAVEL_KEY,
                                             self.quiet) for
            idx, num_walks
            in enumerate(num_walks_lists, 1))

        walks = flatten(walk_results)

        return walks
