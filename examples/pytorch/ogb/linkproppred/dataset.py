import random
import time

import numpy as np
import torch
import dgl

from utils import subgraph_extraction_labeling

class IGMCDataset(torch.utils.data.Dataset):
    def __init__(self, links, graph, 
                hop=1, sample_ratio=1.0, max_nodes_per_hop=200):
        self.links = links
        self.graph = graph

        self.hop = hop
        self.sample_ratio = sample_ratio
        self.max_nodes_per_hop = max_nodes_per_hop
    
    def __len__(self):
        return len(self.links)

    def __getitem__(self, idx):
        u, v = self.links[idx][0], self.links[idx][1]

        subgraph = subgraph_extraction_labeling(
            (u, v), self.graph, 'homo', 
            self.hop, self.sample_ratio, self.max_nodes_per_hop)
        return subgraph

class RandomDataset(torch.utils.data.Dataset):
    def __init__(self, data_size, graph, 
                hop=1, sample_ratio=1.0, max_nodes_per_hop=200):
        self.data_size = data_size
        self.graph = graph
        self.num_nodes = graph.number_of_nodes()

        self.hop = hop
        self.sample_ratio = sample_ratio
        self.max_nodes_per_hop = max_nodes_per_hop
    
    def __len__(self):
        return self.data_size
    
    def __getitem__(self, idx):
        # random sample two different nodes
        ind = torch.randint(0, self.num_nodes, (2, ))
        while (ind[0] == ind[1]):
            ind = torch.randint(0, self.num_nodes, (2, ))
        subgraph = subgraph_extraction_labeling(
            (ind[0], ind[1]), self.graph, 'homo', 
            self.hop, self.sample_ratio, self.max_nodes_per_hop)
        return subgraph    

def collate_igmc(g_list):    
    g = dgl.batch_hetero(g_list)
    return g

if __name__ == "__main__":
    seed = 1234
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    from ogb.linkproppred import DglLinkPropPredDataset

    raw_dataset = DglLinkPropPredDataset(name='ogbl-collab')
    train_graph = dgl.as_heterograph(raw_dataset[0])
    # there is one self edge in valid_neg and test_neg separately, remove it
    edge_split = raw_dataset.get_edge_split()
    edge = edge_split['valid']['edge_neg']
    edge_split['valid']['edge_neg'] = edge[edge[:, 0]!=edge[:, 1]]
    edge = edge_split['test']['edge_neg']
    edge_split['test']['edge_neg'] = edge[edge[:, 0]!=edge[:, 1]]

    edges = edge_split['valid']['edge']
    dataset = IGMCDataset(
        edges, train_graph, 
        hop=3, sample_ratio=1.0, max_nodes_per_hop=200)
    loader = torch.utils.data.DataLoader(dataset, batch_size=256, shuffle=True, 
                                        num_workers=16, collate_fn=collate_igmc)
    # batch = next(iter(train_loader))

    # random_dataset = RandomDataset(
    #     len(pos_train_edge), train_graph, 
    #    hop=3, sample_ratio=1.0, max_nodes_per_hop=200)
    # random_loader = torch.utils.data.DataLoader(random_dataset, batch_size=256, shuffle=True, 
    #                           num_workers=16, collate_fn=collate_igmc)
    # neg_iter = iter(random_loader)
    # batch = next(iter(random_loader))

    iter_dur = []
    for iter_idx, batch in enumerate(loader, start=1):
        t_start = time.time()

        inputs = batch #.to(device)
        # inputs = next(neg_iter) # .to(device)

        iter_dur.append(time.time() - t_start)
        if iter_idx % 100 == 0:
            print("Iter={}, time={:.4f}".format(
                iter_idx, np.average(iter_dur)))
            iter_cur = []
