import random
import time

import numpy as np
import torch
import dgl

from utils import subgraph_extraction_labeling

class IGMCDataset(torch.utils.data.Dataset):
    def __init__(self, links, graph, node_labeling_mode='homo',
                hop=1, sample_ratio=1.0, max_nodes_per_hop=200, neg=False):
        self.links = links
        self.graph = graph

        self.node_labeling_mode = node_labeling_mode
        self.hop = hop
        self.sample_ratio = sample_ratio
        self.max_nodes_per_hop = max_nodes_per_hop
    
        # self.neg = neg
    def __len__(self):
        # if not self.neg:
        #     return len(self.links['edge'])
        # else:
        #     return len(self.links['edge_neg'])
        return len(self.links)

    def __getitem__(self, idx):
        # if not self.neg:
        #     item = {'edge': self.links['edge'][idx], 'year': self.links['year'][idx]}
        # else:
        #     item = {'edge': self.links['edge_neg'][idx], 'year': 2020}
        u, v = self.links[idx][0], self.links[idx][1]

        subgraph = subgraph_extraction_labeling(
            (u, v), self.graph, self.node_labeling_mode, 
            self.hop, self.sample_ratio, self.max_nodes_per_hop)
        return subgraph

class RandomDataset(torch.utils.data.Dataset):
    def __init__(self, data_size, graph, node_labeling_mode='homo',
                hop=1, sample_ratio=1.0, max_nodes_per_hop=200):
        self.data_size = data_size
        self.graph = graph
        self.num_nodes = graph.number_of_nodes()

        self.node_labeling_mode = node_labeling_mode
        self.hop = hop
        self.sample_ratio = sample_ratio
        self.max_nodes_per_hop = max_nodes_per_hop
    
    def __len__(self):
        return self.data_size
    
    def __getitem__(self, idx):
        # random sample two different nodes
        ind = torch.randint(0, self.num_nodes, (2, ))
        while (ind[0]==ind[1]):
            ind = torch.randint(0, self.num_nodes, (2, ))

        # item = {'edge': edge, 'year': 2020} # random edge year will not affect time_mask
        subgraph = subgraph_extraction_labeling(
            (ind[0], ind[1]), self.graph, self.node_labeling_mode, 
            self.hop, self.sample_ratio, self.max_nodes_per_hop)
        return subgraph    

def collate_igmc(g_list):    
    g = dgl.batch(g_list)
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
    edge_split = raw_dataset.get_edge_split()
    train_graph = dgl.as_heterograph(raw_dataset[0])

    # remove self_loop edge in valid_neg and test_neg
    edge = edge_split['valid']['edge_neg']
    edge_split['valid']['edge_neg'] = edge[edge[:, 0]!=edge[:, 1]]
    edge = edge_split['test']['edge_neg']
    edge_split['test']['edge_neg'] = edge[edge[:, 0]!=edge[:, 1]]
    
    # # flatten the train graph
    # pos_train_edges = edge_split['train']['edge']
    # uni_edges, rev_idx = torch.unique(pos_train_edges, dim=0, return_inverse=True)
    # src = torch.cat([uni_edges[:, 0], uni_edges[:, 1]])
    # dst = torch.cat([uni_edges[:, 1], uni_edges[:, 0]])
    
    # _, _, eids = train_graph.edge_ids(pos_train_edges[:, 0], pos_train_edges[:, 1], return_uv=True)
    # edge_weight = train_graph.edata['edge_weight'][eids]
    # edge_weight_sum = torch.zeros((uni_edges.shape[0], 1))
    # for eid in range(len(pos_train_edges)):
    #     edge_weight_sum[rev_idx[eid]] += edge_weight[eid]        
    # edge_weight_sum = torch.cat([edge_weight_sum, edge_weight_sum])

    # train_graph = dgl.graph((src, dst))
    # train_graph.edata['edge_weight'] = edge_weight_sum

    dataset = IGMCDataset(
        edge_split['valid'], train_graph, 'homo',
        hop=3, sample_ratio=1.0, max_nodes_per_hop=200, neg=True)
    loader = torch.utils.data.DataLoader(dataset, batch_size=256, shuffle=False, 
                                        num_workers=16, collate_fn=collate_igmc)
    # batch = next(iter(train_loader))

    random_dataset = RandomDataset(
        len(edge_split['train']['edge']), train_graph, 
       hop=3, sample_ratio=1.0, max_nodes_per_hop=200)
    random_loader = torch.utils.data.DataLoader(random_dataset, batch_size=256, shuffle=True, 
                              num_workers=16, collate_fn=collate_igmc)
    neg_iter = iter(random_loader)
    # batch = next(iter(random_loader))

    # iter_dur = []
    t_start = time.time()
    for iter_idx, batch in enumerate(loader, start=1): # 1.6min

        inputs = batch #.to(device)
        inputs = next(neg_iter) # .to(device)

        # iter_dur.append(time.time() - t_start)
        # if iter_idx % 100 == 0:
        #     print("Iter={}, time={:.4f}".format(
        #         iter_idx, np.average(iter_dur)))
        #     iter_dur = []

    print("time={:.2}".format((time.time() - t_start)/60))
