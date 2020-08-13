import os
import random

import scipy.sparse as sp
import warnings
warnings.simplefilter('ignore', sp.SparseEfficiencyWarning)
warnings.filterwarnings('ignore')

import numpy as np
import torch as th
import dgl 

class MetricLogger(object):
    def __init__(self, save_dir, runs, info=None):
        self.save_dir = save_dir
        # self.log_interval = log_interval
        self.info = info
        self.results = [[] for _ in range(runs)]

    def add_result(self, run, result):
        # assert len(result) == 3
        assert run >= 0 and run < len(self.results)
        self.results[run].append(result)

    def print_statistics(self, run=None):
        if run is not None:
            result = 100 * torch.tensor(self.results[run])
            argmax = result[:, 1].argmax().item()
            print(f'Run {run + 1:02d}:')
            print(f'Highest Train: {result[:, 0].max():.2f}')
            print(f'Highest Valid: {result[:, 1].max():.2f}')
            print(f'  Final Train: {result[argmax, 0]:.2f}')
            print(f'   Final Test: {result[argmax, 2]:.2f}')
        else:
            result = 100 * torch.tensor(self.results)

            best_results = []
            for r in result:
                train1 = r[:, 0].max().item()
                valid = r[:, 1].max().item()
                train2 = r[r[:, 1].argmax(), 0].item()
                test = r[r[:, 1].argmax(), 2].item()
                best_results.append((train1, valid, train2, test))

            best_result = torch.tensor(best_results)

            print(f'All runs:')
            r = best_result[:, 0]
            print(f'Highest Train: {r.mean():.2f} ± {r.std():.2f}')
            r = best_result[:, 1]
            print(f'Highest Valid: {r.mean():.2f} ± {r.std():.2f}')
            r = best_result[:, 2]
            print(f'  Final Train: {r.mean():.2f} ± {r.std():.2f}')
            r = best_result[:, 3]
            print(f'   Final Test: {r.mean():.2f} ± {r.std():.2f}')

    # def dump(self, info, model, optimizer):
    #     epoch, train_loss, test_rmse = info['epoch'], info['train_loss'], info['test_rmse']
    #     with open(os.path.join(self.save_dir, 'log.txt'), 'a') as f:
    #         f.write('Epoch {}, train loss {:.4f}, test rmse {:.6f}\n'.format(
    #             epoch, train_loss, test_rmse))
    #     if type(epoch) == int and epoch % self.log_interval == 0:
    #         print('Saving model states...')
    #         model_name = os.path.join(self.save_dir, 'model_checkpoint{}.pth'.format(epoch))
    #         optimizer_name = os.path.join(
    #             self.save_dir, 'optimizer_checkpoint{}.pth'.format(epoch)
    #         )
    #         if model is not None:
    #             th.save(model.state_dict(), model_name)
    #         if optimizer is not None:
    #             th.save(optimizer.state_dict(), optimizer_name)

def torch_total_param_num(net):
    return sum([np.prod(p.shape) for p in net.parameters()])

def torch_net_info(net, save_path=None):
    info_str = 'Total Param Number: {}\n'.format(torch_total_param_num(net)) +\
               'Params:\n'
    for k, v in net.named_parameters():
        info_str += '\t{}: {}, {}\n'.format(k, v.shape, np.prod(v.shape))
    info_str += str(net)
    if save_path is not None:
        with open(save_path, 'w') as f:
            f.write(info_str)
    return info_str

def MinMaxScaling(x, axis=0):
    dist = x.max(axis=axis) - x.min(axis=axis)
    x = (x - x.min(axis=axis)) / (dist + 1e-7)
    return x

def one_hot(idx, length):
    x = th.zeros([len(idx), length])
    x[th.arange(len(idx)), idx] = 1.0
    return x

def cal_dist(csr_graph, node_to_remove):
    # cal dist to node 0, with target edge nodes 0/1 removed
    nodes = list(set(range(csr_graph.shape[1])) - set([node_to_remove]))
    csr_graph = csr_graph[nodes, :][:, nodes]
    dists = np.clip(sp.csgraph.dijkstra(
                        csr_graph, indices=0, directed=False, unweighted=True, limit=1e6
                    )[1:], 0, 1e7)
    return dists.astype(np.int64)

def get_neighbor_nodes_labels(ind, graph, mode="bipartite",
                              hop=1, sample_ratio=1.0, max_nodes_per_hop=200,
                              enclosing_sub_graph=False):
    
    if mode=="bipartite":
        # 1. neighbor nodes sampling
        dist = 0
        u_nodes, v_nodes = ind[0].unsqueeze(dim=0), ind[1].unsqueeze(dim=0)
        u_dist, v_dist = th.tensor([0]), th.tensor([0])
        u_visited, v_visited = th.unique(u_nodes), th.unique(v_nodes)
        u_fringe, v_fringe = th.unique(u_nodes), th.unique(v_nodes)

        for dist in range(1, hop+1):
            # sample neigh alternately
            u_fringe, v_fringe = graph.in_edges(v_fringe)[0], graph.in_edges(u_fringe)[0]
            u_fringe = th.from_numpy(np.setdiff1d(u_fringe.numpy(), u_visited.numpy()))
            v_fringe = th.from_numpy(np.setdiff1d(v_fringe.numpy(), v_visited.numpy()))
            u_visited = th.unique(th.cat([u_visited, u_fringe]))
            v_visited = th.unique(th.cat([v_visited, v_fringe]))

            if sample_ratio < 1.0:
                shuffled_idx = th.randperm(len(u_fringe))
                u_fringe = u_fringe[shuffled_idx[:int(sample_ratio*len(u_fringe))]]
                shuffled_idx = th.randperm(len(v_fringe))
                v_fringe = v_fringe[shuffled_idx[:int(sample_ratio*len(v_fringe))]]
            if max_nodes_per_hop is not None:
                if max_nodes_per_hop < len(u_fringe):
                    shuffled_idx = th.randperm(len(u_fringe))
                    u_fringe = u_fringe[shuffled_idx[:max_nodes_per_hop]]
                if max_nodes_per_hop < len(v_fringe):
                    shuffled_idx = th.randperm(len(v_fringe))
                    v_fringe = v_fringe[shuffled_idx[:max_nodes_per_hop]]
            if len(u_fringe) == 0 and len(v_fringe) == 0:
                break
            u_nodes = th.cat([u_nodes, u_fringe])
            v_nodes = th.cat([v_nodes, v_fringe])
            u_dist = th.cat([u_dist, th.full((len(u_fringe), ), dist, dtype=th.int64)])
            v_dist = th.cat([v_dist, th.full((len(v_fringe), ), dist, dtype=th.int64)])
 
        nodes = th.cat([u_nodes, v_nodes])

        # 2. node labeling
        u_node_labels = th.stack([x*2 for x in u_dist])
        v_node_labels = th.stack([x*2+1 for x in v_dist])
        node_labels = th.cat([u_node_labels, v_node_labels])
    
    elif mode=="homo": 
        # 1. neighbor nodes sampling
        dist = 0
        nodes = th.stack(ind)
        dists = th.zeros_like(nodes) 
        visited = th.unique(nodes)
        fringe = th.unique(nodes)

        for dist in range(1, hop+1):
            fringe = graph.in_edges(fringe)[0]    
            fringe = th.from_numpy(np.setdiff1d(fringe.numpy(), visited.numpy()))
            visited = th.unique(th.cat([visited, fringe]))

            if sample_ratio < 1.0:
                shuffled_idx = th.randperm(len(fringe))
                fringe = fringe[shuffled_idx[:int(sample_ratio*len(fringe))]]
            if max_nodes_per_hop is not None and max_nodes_per_hop < len(fringe):
                shuffled_idx = th.randperm(len(fringe))
                fringe = fringe[shuffled_idx[:max_nodes_per_hop]]
            if len(fringe) == 0:
                break
            nodes = th.cat([nodes, fringe])
            dists = th.cat([dists, th.full((len(fringe), ), dist, dtype=th.int64)])
        
        # 2. node labeling
        node_labels = dists
    
    elif mode=="grail":
        # 1. neighbor nodes sampling
        # make sure ind not in uv nodes.
        u_nodes, v_nodes = th.tensor([], dtype=th.long), th.tensor([], dtype=th.long)
        # u_dist, v_dist = th.tensor([0]), th.tensor([0])
        u_visited, v_visited = th.tensor([ind[0]]), th.tensor([ind[1]])
        u_fringe, v_fringe = th.tensor([ind[0]]), th.tensor([ind[1]])

        for dist in range(1, hop+1):
            # sample neigh separately
            u_fringe = graph.in_edges(u_fringe)[0]
            v_fringe = graph.in_edges(v_fringe)[0]

            u_fringe = th.from_numpy(np.setdiff1d(u_fringe.numpy(), u_visited.numpy()))
            v_fringe = th.from_numpy(np.setdiff1d(v_fringe.numpy(), v_visited.numpy()))
            u_visited = th.unique(th.cat([u_visited, u_fringe]))
            v_visited = th.unique(th.cat([v_visited, v_fringe]))

            if sample_ratio < 1.0:
                shuffled_idx = th.randperm(len(u_fringe))
                u_fringe = u_fringe[shuffled_idx[:int(sample_ratio*len(u_fringe))]]
                shuffled_idx = th.randperm(len(v_fringe))
                v_fringe = v_fringe[shuffled_idx[:int(sample_ratio*len(v_fringe))]]
            if max_nodes_per_hop is not None:
                if max_nodes_per_hop < len(u_fringe):
                    shuffled_idx = th.randperm(len(u_fringe))
                    u_fringe = u_fringe[shuffled_idx[:max_nodes_per_hop]]
                if max_nodes_per_hop < len(v_fringe):
                    shuffled_idx = th.randperm(len(v_fringe))
                    v_fringe = v_fringe[shuffled_idx[:max_nodes_per_hop]]
            if len(u_fringe) == 0 and len(v_fringe) == 0:
                break
            u_nodes = th.cat([u_nodes, u_fringe])
            v_nodes = th.cat([v_nodes, v_fringe])
            # u_dist = th.cat([u_dist, th.full((len(u_fringe), ), dist, dtype=th.int64)])
            # v_dist = th.cat([v_dist, th.full((len(v_fringe), ), dist, dtype=th.int64)])
        
        if enclosing_sub_graph:
            nodes = th.from_numpy(np.intersect1d(u_nodes.numpy(), v_nodes.numpy()))
        else:
            nodes = th.unique(th.cat([u_nodes, v_nodes]))
            nodes = nodes[nodes!=ind[0]]
            nodes = nodes[nodes!=ind[1]]

        # concatenate ind to front, and node labels of ind can be added easily.
        nodes = th.cat([th.stack(ind), nodes])
       
        # 2. node labeling
        if isinstance(graph, dgl.DGLHeteroGraph):
            graph = dgl.as_immutable_graph(graph)
        csr_subgraph = graph.subgraph(nodes).adjacency_matrix_scipy(return_edge_ids=False)
        dists = th.stack([th.tensor(cal_dist(csr_subgraph, 1)), 
                          th.tensor(cal_dist(csr_subgraph, 0))], axis=1)
        ind_labels = th.tensor([[0, 1], [1, 0]])
        node_labels = th.cat([ind_labels, dists]) if dists.size() else ind_labels

        # 3. prune nodes that are at a distance greater than hop from neigh of the target nodes
        pruned_mask = th.max(node_labels, axis=1)[0] <= hop
        nodes, node_labels = nodes[pruned_mask], node_labels[pruned_mask]
    else:
        raise NotImplementedError
    return nodes, node_labels

def subgraph_extraction_labeling(ind, graph, mode="bipartite", 
                                 hop=1, sample_ratio=1.0, max_nodes_per_hop=200):
    # edge, year = item['edge'], item['year']

    # extract the h-hop enclosing subgraph nodes around link 'ind'
    nodes, node_labels = get_neighbor_nodes_labels((ind[0], ind[1]), graph, mode, 
                                                   hop, sample_ratio, max_nodes_per_hop)

    if isinstance(graph, dgl.DGLGraph):
        graph = dgl.as_heterograph(graph)
    subgraph = graph.subgraph(nodes)
    if mode == "bipartite":
        subgraph.ndata['nlabel'] = one_hot(node_labels, (hop+1)*2)
    elif mode == "homo":
        subgraph.ndata['nlabel'] = one_hot(node_labels, hop+1)
    elif mode == "grail":
        subgraph.ndata['nlabel'] = th.cat([one_hot(node_labels[:, 0], hop+1), 
                                      one_hot(node_labels[:, 1], hop+1)], dim=1)
    else:
        raise NotImplementedError
    
    # subgraph.ndata['x'] = th.cat([subgraph.ndata['nlabel'], subgraph.ndata['refex']], dim=1)
    subgraph.ndata['x'] = subgraph.ndata['nlabel']
    
    # set edge weight to zero as to remove links between target nodes in training process
    subgraph.edata['edge_mask'] = th.ones((subgraph.number_of_edges(), 1))
    su = subgraph.nodes()[subgraph.ndata[dgl.NID]==ind[0]]
    sv = subgraph.nodes()[subgraph.ndata[dgl.NID]==ind[1]]
    _, _, target_edges = subgraph.edge_ids([su, sv], [sv, su], return_uv=True)
    # time_mask = subgraph.edata['edge_year'] >= year
    subgraph.edata['edge_mask'][target_edges] = 0

    return subgraph

if __name__ == "__main__":
    seed = 1234
    random.seed(seed)
    np.random.seed(seed)
    th.manual_seed(seed)
    if th.cuda.is_available():
        th.cuda.manual_seed_all(seed)

    from ogb.linkproppred import DglLinkPropPredDataset

    raw_dataset = DglLinkPropPredDataset(name='ogbl-collab')
    edge_split = raw_dataset.get_edge_split()
    train_graph = dgl.as_heterograph(raw_dataset[0])

    pos_train_edge = edge_split['train']['edge']

    u, v = th.tensor(124049), th.tensor(25199)
    subgraph = subgraph_extraction_labeling((u, v), train_graph, 'homo', 
            hop=3, sample_ratio=1.0, max_nodes_per_hop=200)
    pass
