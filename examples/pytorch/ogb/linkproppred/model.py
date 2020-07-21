import torch as th 
import torch.nn as nn
import torch.nn.functional as F

import dgl.function as fn
from dgl.utils import expand_as_pair, check_eq_shape

"""Customized GraphSAGE Layer for edge dropping and weight computation"""
class SAGEConv_Custom(nn.Module):
    def __init__(self,
                 in_feats,
                 out_feats,
                 aggregator_type,
                 feat_drop=0.,
                 bias=True,
                 norm=None,
                 activation=None):
        super(SAGEConv_Custom, self).__init__()

        self._in_src_feats, self._in_dst_feats = expand_as_pair(in_feats)
        self._out_feats = out_feats
        self._aggre_type = aggregator_type
        self.norm = norm
        self.feat_drop = nn.Dropout(feat_drop)
        self.activation = activation
        # aggregator type: mean/pool/lstm/gcn
        # if aggregator_type == 'pool':
        #     self.fc_pool = nn.Linear(self._in_src_feats, self._in_src_feats)
        # if aggregator_type == 'lstm':
        #     self.lstm = nn.LSTM(self._in_src_feats, self._in_src_feats, batch_first=True)
        # if aggregator_type != 'gcn':
        self.fc_self = nn.Linear(self._in_dst_feats, out_feats, bias=bias)
        self.fc_neigh = nn.Linear(self._in_src_feats, out_feats, bias=bias)
        self.reset_parameters()

    def reset_parameters(self):
        """Reinitialize learnable parameters."""
        gain = nn.init.calculate_gain('relu')
        # if self._aggre_type == 'pool':
        #     nn.init.xavier_uniform_(self.fc_pool.weight, gain=gain)
        # if self._aggre_type == 'lstm':
        #     self.lstm.reset_parameters()
        # if self._aggre_type != 'gcn':
        nn.init.xavier_uniform_(self.fc_self.weight, gain=gain)
        nn.init.xavier_uniform_(self.fc_neigh.weight, gain=gain)

    # def _lstm_reducer(self, nodes):
    #     m = nodes.mailbox['m'] # (B, L, D)
    #     batch_size = m.shape[0]
    #     h = (m.new_zeros((1, batch_size, self._in_src_feats)),
    #          m.new_zeros((1, batch_size, self._in_src_feats)))
    #     _, (rst, _) = self.lstm(m, h)
    #     return {'neigh': rst.squeeze(0)}

    def message_func(self, edges):
        msg = edges.src['h'] * edges.data['edge_weight']
        msg = msg * edges.data['edge_mask']
        return {'msg': msg}

    def forward(self, graph, feat):
        with graph.local_scope():
            # if isinstance(feat, tuple):
            #     feat_src = self.feat_drop(feat[0])
            #     feat_dst = self.feat_drop(feat[1])
            # else:
            feat_src = feat_dst = self.feat_drop(feat)
            h_self = feat_dst

            # # Handle the case of graphs without edges
            # if graph.number_of_edges() == 0:
            #     graph.dstdata['neigh'] = th.zeros(
            #         feat_dst.shape[0], self._in_src_feats).to(feat_dst)

            if self._aggre_type == 'mean':
                graph.srcdata['h'] = feat_src
                graph.update_all(self.message_func, fn.mean('msg', 'neigh'))
                # graph.update_all(fn.copy_src('h', 'm'), fn.mean('m', 'neigh'))
                h_neigh = graph.dstdata['neigh']
            # elif self._aggre_type == 'gcn':
            #     check_eq_shape(feat)
            #     graph.srcdata['h'] = feat_src
            #     graph.dstdata['h'] = feat_dst     # same as above if homogeneous
            #     graph.update_all(fn.copy_src('h', 'm'), fn.sum('m', 'neigh'))
            #     # divide in_degrees
            #     degs = graph.in_degrees().to(feat_dst)
            #     h_neigh = (graph.dstdata['neigh'] + graph.dstdata['h']) / (degs.unsqueeze(-1) + 1)
            # elif self._aggre_type == 'pool':
            #     graph.srcdata['h'] = F.relu(self.fc_pool(feat_src))
            #     graph.update_all(fn.copy_src('h', 'm'), fn.max('m', 'neigh'))
            #     h_neigh = graph.dstdata['neigh']
            # elif self._aggre_type == 'lstm':
            #     graph.srcdata['h'] = feat_src
            #     graph.update_all(fn.copy_src('h', 'm'), self._lstm_reducer)
            #     h_neigh = graph.dstdata['neigh']
            else:
                raise KeyError('Aggregator type {} not recognized.'.format(self._aggre_type))

            # GraphSAGE GCN does not require fc_self.
            # if self._aggre_type == 'gcn':
            #     rst = self.fc_neigh(h_neigh)
            # else:
            rst = self.fc_self(h_self) + self.fc_neigh(h_neigh)

            # # activation
            # if self.activation is not None:
            #     rst = self.activation(rst)
            # # normalization
            # if self.norm is not None:
            #     rst = self.norm(rst)
            return rst

class IGMC(nn.Module):
    def __init__(self, in_feats, gconv=SAGEConv_Custom, latent_dim=[32, 32, 32, 32], 
                regression=False, edge_dropout=0.2, 
                force_undirected=False, side_features=False, n_side_features=0, 
                multiply_by=1):
        super(IGMC, self).__init__()

        self.regression = regression
        self.edge_dropout = edge_dropout
        self.force_undirected = force_undirected
        self.side_features = side_features
        self.multiply_by = multiply_by

        self.convs = th.nn.ModuleList()
        self.convs.append(gconv(in_feats, latent_dim[0],
                                aggregator_type='mean', feat_drop=0., bias=True, 
                                norm=None, activation=None))
        for i in range(0, len(latent_dim)-1):
            self.convs.append(gconv(latent_dim[i], latent_dim[i+1], 
                                    aggregator_type='mean', feat_drop=0., bias=True, 
                                    norm=None, activation=None))
        
        self.lin1 = nn.Linear(2 * sum(latent_dim), 128, bias=True)
        # if side_features:
        #     self.lin1 = nn.Linear(2 * sum(latent_dim) + n_side_features, 128)
        self.lin2 = nn.Linear(128, 1, bias=True)
        self.reset_parameters()

    def reset_parameters(self):
        for conv in self.convs:
            conv.reset_parameters()
            # print (conv.weight.abs().mean(), conv.bias.abs().mean())
        self.lin1.reset_parameters()
        self.lin2.reset_parameters()

    def forward(self, block):
        block = edge_drop(block, self.edge_dropout, self.training)

        concat_states = []
        x = block.ndata['x']
        for conv in self.convs:
            # drop mask zero denotes the edge dropped
            # [zhoujf] try relu and dropout
            x = th.tanh(conv(block, x))
            concat_states.append(x)
        concat_states = th.cat(concat_states, 1)    

        query = block.ndata['x'][:, 0] == 1
        query_feat = concat_states[query].reshape([-1, 2, concat_states.shape[-1]])
        x = th.cat([query_feat[:, 0, :], query_feat[:, 1, :]], 1)
        # if self.side_features:
        #     x = th.cat([x, data.u_feature, data.v_feature], 1)

        x = F.relu(self.lin1(x))
        x = F.dropout(x, p=0.5, training=self.training)
        x = self.lin2(x)
        return th.sigmoid(x)

    def __repr__(self):
        return self.__class__.__name__

def edge_drop(graph, edge_dropout=0.2, training=True):
    assert edge_dropout >= 0.0 and edge_dropout <= 1.0, 'Invalid dropout rate.'

    if not training:
        return graph

    # set edge weight to zero in directional mode
    src, _ = graph.edges()
    to_drop = src.new_full((graph.number_of_edges(), ), edge_dropout, dtype=th.float)
    to_drop = th.bernoulli(to_drop).to(th.bool)
    graph.edata['edge_mask'][to_drop] = 0

    return graph
