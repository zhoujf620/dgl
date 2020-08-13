import os
import sys
import time
import glob
import random
import argparse
from shutil import copy

import numpy as np
import torch
from torch.utils.data import DataLoader
from ogb.linkproppred import DglLinkPropPredDataset, Evaluator
import dgl

from dataset import IGMCDataset, RandomDataset, collate_igmc
from model import IGMC
from utils import MetricLogger, MinMaxScaling

os.environ['TZ'] = 'Asia/Shanghai'
time.tzset()

@torch.no_grad()
def test_split(model, device, edges, train_graph, args):
    model.eval()
    
    dataset = IGMCDataset(
        edges, train_graph, args.node_labeling_mode,
        args.hop, args.sample_ratio, args.max_nodes_per_hop)
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True, 
                              num_workers=args.num_workers, collate_fn=collate_igmc)
    preds = []
    for batch in loader:
        inputs = batch.to(device)
        preds += [model(inputs).squeeze().cpu()]
    preds = torch.cat(preds, dim=0)
    return preds

def test_epoch(model, loss_fn, device, 
        evaluator, edge_split, train_graph, args):
    print("=== start testing on pos_train_edges... ===")
    pos_train_preds = test_split(model, device, edge_split['train']['edge'], train_graph, args)
    print("=== start testing on pos_valid_edges... ===")
    pos_valid_preds = test_split(model, device, edge_split['valid']['edge'], train_graph, args)
    print("=== start testing on neg_valid_edges... ===")
    neg_valid_preds = test_split(model, device, edge_split['valid']['edge_neg'], train_graph, args)
    print("=== start testing on pos_test_edges... ===")
    pos_test_preds = test_split(model, device, edge_split['test']['edge'], train_graph, args)
    print("=== start testing on neg_test_edges... ===")
    neg_test_preds = test_split(model, device, edge_split['test']['edge_neg'], train_graph, args)

    # results = {}
    # for K in [10, 50, 100]:
    K = 50
    evaluator.K = K
    train_hits = evaluator.eval({
        'y_pred_pos': pos_train_preds,
        'y_pred_neg': neg_valid_preds,
    })[f'hits@{K}']
    valid_hits = evaluator.eval({
        'y_pred_pos': pos_valid_preds,
        'y_pred_neg': neg_valid_preds,
    })[f'hits@{K}']
    test_hits = evaluator.eval({
        'y_pred_pos': pos_test_preds,
        'y_pred_neg': neg_test_preds,
    })[f'hits@{K}']

    return (train_hits, valid_hits, test_hits)

# @profile
def train_epoch(model, loss_fn, optimizer, device, log_interval, 
                pos_train_edge, train_graph, args, epoch_idx):
    model.train()

    train_dataset = IGMCDataset(
        pos_train_edge, train_graph, args.node_labeling_mode,
        args.hop, args.sample_ratio, args.max_nodes_per_hop)
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, 
                              num_workers=args.num_workers, collate_fn=collate_igmc)

    random_dataset = RandomDataset(
        len(pos_train_edge), train_graph, args.node_labeling_mode,
        args.hop, args.sample_ratio, args.max_nodes_per_hop)
    random_loader = DataLoader(random_dataset, batch_size=args.batch_size, shuffle=True, 
                              num_workers=args.num_workers, collate_fn=collate_igmc)
    neg_iter = iter(random_loader)

    total_loss = total_examples = 0

    iter_loss = 0.
    iter_cnt = 0
    iter_dur = []
    for iter_idx, batch in enumerate(train_loader, start=1):
        t_start = time.time()

        inputs = batch.to(device)
        pos_preds = model(inputs)
        pos_loss = loss_fn(pos_preds, torch.ones_like(pos_preds))

        # Just do some trivial random sampling.
        inputs = next(neg_iter).to(device)
        neg_preds = model(inputs)
        neg_loss = loss_fn(neg_preds, torch.zeros_like(neg_preds))

        optimizer.zero_grad()
        loss = pos_loss + neg_loss
        loss.backward()
        # torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()

        num_examples = pos_preds.size(0)
        total_loss += loss.item() * num_examples
        total_examples += num_examples

        iter_loss += loss.item() * num_examples
        iter_cnt += num_examples
        iter_dur.append(time.time() - t_start)

        if iter_idx % log_interval == 0:
            print("Iter={}, loss={:.4f}, time={:.4f}".format(
                iter_idx, iter_loss/iter_cnt, np.average(iter_dur)))
            iter_loss = 0.
            iter_cnt = 0
            iter_cur = []

        # if iter_idx % 1000 == 0:
        #     file_name = 'ckpt_' + str(epoch_idx) + '_' + str(iter_idx) + '.pt'
        #     torch.save(model.state_dict(), os.path.join(args.save_dir, file_name))

    return total_loss / total_examples # len(loader.dataset)

def main(args):
    #nodes: 235868
    #edges: 1179052, 60084, 100000, 46329, 100000
    #       967632,  28072, 8,      16965, 8
    data_name = 'ogbl-collab'
    raw_dataset = DglLinkPropPredDataset(name=data_name)
    edge_split = raw_dataset.get_edge_split()
    train_graph = dgl.as_heterograph(raw_dataset[0])

    # # add refex feature
    # refex_feature = torch.load("./refex_feature.pt")
    # print("refex feature shape: {}".format(refex_feature.numpy().shape))
    # train_graph.ndata['refex'] = refex_feature

    # # add gdv feature
    # gdv_feature = np.loadtxt('./{}.gdv'.format(data_name), dtype=np.float32)
    # print("gdv feature shape: {}".format(gdv_feature.shape))
    # gdv_feature = MinMaxScaling(gdv_feature, axis=0)
    # train_graph.ndata['gdv'] = torch.from_numpy(gdv_feature)

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

    in_feats = args.hop+1 if args.node_labeling_mode=='homo' else (args.hop+1)*2
    # in_feats = train_graph.ndata['refex'].shape[1]
    model = IGMC(in_feats).to(args.device)

    evaluator = Evaluator(name='ogbl-collab')
    logger = MetricLogger(args.save_dir, args.runs, args)

    for run_idx in range(args.runs):
        model.reset_parameters()
        loss_fn = torch.nn.BCELoss().to(args.device)
        optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

        best_epoch = 0
        best_hits = -np.inf
        for epoch_idx in range(1, args.epochs+1):
            print ('Epoch', epoch_idx)
            train_loss = train_epoch(model, loss_fn, optimizer, 
                                    args.device, args.train_log_interval, 
                                    edge_split['train']['edge'], train_graph, args, epoch_idx)
            test_result = test_epoch(model, loss_fn, args.device, 
                                    evaluator, edge_split, train_graph, args)

            logger.add_result(run_idx, test_result)
            train_hits, valid_hits, test_hits = test_result
            test_info = (f'Run: {run_idx + 1:02d}, '
                        f'Epoch: {epoch_idx:02d}, '
                        f'Loss: {train_loss:.4f}, '
                        f'Train: {100 * train_hits:.2f}%, '
                        f'Valid: {100 * valid_hits:.2f}%, '
                        f'Test: {100 * test_hits:.2f}%')
            print('=== {} ==='.format(test_info))
            with open(os.path.join(args.save_dir, 'log.txt'), 'a') as f:
                f.write(test_info)
                f.write('\n')

            # if epoch_idx % args.train_lr_decay_step == 0:
            #     for param in optimizer.param_groups:
            #         param['lr'] = args.train_lr_decay_factor * param['lr']

            # logger.log(test_info, model, optimizer)
            if best_hits < test_hits:
                best_hits = test_hits
                best_epoch = epoch_idx
        test_info = "Training ends. The best testing hits is {:.6f} at epoch {}".format(best_hits, best_epoch)
        print(test_info)
        with open(os.path.join(args.save_dir, 'log.txt'), 'a') as f:
            f.write(test_info)
            f.write('\n')
        logger.print_statistics(run_idx)

    logger.print_statistics()

def config():
    parser = argparse.ArgumentParser(description='OGBL-COLLAB')
    parser.add_argument('--device', type=int, default=0)
    parser.add_argument('--log_steps', type=int, default=1)
    # parser.add_argument('--use_sage', action='store_true')
    # parser.add_argument('--num_layers', type=int, default=3)
    # parser.add_argument('--hidden_feats', type=int, default=256)
    # parser.add_argument('--dropout', type=float, default=0.0)
    parser.add_argument('--batch_size', type=int, default=256)
    parser.add_argument('--lr', type=float, default=1e-3)
    parser.add_argument('--epochs', type=int, default=200)
    parser.add_argument('--eval_steps', type=int, default=1)
    parser.add_argument('--runs', type=int, default=10)

    parser.add_argument('--seed', type=int, default=1234)
    parser.add_argument('--node_labeling_mode', type=str, default='homo')
    parser.add_argument('--hop', type=int, default=2)
    parser.add_argument('--num_workers', type=int, default=16)
    parser.add_argument('--sample_ratio', type=float, default=1.0)
    parser.add_argument('--max_nodes_per_hop', type=int, default=200)
    parser.add_argument('--train_log_interval', type=int, default=100)
    # parser.add_argument('--valid_log_interval', type=int, default=10)
    parser.add_argument('--save_appendix', type=str, default='debug')
    args = parser.parse_args()

    device = f'cuda:{args.device}' if torch.cuda.is_available() else 'cpu'
    args.device = torch.device(device)

    ### set save_dir according to localtime and test mode
    file_dir = os.path.dirname(os.path.realpath('__file__'))
    local_time = time.strftime('%y%m%d%H%M', time.localtime())
    args.save_dir = os.path.join(
        file_dir, 'log/{}_{}'.format(
            args.save_appendix, local_time
        )
    )
    if not os.path.exists(args.save_dir):
        os.makedirs(args.save_dir) 
    print(args)

    # backup current .py files
    for f in glob.glob(r"*.py"):
        copy(f, args.save_dir)

    # save command line input
    cmd_input = 'python3 ' + ' '.join(sys.argv)
    with open(os.path.join(args.save_dir, 'cmd_input.txt'), 'a') as f:
        f.write(cmd_input)
        f.write("\n")
    
    return args

if __name__ == '__main__':
    args = config()
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)
    main(args)
