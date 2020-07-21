import os
import time
import random
import argparse
import numpy as np

import torch
from torch.utils.data import DataLoader
from ogb.linkproppred import DglLinkPropPredDataset, Evaluator
import dgl

from logger import Logger

from dataset import IGMCDataset, RandomDataset, collate_igmc
from model import IGMC
from utils import MetricLogger

os.environ['TZ'] = 'Asia/Shanghai'
time.tzset()

@torch.no_grad()
def test_split(model, device, edges, train_graph, args):
    model.eval()

    dataset = IGMCDataset(
        edges, train_graph, 
        args.hop, args.sample_ratio, args.max_nodes_per_hop)
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True, 
                              num_workers=args.num_workers, collate_fn=collate_igmc)
    preds = []
    for batch in loader:
        inputs = batch.to(device)
        preds += [model(inputs).squeeze().cpu()]
    return preds

def test(model, loss_fn, device, 
        edge_split, train_graph, args):
    print("\n=== start testing on pos_train_edges... ===\n")
    pos_train_preds = test_split(model, device, edge_split['train']['edge'], train_graph, args)
    print("\n=== start testing on pos_valid_edges... ===\n")
    pos_valid_preds = test_split(model, device, edge_split['valid']['edge'], train_graph, args)
    print("\n=== start testing on neg_valid_edges... ===\n")
    neg_valid_preds = test_split(model, device, edge_split['valid']['edge_neg'], train_graph, args)
    print("\n=== start testing on pos_test_edges... ===\n")
    pos_test_preds = test_split(model, device, edge_split['test']['edge'], train_graph, args)
    print("\n=== start testing on neg_test_edges... ===\n")
    neg_test_preds = test_split(model, device, edge_split['test']['edge_neg'], train_graph, args)

    results = {}
    for K in [10, 50, 100]:
        evaluator.K = K
        train_hits = evaluator.eval({
            'y_pred_pos': pos_train_pred,
            'y_pred_neg': neg_valid_pred,
        })[f'hits@{K}']
        valid_hits = evaluator.eval({
            'y_pred_pos': pos_valid_pred,
            'y_pred_neg': neg_valid_pred,
        })[f'hits@{K}']
        test_hits = evaluator.eval({
            'y_pred_pos': pos_test_pred,
            'y_pred_neg': neg_test_pred,
        })[f'hits@{K}']

        results[f'Hits@{K}'] = (train_hits, valid_hits, test_hits)

    return results

# @profile
def train_epoch(model, loss_fn, optimizer, device, log_interval, 
                pos_train_edge, train_graph, args):
    model.train()

    train_dataset = IGMCDataset(
        pos_train_edge, train_graph, 
        args.hop, args.sample_ratio, args.max_nodes_per_hop)
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, 
                              num_workers=args.num_workers, collate_fn=collate_igmc)

    random_dataset = RandomDataset(
        len(pos_train_edge), train_graph, 
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
    return total_loss / total_examples # len(loader.dataset)

def main(args):
    raw_dataset = DglLinkPropPredDataset(name='ogbl-collab')
    train_graph = dgl.as_heterograph(raw_dataset[0])
    # there is one self_loop edge in valid_neg and test_neg separately, remove it
    edge_split = raw_dataset.get_edge_split()
    edge = edge_split['valid']['edge_neg']
    edge_split['valid']['edge_neg'] = edge[edge[:, 0]!=edge[:, 1]]
    edge = edge_split['test']['edge_neg']
    edge_split['test']['edge_neg'] = edge[edge[:, 0]!=edge[:, 1]]

    model = IGMC(args.hop+1).to(args.device)

    evaluator = Evaluator(name='ogbl-collab')
    loggers = {
        'Hits@10': Logger(args.runs, args),
        'Hits@50': Logger(args.runs, args),
        'Hits@100': Logger(args.runs, args),
    }

    # save_logger = MetricLogger(args.save_dir, args.valid_log_interval)

    # for run_idx in range(1, args.runs+1):
    # model.reset_parameters()
    loss_fn = torch.nn.BCELoss().to(args.device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

    for epoch_idx in range(1, args.epochs+1):
        train_loss = train_epoch(model, loss_fn, optimizer, 
                                args.device, args.train_log_interval, 
                                edge_split['train']['edge'], train_graph, args)

        if epoch_idx % args.eval_steps == 0:
            results = test(model, loss_fn, args.device,
                           edge_split, train_graph, args)
            for key, result in results.items():
                loggers[key].add_result(run, result)
            
            if epoch_idx % args.log_steps == 0:
                for key, result in results.items():
                    train_hits, valid_hits, test_hits = result
                    print(key)
                    print(f'Run: {run + 1:02d}, '
                            f'Epoch: {epoch:02d}, '
                            f'Loss: {loss:.4f}, '
                            f'Train: {100 * train_hits:.2f}%, '
                            f'Valid: {100 * valid_hits:.2f}%, '
                            f'Test: {100 * test_hits:.2f}%')
                print('---')

    for key in loggers.keys():
        print(key)
        loggers[key].print_statistics(run)

    # for key in loggers.keys():
    #     print(key)
    #     loggers[key].print_statistics()

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
    parser.add_argument('--hop', type=int, default=3)
    parser.add_argument('--num_workers', type=int, default=16)
    parser.add_argument('--sample_ratio', type=float, default=1.0)
    parser.add_argument('--max_nodes_per_hop', type=int, default=200)
    parser.add_argument('--train_log_interval', type=int, default=100)
    parser.add_argument('--valid_log_interval', type=int, default=10)
    args = parser.parse_args()

    print(args)
    device = f'cuda:{args.device}' if torch.cuda.is_available() else 'cpu'
    args.device = torch.device(device)
    return args

if __name__ == '__main__':
    args = config()
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)
    main(args)
