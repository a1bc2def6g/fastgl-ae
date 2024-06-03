#*************************************************************************
#   > Filename    : train.py
#   > Description : The train function for gcn, gin and gat on various datasets
#*************************************************************************
import argparse
import datetime
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import dgl
import time
import numpy as np
import torch.multiprocessing as mp
import math
import sys
from common_config import *
from dgl.data import AsNodePredDataset
from utilis import *
from MatchReorderServer import *
from GraphConv import *
from FusedMapSampler import *
from dgl_model import *





def parse_args(default_run_config):
    argparser = argparse.ArgumentParser("GCN Training")
    argparser.add_argument('--use-gpu-sampling', action='store_true',
                           default=default_run_config['use_gpu_sampling'])
    argparser.add_argument('--no-use-gpu-sampling',
                           dest='use_gpu_sampling', action='store_false')
    argparser.add_argument('--devices', nargs='+',
                           type=int, default=default_run_config['devices'])
    argparser.add_argument('--dataset', type=str,
                           default=default_run_config['dataset'])
    argparser.add_argument('--pipelining', action='store_true',
                           default=default_run_config['pipelining'])
    argparser.add_argument(
        '--no-pipelining', dest='pipelining', action='store_false',)
    argparser.add_argument('--num-sampling-worker', type=int,
                           default=default_run_config['num_sampling_worker'])

    argparser.add_argument('--fanout', nargs='+',
                           type=int, default=default_run_config['fanout'])
    argparser.add_argument('--num-epoch', type=int,
                           default=default_run_config['num_epoch'])
    argparser.add_argument('--num-hidden', type=int,
                           default=default_run_config['num_hidden'])
    argparser.add_argument('--batch-size', type=int,
                           default=default_run_config['batch_size'])
    argparser.add_argument(
        '--lr', type=float, default=default_run_config['lr'])
    argparser.add_argument('--dropout', type=float,
                           default=default_run_config['dropout'])
    argparser.add_argument('--weight-decay', type=float,
                           default=default_run_config['weight_decay'])

    argparser.add_argument('--validate-configs',
                           action='store_true', default=False)
    
    argparser.add_argument('--is-reorder', type=bool,
                           default=default_run_config['is_reorder'])

    return vars(argparser.parse_args())


def get_run_config(is_reorder,args,dataset):
    # default_run_config = {}
    # default_run_config['use_gpu_sampling'] = True
    # default_run_config['devices'] = args.devices
    # default_run_config['dataset'] = 'products'
    # default_run_config['pipelining'] = False
    # default_run_config['num_sampling_worker'] = 0
    # # default_run_config['num_sampling_worker'] = 16

    # # The sampled numbers for each layer
    # default_run_config['fanout'] = args.fanout
    # default_run_config['num_epoch'] = args.num_epoch
    # default_run_config['num_hidden'] = args.num_hidden
    # default_run_config['batch_size'] = args.batch_size
    # default_run_config['lr'] = 0.01
    # default_run_config['dropout'] = 0.5
    # default_run_config['weight_decay'] = 0.0005
    
    # default_run_config['is_reorder'] = is_reorder
    

    # run_config = parse_args(default_run_config)
    run_config = vars(args)

    assert(len(run_config['devices']) > 0)
    assert(run_config['num_sampling_worker'] >= 0)

    # The first epoch is used to warm up the system
    run_config['num_epoch'] += 1
    run_config['num_worker'] = len(run_config['devices'])
    run_config['num_fanout'] = run_config['num_layer'] = len(
        run_config['fanout'])
    run_config['num_sampling_worker'] = run_config['num_sampling_worker'] // run_config['num_worker']

    if(not hasattr(dataset,'train_idx')):
        num_train_set = dataset[0].ndata['train_mask'].size(0)
    else:
        num_train_set = dataset.train_idx.size(0)

    # [prefetch_factor]: number of samples loaded in advance by each worker.
    # 2 means there will be a total of 2 * num_workers samples prefetched across all workers. (default: 2)
    # DGL uses a custom dataset, it makes PyTorch thinks a batch is a sample.
    if run_config['pipelining'] == False and run_config['num_sampling_worker'] > 0:
        # make it sequential. sample all the batch before training.
        # assumed that drop last = False
        num_samples_per_epoch = math.ceil(
            num_train_set / run_config['num_worker'])
        num_batch_per_epoch = math.ceil(
            num_samples_per_epoch / run_config['batch_size'])
        run_config['num_prefetch_batch'] = num_batch_per_epoch
        run_config['prefetch_factor'] = math.ceil(
            num_batch_per_epoch / run_config['num_sampling_worker'])
    else:
        # default prefetch factor is 2
        run_config['prefetch_factor'] = 2

    if run_config['use_gpu_sampling']:
        run_config['sample_devices'] = run_config['devices']
        run_config['train_devices'] = run_config['devices']
        #  GPU sampling requires sample_device to be 0
        run_config['num_sampling_worker'] = 0
        # default prefetch factor is 2
        run_config['prefetch_factor'] = 2
    else:
        run_config['sample_devices'] = ['cpu' for _ in run_config['devices']]
        run_config['train_devices'] = run_config['devices']

    run_config['num_thread'] = torch.get_num_threads(
    ) // run_config['num_worker']

    print('config:eval_tsp="{:}"'.format(time.strftime(
        "%Y-%m-%d %H:%M:%S", time.localtime())))
    for k, v in run_config.items():
        print('config:{:}={:}'.format(k, v))

    run_config['dataset'] = dataset

    if run_config['validate_configs']:
        sys.exit()

    return run_config


def get_data_iterator(run_config, dataloader):
    if run_config['use_gpu_sampling']:
        return iter(dataloader)
    else:
        if run_config['num_sampling_worker'] > 0 and not run_config['pipelining']:
            return [data for data in iter(dataloader)]
        else:
            return iter(dataloader)


def run(worker_id, run_config, is_presample=True,model='gin',step_n=10,log_name=''):
    torch.set_num_threads(run_config['num_thread'])
    sample_device = torch.device(run_config['sample_devices'][worker_id])
    train_device = torch.device(run_config['train_devices'][worker_id])
    num_worker = run_config['num_worker']

    if num_worker > 1:
        dist_init_method = 'tcp://{master_ip}:{master_port}'.format(
            master_ip='127.0.0.1', master_port='25000')
        world_size = num_worker
        torch.distributed.init_process_group(backend="nccl",
                                             init_method=dist_init_method,
                                             world_size=world_size,
                                             rank=worker_id,
                                             timeout=datetime.timedelta(seconds=get_default_timeout()))
    
    dataset = run_config['dataset']
    g_raw = dataset[0]
    g = dgl.graph((g_raw.edges()[0],g_raw.edges()[1])).to(train_device)
    feat = g_raw.ndata['feat'].to(train_device)
    label = g_raw.ndata['label'].long()
    train_nids = torch.where(g_raw.ndata['train_mask'])[0].to(train_device)
    test_nids = torch.where(g_raw.ndata['test_mask'])[0].to(train_device)
    in_feats = feat.size(1)
    nan_index = torch.isnan(g_raw.ndata['label'])
    n_classes = int(g_raw.ndata['label'][~nan_index].max().item() + 1)
    # n_classes = dataset.num_classes
    # n_classes = 8

    sampler = DGLNeighborSampler(run_config['fanout'],replace=False)
    # sampler = PinSAGESampler(g, 3, 0.5, 4, 5, 3)
    
    dataloader = dgl.dataloading.DataLoader(
        g,
        train_nids,
        sampler,
        use_ddp=num_worker > 1,
        # use_ddp=False,
        # device=train_device,
        batch_size=run_config['batch_size'],
        shuffle=True,
        drop_last=False,
        prefetch_factor=run_config['prefetch_factor'],
        num_workers=run_config['num_sampling_worker'],
        # use_uva=True
        # num_workers = 0
        )
    num_epoch = run_config['num_epoch']

    # model.train()



    
    # construct gnn model
    gat_head_dim = 8
    
    
    if(model=='gcn'):
        test_nn = GCN(in_feats, run_config['num_hidden'],
                    n_classes, run_config['num_layer'], F.relu, run_config['dropout'])
    elif(model=='gin'):
        test_nn = GIN(in_feats, run_config['num_hidden'],
                    n_classes, run_config['num_layer'], F.relu, run_config['dropout'])
    elif(model=='gat'):
        test_nn = GAT(in_feats, gat_head_dim,
                    n_classes, run_config['num_layer'], F.relu, run_config['dropout'], heads=8)
    else:
        raise RuntimeError('no model')
    test_nn = test_nn.to(train_device)
    loss_fcn = nn.CrossEntropyLoss()
    loss_fcn = loss_fcn.to(train_device)
    
    optimizer = optim.Adam(
        test_nn.parameters(), lr=run_config['lr'], weight_decay=run_config['weight_decay'])
    test_nn.train()
    
    epoch_dgl_forward_time = []
    epoch_dgl_backward_time = []
    epoch_copy_time = []
    epoch_sample_time = []
    
    time_dgl_forward_end = 0
    
    dgl_loss = []
    acc_loss = []
    
    if worker_id==0:
        print(test_nn)
        
    iteration = 0
    total_nodes = 0
    # total_nodes_list = []
    for epoch in range(num_epoch):

        dgl_forward_time = 0
        dgl_backward_time = 0
        copy_time = 0
        sample_time = 0
        t0 = time.time()
        for step, (input_nodes, output_nodes, blocks) in enumerate(iter(dataloader)):
            total_nodes += input_nodes.size()[0]
            iteration += 1
            t_sample_end = time.time()
            t_copy_start = time.time()
            input_feat = torch.index_select(
                feat, 0, input_nodes.to(feat.device).long()).to(train_device)
            batch_labels = torch.index_select(
                label, 0, output_nodes.to(label.device)).to(train_device) 
            if not run_config['pipelining']:
                event_sync()
            time_forward_start = time.time()
            out = test_nn(blocks,input_feat)
            event_sync()
            time_dgl_forward_end = time.time()
            loss = loss_fcn(out, batch_labels)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            torch.cuda.empty_cache()
            event_sync()
            time_dgl_end = time.time()
            if not run_config['pipelining']:
                event_sync()
            
           
            
            dgl_forward_time += (time_dgl_forward_end - time_forward_start)
            dgl_backward_time += (time_dgl_end - time_dgl_forward_end)
            copy_time += (time_forward_start - t_copy_start)
            sample_time += (t_sample_end - t0)
            
            if not run_config['pipelining']:
                event_sync()
            t1 = time.time()
            t0 = time.time()
            if worker_id == 0:
                print("step:",step,"loss_dgl:",loss.cpu().item())
            
            
        event_sync()
        epoch_dgl_forward_time.append(dgl_forward_time)
        epoch_dgl_backward_time.append(dgl_backward_time)
        epoch_copy_time.append(copy_time)
        epoch_sample_time.append(sample_time)
        

        if num_worker > 1:
            torch.distributed.barrier()

        toc = time.time()

    if num_worker > 1:
        torch.distributed.barrier()

    
    if num_worker > 1:
        torch.distributed.barrier()
    if worker_id == 0:
        test_result = {}
        test_result['sample_time'] = np.mean(epoch_sample_time[1:])
        test_result['copy_time'] = np.mean(epoch_copy_time[1:])
        test_result['forward_time'] = np.mean(epoch_dgl_forward_time[1:])
        test_result['backward_time'] = np.mean(epoch_dgl_backward_time[1:])
        test_result['epoch_time'] = test_result['sample_time'] + test_result['copy_time'] +  test_result['forward_time'] + test_result['backward_time']
        for k, v in test_result.items():
            print('test_result:{:}={:.5f}'.format(k, v))
        print('finish')
        
        with open(log_name,'w') as f:#dict2txt
            for k,v in test_result.items():
                f.write('%s:%s\n'%(k, v))

def setup_seed(seed):
    import random
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True
    


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--pipelining', action='store_true',
                           default=False)
    parser.add_argument('--use-gpu-sampling', action='store_true',
                           default=True)
    parser.add_argument('--num-sampling-worker', type=int,
                           default=0)
    parser.add_argument('--validate-configs',
                           action='store_true', default=False)
    parser.add_argument('--devices', nargs='+',
                           type=int, default=[0,1])
    parser.add_argument("--step-n",type=int,default=5,)
    parser.add_argument("--model",type=str,default='gat',)
    parser.add_argument(
        '--lr', type=float, default=0.01)
    parser.add_argument('--dropout', type=float,
                           default=0.5)
    parser.add_argument('--weight-decay', type=float,
                           default=0.0005)
    
    parser.add_argument('--fanout', nargs='+',
                           type=int, default=[5, 10, 15])
    parser.add_argument("--num_epoch",type=int,default=3,)
    parser.add_argument("--num_hidden",type=int,default=64,)
    parser.add_argument("--batch_size",type=int,default=8000,)
    parser.add_argument("--is-reorder",type=bool,default=False,)
    parser.add_argument("--is-presample",type=bool,default=True,)
    parser.add_argument("--dataset_name",type=str,default='reddit',)
    parser.add_argument("--dataset_path",type=str,default='../dataset',)
    parser.add_argument("--log_path",type=str,default='dgl_logs_fig9/',)
    args = parser.parse_args()
    setup_seed(12345)
    layers = len(args.fanout)
    dev_num = len(args.devices)
    log_name = args.log_path+args.model+'_'+args.dataset_name+'_'+str(layers)+'_'\
        +str(args.num_hidden)+'_'+str(args.batch_size)+'_'+str(dev_num)+'.txt'
    if(args.dataset_name in ['ogbn-arxiv','ogbn-products','ogbn-papers100M']):
        dataset = AsNodePredDataset(
            DglNodePropPredDataset(args.dataset_name, root=args.dataset_path)
        )
    elif(args.dataset_name in ['reddit']):
        from dgl.data import RedditDataset
        dataset = RedditDataset(raw_dir=args.dataset_path,verbose=True)
    elif(args.dataset_name in ['mag']):
        from dgl.data.utils import load_graphs
        dataset = load_graphs(args.dataset_path+'/mag_coarse.bin')[0]
    elif(args.dataset_name in ['igb-large']):
        from dgl.data.utils import load_graphs
        dataset = load_graphs(args.dataset_path+'/igb_large.bin')[0]
    run_config = get_run_config(args.is_reorder,args,dataset=dataset,)
    num_worker = run_config['num_worker']
    
    if num_worker == 1:
        run(0, run_config,args.is_presample,args.model,args.step_n,log_name)
    else:
        mp.spawn(run, args=(run_config,args.is_presample,args.model,args.step_n,log_name), nprocs=num_worker, join=True)

