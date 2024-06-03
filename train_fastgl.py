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


def load_labels(label, output_nodes, train_device):
    output_nodes = output_nodes.to(label.device)
    
    batch_labels = torch.index_select(
        label, 0, output_nodes.long()).to(train_device)

    return batch_labels


def load_subtensors(feat, input_nodes, epoch, step, train_device, input_dim, match=None, last_batch_feat=None):
    if (epoch==0) & (step==0) :
        input_nodes = input_nodes.to(feat.device)
        batch_inputs = torch.index_select(
            feat, 0, input_nodes.long()).to(train_device)
        match.init_cache(input_nodes.long(),feat)
        return batch_inputs
    else:
        # match process
        cacheid, cacheid_batch, nids_in_cpu, gpu_mask, gpu_batch_mask, cpu_mask = match.fetch_cpu_data_id(input_nodes.long().to(feat.device))
        batch_inputs = torch.cuda.FloatTensor(input_nodes.size(0),input_dim)
        if(nids_in_cpu.size(0)>0):
            # fetch data from cpu
            batch_inputs[cpu_mask] = torch.index_select(
                feat, 0, nids_in_cpu.long()).to(train_device)
            # when there is sufficient GPU memory left, use the some GPU memory as cache to maximise the acceleration of memory IO
            if(match.use_cache):
                batch_inputs[gpu_mask] = match.memory[cacheid.long()]
            # reuse the features of overlapping nodes 
            batch_inputs[gpu_batch_mask] = last_batch_feat[cacheid_batch.long()]
            match.fresh_cache(input_nodes.int(),batch_inputs)
        else:
            batch_inputs = match.memory[cacheid.long()]
        return batch_inputs
        


def get_data_iterator(run_config, dataloader):
    if run_config['use_gpu_sampling']:
        return iter(dataloader)
    else:
        if run_config['num_sampling_worker'] > 0 and not run_config['pipelining']:
            return [data for data in iter(dataloader)]
        else:
            return iter(dataloader)


def run(worker_id, run_config, is_presample=True,model='gin',step_n=10,cache_ratio=0,log_name=''):
    torch.set_num_threads(run_config['num_thread'])
    sample_device = torch.device(run_config['sample_devices'][worker_id])
    train_device = torch.device(run_config['train_devices'][worker_id])
    num_worker = run_config['num_worker']

    if num_worker > 1:
        dist_init_method = 'tcp://{master_ip}:{master_port}'.format(
            master_ip='127.0.0.1', master_port='25003')
        world_size = num_worker
        torch.distributed.init_process_group(backend="nccl",
                                             init_method=dist_init_method,
                                             world_size=world_size,
                                             rank=worker_id,
                                             timeout=datetime.timedelta(seconds=get_default_timeout()))
    
    dataset = run_config['dataset']
    g_raw = dataset[0]
    
    # prepare graph
    g = dgl.graph((g_raw.edges()[0],g_raw.edges()[1]))
    in_deg = g.in_degrees().int().to(train_device)

    edge_dst = g_raw.edges()[1].int()
    edge_src = g_raw.edges()[0].int()
    edge_dst,index = edge_dst.sort()
    edge_src = edge_src[index].to(train_device)
    edge_dst = edge_dst.to(train_device)
    
    # transform to csr for sampling
    edge_ptr = torch.cumsum(in_deg,dim=0).int()
    zero_tensor = edge_ptr.new_zeros((1,)).int()
    edge_ptr = torch.cat((zero_tensor,edge_ptr),dim=0)
    
    
    feat = g_raw.ndata['feat']
    label = g_raw.ndata['label'].long()
    train_nids = torch.where(g_raw.ndata['train_mask'])[0].to(train_device)
    in_feats = feat.size(1)
    nan_index = torch.isnan(g_raw.ndata['label'])
    n_classes = int(g_raw.ndata['label'][~nan_index].max().item() + 1)

    sampler = FusedMapSampler(run_config['fanout'],replace=False)
    sampler.edge_dst = edge_dst
    sampler.edge_src = edge_src
    sampler.edge_ptr = edge_ptr.int()
    sampler.nodes_num = g_raw.num_nodes()
    # g is only used as a padding argument
    g = dgl.graph((torch.tensor([0]),torch.tensor([1]))).to(train_device)
    # reuse the dataloader of dgl
    dataloader = dgl.dataloading.DataLoader(
        g,
        train_nids,
        sampler,
        use_ddp=num_worker > 1,
        batch_size=run_config['batch_size'],
        shuffle=True,
        drop_last=False,
        prefetch_factor=run_config['prefetch_factor'],
        num_workers=run_config['num_sampling_worker'],
        )
    
    gat_dim = 8
    # 100MB显存
    if(model=='gcn'):
        acc_model = GCN(in_feats, run_config['num_hidden'],
                    n_classes, run_config['num_layer'], F.relu, True, run_config['dropout'])
    elif(model=='gin'):
        acc_model = GIN(in_feats, run_config['num_hidden'],
                    n_classes, run_config['num_layer'], F.relu, True, run_config['dropout'])
    elif(model=='gat'):
        acc_model = GAT(in_feats, gat_dim,
                    n_classes, run_config['num_layer'], F.relu, run_config['dropout'])
    else:
        raise RuntimeError('no model')
    acc_model = acc_model.to(train_device)

    loss_fcn = nn.CrossEntropyLoss()
    loss_fcn = loss_fcn.to(train_device)
    optimizer = optim.Adam(
        acc_model.parameters(), lr=run_config['lr'], weight_decay=run_config['weight_decay'])
    num_epoch = run_config['num_epoch']

    acc_model.train()

    epoch_sample_times = []
    epoch_graph_copy_times = []
    epoch_copy_times = []
    epoch_train_times = []
    epoch_total_times = []
    epoch_num_nodes = []
    epoch_num_samples = []

    sample_times = []
    graph_copy_times = []
    copy_times = []
    train_times = []
    total_times = []
    num_nodes = []
    num_samples = []
    epoch_back_times = []
    
    node_num = g_raw.num_nodes()
    cache = None

    batch_inputs = None
    
    steps = step_n
    
    epoch_steps = math.ceil(train_nids.numel()/run_config['batch_size']/num_worker)
    
    steps = math.ceil(train_nids.numel()/run_config['batch_size']/num_worker)
    for epoch in range(num_epoch):
        epoch_sample_time = 0.0
        epoch_graph_copy_time = 0.0
        epoch_copy_time = 0.0
        epoch_train_time = 0.0
        epoch_num_node = 0
        epoch_num_sample = 0
        epoch_back_time = 0

        tic = time.time()
        if(epoch==0):
            high_nid = pre_sample(dataloader=dataloader,node_num=node_num)
        
        t0 = time.time()
        # init the match_reorder operation
        match = match_reorder(num_nodes=node_num,steps=steps,device=train_device)
        
        if(run_config['is_reorder']==False):
            n_pack = 1
        else:
            n_pack = math.ceil(epoch_steps/steps)
        
        for i in range(n_pack):
            if(steps>= epoch_steps):
                match.steps = epoch_steps
            else:
                if(i==(n_pack-1)):
                    match.steps = epoch_steps % steps
            if(run_config['is_reorder']==True):
                batch_list_re = match.reorder(dataloader=dataloader) 
                
            batch_list = batch_list_re if run_config['is_reorder']==True else dataloader
        # # reorder
        # batch_list = match.reorder(dataloader=dataloader) 
            
        
            for step, (input_nodes, output_nodes, blocks) in enumerate(batch_list):
                if not run_config['pipelining']:
                    event_sync()
                t1 = time.time()
                if((step==0)&(epoch==0)):
                    match_mem = MatchServer(node_num,train_device,high_nid,cache_ratio)
                # graph are copied to GPU here
                blocks = [block.int().to(train_device) for block in blocks]
                t2 = time.time()
                batch_labels = load_labels(
                    label, output_nodes, train_device)
                batch_inputs = load_subtensors(
                    feat, input_nodes, epoch, step, train_device, input_dim=feat.size(1), 
                    match=match_mem, last_batch_feat=batch_inputs
                )
                if not run_config['pipelining']:
                    event_sync()
                t3 = time.time()
                # Compute loss and prediction
                batch_pred = acc_model(blocks, batch_inputs, neighbor_list=run_config['fanout'])
                t4 = time.time()
                loss = loss_fcn(batch_pred, batch_labels)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()            

                if not run_config['pipelining']:
                    event_sync()
                
                t5 = time.time()

                num_samples.append(sum([block.num_edges() for block in blocks]))
                num_nodes.append(blocks[0].num_src_nodes())

                batch_labels = None
                blocks = None


                sample_times.append(t1 - t0)
                graph_copy_times.append(t2 - t1)
                copy_times.append(t3 - t1)
                train_times.append(t4 - t3)
                total_times.append(t4 - t0)

                epoch_sample_time += sample_times[-1]
                epoch_graph_copy_time += graph_copy_times[-1]
                epoch_copy_time += copy_times[-1]
                epoch_train_time += train_times[-1]
                epoch_num_node += num_nodes[-1]
                epoch_num_sample += num_samples[-1]
                epoch_back_time += (t5-t4)

                if worker_id == 0:
                    print('Epoch {:05d} | Step {:05d} | Nodes {:.0f} | Samples {:.0f} | Time {:.4f} | Sample Time {:.4f} | Graph copy {:.4f} | Copy Time {:.4f} | Train time {:4f} |  Loss {:.4f} '.format(
                        epoch, step, np.mean(num_nodes), np.mean(num_samples), np.mean(total_times), np.mean(t1-t0), np.mean(graph_copy_times), np.mean(t3-t1), t4-t3, loss))
                t0 = time.time()
            event_sync()

        if num_worker > 1:
            torch.distributed.barrier()

        toc = time.time()

        epoch_sample_times.append(epoch_sample_time)
        epoch_graph_copy_times.append(epoch_graph_copy_time)
        epoch_copy_times.append(epoch_copy_time)
        epoch_train_times.append(epoch_train_time)
        epoch_total_times.append(toc - tic)
        epoch_num_samples.append(epoch_num_sample)
        epoch_num_nodes.append(epoch_num_node)
        epoch_back_times.append(epoch_back_time)

    if num_worker > 1:
        torch.distributed.barrier()

    print('[Worker {:d}({:s})] Avg Epoch Time {:.4f} | Avg Nodes {:.0f} | Avg Samples {:.0f} | Sample Time {:.4f} | Graph copy {:.4f} | Copy Time {:.4f} | Train Time {:.4f}'.format(
        worker_id, torch.cuda.get_device_name(train_device), np.mean(epoch_total_times[1:]), np.mean(epoch_num_nodes), np.mean(epoch_num_samples), np.mean(epoch_sample_times[1:]), np.mean(epoch_graph_copy_times[1:]), np.mean(epoch_copy_times[1:]), np.mean(epoch_train_times[1:])))

    if num_worker > 1:
        torch.distributed.barrier()

    if worker_id == 0:
        test_result = {}
        test_result['sample_time'] = np.mean(epoch_sample_times[1:])
        test_result['copy_time'] = np.mean(epoch_copy_times[1:])
        test_result['forward_time'] = np.mean(epoch_train_times[1:])
        test_result['backward_time'] = np.mean(epoch_back_times[1:])
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
    
def test_func(mat):
    return 2*mat

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
    parser.add_argument("--model",type=str,default='gcn',)
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
    parser.add_argument("--cache_ratio",type=int,default=1,)
    parser.add_argument("--is-reorder",type=bool,default=False,)
    parser.add_argument("--is-presample",type=bool,default=True,)
    parser.add_argument("--dataset_name",type=str,default='reddit',)
    parser.add_argument("--dataset_path",type=str,default='../dataset',)
    parser.add_argument("--log_path",type=str,default='logs_fig9/',)
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
        run(0, run_config,args.is_presample,args.model,args.step_n,args.cache_ratio,log_name)
    else:
        mp.spawn(run, args=(run_config,args.is_presample,args.model,args.step_n,args.cache_ratio,log_name), nprocs=num_worker, join=True)

