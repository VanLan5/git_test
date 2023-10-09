from __future__ import print_function
from scipy.signal import savgol_filter # 轨迹平滑
import warnings
warnings.filterwarnings("ignore")
import torch
from model import highwayNet
from utils import ngsimDataset_ZLY,maskedNLL,maskedMSETest,maskedNLLTest,maskedMSE
from torch.utils.data import DataLoader
import time
import math
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "2"
import setproctitle
setproctitle.setproctitle('fjs_GPU2_csp_train')

## Network Arguments
args = {}
args['use_cuda'] = True                  # 并行计算
args['encoder_size'] = 64                # 编码
args['decoder_size'] = 128               # 解码
args['in_length'] = 20                   # 输入序列长度
args['out_length'] = 20                  # 输出序列长度
args['grid_size'] = (13,3)               # 在被预测的车辆周围定义了一个 13 × 3的空间网格，其中每一列对应一个车道，并且行之间的距离为 15 英尺，大约等于一辆车的长度
args['soc_conv_depth'] = 64              # 卷积1
args['conv_3x1_depth'] = 16              # 卷积2
args['dyn_embedding_size'] = 32          # 机动层
args['input_embedding_size'] = 32        #
args['num_lat_classes'] = 3              # 横向特征
args['num_lon_classes'] = 2              # 纵向特征
args['use_maneuvers'] = True             # 使用特征
args['train_flag'] = True                #


# Initialize network
net = highwayNet(args)
if args['use_cuda']:
    net = net.cuda()


## Initialize optimizer
pretrainEpochs = 5
trainEpochs = 100   # 30
optimizer = torch.optim.Adam(net.parameters()) # 学习率默认：1e-3
batch_size = 512  # 256
crossEnt = torch.nn.BCELoss() # 计算目标值和预测值之间的二进制交叉熵损失函数


## Initialize data loaders
trSet = ngsimDataset_ZLY('/data/traj_prediction/data_input/TrainData_1.mat', t_h=args['in_length'], t_f=args['out_length'])
# print(len(trSet)) 3982089
valSet= ngsimDataset_ZLY('/data/traj_prediction/data_input/ValData_1.mat', t_h=args['in_length'], t_f=args['out_length'])
# print(len(valSet)) 36199
trDataloader = DataLoader(trSet,batch_size=batch_size,shuffle=True,num_workers=8,collate_fn=trSet.collate_fn)
valDataloader = DataLoader(valSet,batch_size=batch_size,shuffle=True,num_workers=8,collate_fn=valSet.collate_fn)

## Variables holding train and validation loss values:
train_loss = []
val_loss = []
prev_val_loss = math.inf

for epoch_num in range(pretrainEpochs+trainEpochs):
    if epoch_num == 30:
        torch.save(net.state_dict(), '/data/traj_prediction/result/cslstm20_1_'+"_"+str(epoch_num)+'.tar')
    if epoch_num == 50:
        torch.save(net.state_dict(), '/data/traj_prediction/result/cslstm20_1_'+"_"+str(epoch_num)+'.tar')
    if epoch_num == 0:
        print('Pre-training with MSE loss')
    elif epoch_num == pretrainEpochs:
        print('Training with NLL loss')


    ## Train:_________________________________________________________________________________________________________________________________________________________________________________________________________________________________________________________________________________________________
    net.train_flag = True

    # Variables to track training performance:
    avg_tr_loss = 0
    avg_tr_time = 0
    avg_lat_acc = 0
    avg_lon_acc = 0

    for i, data in enumerate(trDataloader):
        # i表示第几个batch， data表示该batch对应的数据
        st_time = time.time()
        hist, nbrs, mask, lat_enc, lon_enc, fut, op_mask = data
        # hist 21*512*2 tensor 21最后一行为当前帧
        # nbrs 21*1132*2 tensor
        # mask 512*3*13*64 False
        # lat_enc 512*3 [1,0,0]
        # lon_enc 512*2 [1,0]
        # fut 20*512*2 tensor 
        # op_mask 20*512*2 全为1

        # fut_pred 20*512*2 tensor 

        # # 对历史轨迹和未来轨迹平滑处理
        # hist_sf = hist.numpy()
        # for m in range(hist.shape[2]): # hist 21*512*2 tensor
        #     for k in range(hist.shape[1]):
        #         hist_sf[:(hist.shape[0]-1), k, m] = torch.Tensor(savgol_filter(hist[:(hist.shape[0]-1), k, m].tolist(), 19, 8))       
        # hist = torch.tensor(hist_sf)
        # fut_sf = fut.numpy()
        # for m in range(fut.shape[2]): # fut 20*512*2 tensor
        #     for k in range(fut.shape[1]):
        #         fut_sf[:(fut.shape[0]), k, m] = torch.Tensor(savgol_filter(fut[:(fut.shape[0]), k, m].tolist(), 19, 8)) 
        # fut = torch.tensor(fut_sf)

        if args['use_cuda']:
            hist = hist.cuda()
            nbrs = nbrs.cuda()
            mask = mask.bool().cuda()
            lat_enc = lat_enc.cuda()
            lon_enc = lon_enc.cuda()
            fut = fut.cuda()
            op_mask = op_mask.cuda()

        # Forward pass
        if args['use_maneuvers']:
            # print('input_size:', hist.size())  # args['in_length'] = 20 --> input_size: torch.Size([21, 512, 2])
            # 输出未来轨迹，制动状态，换道状态
            fut_pred, lat_pred, lon_pred = net(hist, nbrs, mask, lat_enc, lon_enc)
            # Pre-train with MSE loss to speed up training
            if epoch_num < pretrainEpochs:
                l = maskedMSE(fut_pred, fut, op_mask)
            else:
            # Train with NLL loss
                l = maskedNLL(fut_pred, fut, op_mask) + crossEnt(lat_pred, lat_enc) + crossEnt(lon_pred, lon_enc)
                avg_lat_acc += (torch.sum(torch.max(lat_pred.data, 1)[1] == torch.max(lat_enc.data, 1)[1])).item() / lat_enc.size()[0]
                avg_lon_acc += (torch.sum(torch.max(lon_pred.data, 1)[1] == torch.max(lon_enc.data, 1)[1])).item() / lon_enc.size()[0]
        else:
            fut_pred = net(hist, nbrs, mask, lat_enc, lon_enc)
            if epoch_num < pretrainEpochs:
                l = maskedMSE(fut_pred, fut, op_mask)
            else:
                l = maskedNLL(fut_pred, fut, op_mask)

        # Backprop and update weights
        optimizer.zero_grad()
        l.backward()
        a = torch.nn.utils.clip_grad_norm_(net.parameters(), 10)
        optimizer.step()

        # Track average train loss and average train time:
        batch_time = time.time()-st_time
        avg_tr_loss += l.item()
        avg_tr_time += batch_time

        if i%100 == 99: # 100个batch记录一次
            eta = avg_tr_time/100*(len(trSet)/batch_size-i)
            print("Epoch no:",epoch_num+1,"| Epoch progress(%):",format(i/(len(trSet)/batch_size)*100,'0.2f'), "| Avg train loss:",format(avg_tr_loss/100,'0.4f'),"| Acc:",format(avg_lat_acc,'0.4f'),format(avg_lon_acc,'0.4f'), "| Validation loss prev epoch",format(prev_val_loss,'0.4f'), "| ETA(s):",int(eta))
            train_loss.append(avg_tr_loss/100)
            avg_tr_loss = 0
            avg_lat_acc = 0
            avg_lon_acc = 0
            avg_tr_time = 0
    # _________________________________________________________________________________________________________________________________________________________________________________________________________________________________________________________________________________________________________



    ## Validate:______________________________________________________________________________________________________________________________________________________________________________________________________________________________________________________________________________________________
    net.train_flag = False

    print("Epoch",epoch_num+1,'complete. Calculating validation loss...')
    avg_val_loss = 0
    avg_val_lat_acc = 0
    avg_val_lon_acc = 0
    val_batch_count = 0
    total_points = 0

    for i, data  in enumerate(valDataloader):
        st_time = time.time()
        hist, nbrs, mask, lat_enc, lon_enc, fut, op_mask = data


        if args['use_cuda']:
            hist = hist.cuda()
            nbrs = nbrs.cuda()
            mask = mask.cuda()
            lat_enc = lat_enc.cuda()
            lon_enc = lon_enc.cuda()
            fut = fut.cuda()
            op_mask = op_mask.cuda()
        # print('val_input_size:', hist.size())
        # Forward pass
        if args['use_maneuvers']:
            if epoch_num < pretrainEpochs:
                # During pre-training with MSE loss, validate with MSE for true maneuver class trajectory
                net.train_flag = True
                fut_pred, _ , _ = net(hist, nbrs, mask, lat_enc, lon_enc)
                l = maskedMSE(fut_pred, fut, op_mask)
            else:
                # During training with NLL loss, validate with NLL over multi-modal distribution
                fut_pred, lat_pred, lon_pred = net(hist, nbrs, mask, lat_enc, lon_enc)
                l = maskedNLLTest(fut_pred, lat_pred, lon_pred, fut, op_mask,avg_along_time = True)
                avg_val_lat_acc += (torch.sum(torch.max(lat_pred.data, 1)[1] == torch.max(lat_enc.data, 1)[1])).item() / lat_enc.size()[0]
                avg_val_lon_acc += (torch.sum(torch.max(lon_pred.data, 1)[1] == torch.max(lon_enc.data, 1)[1])).item() / lon_enc.size()[0]
        else:
            fut_pred = net(hist, nbrs, mask, lat_enc, lon_enc)
            if epoch_num < pretrainEpochs:
                l = maskedMSE(fut_pred, fut, op_mask)
            else:
                l = maskedNLL(fut_pred, fut, op_mask)

        avg_val_loss += l.item()
        val_batch_count += 1
    
    print(avg_val_loss/val_batch_count)
    
    # Print validation loss and update display variables
    print('Validation loss :',format(avg_val_loss/val_batch_count,'0.4f'),"| Val Acc:",format(avg_val_lat_acc/val_batch_count*100,'0.4f'),format(avg_val_lon_acc/val_batch_count*100,'0.4f'))
    val_loss.append(avg_val_loss/val_batch_count)
    prev_val_loss = avg_val_loss/val_batch_count

    #__________________________________________________________________________________________________________________________________________________________________________________________________________________________________________________________________________________________________________

torch.save(net.state_dict(), '/data/traj_prediction/result/cslstm20_1_100.tar')

