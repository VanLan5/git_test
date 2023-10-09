import os
os.environ["CUDA_VISIBLE_DEVICES"] = "2"
import torch
from model import highwayNet
from utils import ngsimDataset_ZLY,maskedNLL,maskedMSETest,maskedNLLTest
from new_loss import *
# from trajectory_prediction_CSP.new_loss import *
# from trajectory_prediction_CSP.model import highwayNet
# from trajectory_prediction_CSP.utils import ngsimDataset_ZLY,maskedNLL,maskedMSETest,maskedNLLTest, maskedMSE
from torch.utils.data import DataLoader
import time
from tqdm import tqdm

import setproctitle
setproctitle.setproctitle('fjs_GPU2_traj_eval')

## Network Arguments
args = {}
args['use_cuda'] = True
args['encoder_size'] = 64
args['decoder_size'] = 128
args['in_length'] = 20 
args['out_length'] = 20 
args['grid_size'] = (13,3)
args['soc_conv_depth'] = 64
args['conv_3x1_depth'] = 16
args['dyn_embedding_size'] = 32
args['input_embedding_size'] = 32
args['num_lat_classes'] = 3
args['num_lon_classes'] = 2
args['use_maneuvers'] = True
args['train_flag'] = False

# Evaluation metric:
metric = 'rmse'  #or rmse

# Initialize network
net = highwayNet(args)
ckpt_path = '/data/traj_prediction/result/cslstm20_m_54.tar'
net.load_state_dict(torch.load(ckpt_path))
if args['use_cuda']:
    net = net.cuda()

mat_file = '/data/traj_prediction/data_test/TrainData_1.mat'
tsSet = ngsimDataset_ZLY(mat_file, t_h=args['in_length'], t_f=args['out_length'])
tsDataloader = DataLoader(tsSet,batch_size=128,shuffle=True,num_workers=8,collate_fn=tsSet.collate_fn)

lossVals = torch.zeros(args['out_length']).cuda()
counts = torch.zeros(args['out_length']).cuda() 
step_num = int(args['out_length']/10) +1
ADE = torch.zeros(step_num).cuda()
FDE = torch.zeros(step_num).cuda()

for i, data in tqdm(enumerate(tsDataloader)):
    st_time = time.time()
    hist, nbrs, mask, lat_enc, lon_enc, fut, op_mask = data
    if args['use_cuda']:
        hist = hist.cuda()
        nbrs = nbrs.cuda()
        mask = mask.bool().cuda()
        lat_enc = lat_enc.cuda()
        lon_enc = lon_enc.cuda()
        fut = fut.cuda()
        op_mask = op_mask.cuda()

    if metric == 'nll':

        if args['use_maneuvers']:

            fut_pred, lat_pred, lon_pred = net(hist, nbrs, mask, lat_enc, lon_enc)
            l,c = maskedNLLTest(fut_pred, lat_pred, lon_pred, fut, op_mask)

        else:
            fut_pred = net(hist, nbrs, mask, lat_enc, lon_enc)
            l, c = maskedNLLTest(fut_pred, 0, 0, fut, op_mask,use_maneuvers=False)
    else:
        if args['use_maneuvers']:
            fut_pred, lat_pred, lon_pred = net(hist, nbrs, mask, lat_enc, lon_enc)
            fut_pred_max = torch.zeros_like(fut_pred[0])
            for k in range(lat_pred.shape[0]):
                lat_man = torch.argmax(lat_pred[k, :]).detach()
                lon_man = torch.argmax(lon_pred[k, :]).detach()
                indx = lon_man*3 + lat_man
                fut_pred_max[:,k,:] = fut_pred[indx][:,k,:]
            l, c = maskedMSETest(fut_pred_max, fut, op_mask)
            ade = displacement_error(fut_pred_max[:,:,[0,1]],fut,ade_step=10)*0.01
            fde = final_displacement_error(fut_pred_max[:,:,[0,1]],fut,fde_step=10)*0.01

            ADE+=ade.detach()
            FDE+=fde.detach()
    
        else:
            fut_pred = net(hist, nbrs, mask, lat_enc, lon_enc)
            l, c = maskedMSETest(fut_pred, fut, op_mask)
            ade = displacement_error(fut_pred[:,:,[0,1]],fut,ade_step=10)*0.01
            fde = final_displacement_error(fut_pred[:,:,[0,1]],fut,fde_step=10)*0.01

            ADE+=ade.detach()
            FDE+=fde.detach()


    lossVals +=l.detach()
    counts += c.detach()
print(f'total {i} and mean_ADE is:', ADE/i)
print(f'total {i} and mean_FDE is:', FDE/i)


# 0918
# total 30 and mean_ADE is: tensor([0.5073, 1.1346, 1.1346], device='cuda:0')
# total 30 and mean_FDE is: tensor([0.8819, 2.9008, 2.9008], device='cuda:0')

# 0919-50
