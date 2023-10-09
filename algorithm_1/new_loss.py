import torch
import numpy as np
def displacement_error(pred_traj, pred_traj_gt, ade_step = None,consider_ped=None, mode='mean'):
    """
    Input:
    - pred_traj: Tensor of shape (seq_len, batch, 2). Predicted trajectory.
    - pred_traj_gt: Tensor of shape (seq_len, batch, 2). Ground truth
    predictions.
    - consider_ped: Tensor of shape (batch)
    - mode: Can be one of sum, raw
    Output:
    - loss: gives the eculidian displacement error
    """
    MASK = torch.where((pred_traj_gt[-1]).sum(dim=1)!=0)[0]
    # print(MASK)
    # MASK = MASK.cpu().detach().numpy()
    pred_traj = pred_traj[:,MASK,:] # torch.Size([20, 110, 2])
    pred_traj_gt = pred_traj_gt[:,MASK,:]
    seq_len, _, _ = pred_traj.size()
    ade_num = seq_len//ade_step if ade_step is not None else 1
    loss = pred_traj_gt.permute(1, 0, 2) - pred_traj.permute(1, 0, 2) # Tensor of shape (batch, seq_len, 2)
    loss = loss**2 # Tensor of shape (batch, seq_len, 2)
    loss = torch.sqrt(loss.sum(dim=2)) # torch.Size([110, 20])
    if ade_num  == 1:
        if consider_ped is not None:
            loss = loss.sum(dim=1) * consider_ped # Tensor of shape (batch)
        else:
            loss = loss.sum(dim=1) # Tensor of shape (batch)
        if mode == 'mean':
            return torch.mean(loss)
        elif mode == 'raw':
            return loss
    else:
        new_loss = torch.zeros((loss.shape[0],ade_num+1)).cuda() # Tensor of shape (batch,ade_num+1)
        if consider_ped is not None:
            for i in range(ade_num):
                new_loss[:,i]= loss[:,0:(i+1)*ade_step].mean(dim=1) * consider_ped
            new_loss[:,-1] = loss.mean(dim=1) * consider_ped # Tensor of shape (batch,ade_num+1)
        else:
            for i in range(ade_num):
                new_loss[:, i] = loss[:, 0:(i + 1) * ade_step].mean(dim=1)
            new_loss[:, -1] = loss.mean(dim=1) # Tensor of shape (batch,ade_num+1)
        # print(new_loss)
        # print(torch.mean(new_loss,dim=0))
        if mode == 'mean':
            return torch.mean(new_loss,dim=0) # Tensor of shape (ade_num+1)
        elif mode == 'raw':
            return new_loss # # Tensor of shape (batch,ade_num+1)



def final_displacement_error(
    pred_traj, pred_traj_gt,fde_step=None, consider_ped=None, mode='mean'
):
    """
    Input:
    - pred_traj: Tensor of shape (seq_len, batch, 2). Predicted trajectory.
    - pred_traj_gt: Tensor of shape (seq_len, batch, 2). Ground truth
    last pos
    - consider_ped: Tensor of shape (batch)
    Output:
    - loss: gives the eculidian displacement error
    """
    MASK = torch.where((pred_traj_gt[-1]).sum(dim=1)!=0)[0]
    pred_traj = pred_traj[:,MASK,:]
    pred_traj_gt = pred_traj_gt[:,MASK,:]
    seq_len, _, _ = pred_traj.size()
    fde_num = seq_len // fde_step if fde_step is not None else 1
    if fde_num == 1:
        pred_pos=pred_traj[-1] # Tensor of shape (batch, 2)
        pred_pos_gt = pred_traj_gt[-1] # Tensor of shape (batch, 2)
        loss = pred_pos_gt - pred_pos # Tensor of shape (batch, 2)
        loss = loss**2
        if consider_ped is not None:
            loss = torch.sqrt(loss.sum(dim=1)) * consider_ped
        else:
            loss = torch.sqrt(loss.sum(dim=1))
        if mode == 'raw':
            return loss
        elif mode == "mean" :
            return torch.mean(loss)
    else:
        seq = [(i+1)*fde_step-1 for i in range(fde_num)]
        seq.append(-1)
        pred_pos = pred_traj[seq] # Tensor of shape (1,fde_num +1,batch, 2)
        pred_pos_gt = pred_traj_gt[seq] # Tensor of shape (1,fde_num +1,batch, 2)

        pred_pos = torch.squeeze(pred_pos) # Tensor of shape (fde_num +1,batch, 2)
        pred_pos_gt = torch.squeeze(pred_pos_gt) # Tensor of shape (fde_num +1,batch, 2)
        loss = pred_pos_gt.permute(1, 0, 2) - pred_pos.permute(1, 0, 2)  # Tensor of shape (batch, fde_num +1, 2)
        loss = loss ** 2 # Tensor of shape (batch, fde_num +1, 2)
        if consider_ped is not None:
            loss = torch.sqrt(loss.sum(dim=2)) * consider_ped # Tensor of shape (batch, fde_num +1)
        else:
            loss = torch.sqrt(loss.sum(dim=2)) # Tensor of shape (batch, fde_num +1)
        if mode == 'raw':
            return loss # Tensor of shape (batch, fde_num +1)
        elif mode == "mean":
            return torch.mean(loss,dim=0) # Tensor of shape (fde_num +1)
def all_distance(pred_traj,pred_traj_gt,dis_step=None):
    """
    Input:
    - pred_traj: Tensor of shape (seq_len, batch, 2). Predicted trajectory.
    - pred_traj_gt: Tensor of shape (seq_len, batch, 2). Ground truth
    """
    MASK = torch.where((pred_traj_gt[-1]).sum(dim=1)!=0)[0]
    pred_traj = pred_traj[:,MASK,:]
    pred_traj_gt = pred_traj_gt[:,MASK,:]
    seq_len, _, _ = pred_traj.size()
    dis_num = seq_len //dis_step if dis_step is not None else 1
    seq = [(i + 1) * dis_step - 1 for i in range(dis_num)]
    seq.append(-1)

    if dis_num ==1:
        pred_pos = pred_traj[-1]  # Tensor of shape (batch, 2)
        pred_pos_gt = pred_traj_gt[-1]  # Tensor of shape (batch, 2)
        dis = (torch.sqrt((pred_pos**2).sum(dim=1)) +torch.sqrt((pred_pos_gt**2).sum(dim=1)))/2
    else:
        pred_pos = pred_traj[seq]  # Tensor of shape (1,dis_num +1,batch, 2)
        pred_pos_gt = pred_traj_gt[seq]  # Tensor of shape (1,dis_num +1,batch, 2)
        pred_pos = torch.squeeze(pred_pos)  # Tensor of shape (dis_num +1,batch, 2)
        pred_pos_gt = torch.squeeze(pred_pos_gt)  # Tensor of shape (dis_num +1,batch, 2)
        dis = (torch.sqrt(((pred_pos.permute(1, 0, 2) )**2).sum(dim=2)) +torch.sqrt(((pred_pos_gt.permute(1, 0, 2) )**2).sum(dim=2)))/2  # Tensor of shape (batch, dis_num +1)
        # print(torch.sqrt(((pred_pos.permute(1, 0, 2) )**2).sum(dim=2)))
        # print(torch.sqrt(((pred_pos_gt.permute(1, 0, 2) )**2).sum(dim=2)))
    dis = dis.mean(dim=0)
    return dis