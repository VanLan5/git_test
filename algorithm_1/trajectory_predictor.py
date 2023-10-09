import time
import numpy as np
import torch
import sys
import matplotlib.pyplot as plt
from scipy.signal import savgol_filter # 轨迹平滑
from model import highwayNet
from utils import ngsimDataset_ZLY,maskedNLL,maskedMSETest,maskedNLLTest

global args
args = {}
args['use_cuda'] = True # 并行计算
args['encoder_size'] = 64 # 编码器
args['decoder_size'] = 128 # 解码器
args['in_length'] = 30   # 30 # 输入序列
args['out_length'] = 20  # 50 # 输出序列
args['grid_size'] = (13,3) # cuda_kernel对应的参数
args['soc_conv_depth'] = 64  # 卷积深度
args['conv_3x1_depth'] = 16
args['dyn_embedding_size'] = 32 # 嵌入层
args['input_embedding_size'] = 32
args['num_lat_classes'] = 3 # 标签
args['num_lon_classes'] = 2
args['use_maneuvers'] = True
args['train_flag'] = False

class TrajectoryPredictor():
    def __init__(self, model_path, input_len):
        global args
        args['in_length'] = input_len
        args = self.load_model(model_path)
        self.obs_len = args['in_length']  # 30
        self.pred_len = args['out_length']  # 50
        self.grid_size=args['grid_size']   # (13,3)
        self.enc_size = args['encoder_size']  # 64
        # self.linear_threshold = 0.002
        self.min_object_num = 1
        self._history_frame_num = 0  # maxnum = obs_len + 1
        self._history_frame_info = None
        self.obs_traj = None
        self.obs_traj_rel = None
        self.old_seq_end = []
        self.d_s =1
        # ####==== add zqj20230627 ====####
        self._history_frame_info1 = None
        self.old_seq_end1 = []
        self.id_label_theta = None

    def load_model(self, model_path):
        # checkpoint = torch.load(model_path)
        global args
        # Evaluation metric:
        metric = 'nll'  # or rmse # 可能为nll损失函数
        # Initialize network
        net = highwayNet(args) 
        # print(net)
        net.load_state_dict(torch.load(model_path))
        if args['use_cuda']:
            net = net.cuda()
        self.generator = net # 给高速网络配置地址与cuda
        return args

    def update(self, trackers_cov): 
        # trackers_trajectory = trackers_cov[:, [9, 1, 0]] # id y x ,交换xy ->> yx
        trackers_trajectory = trackers_cov[:, [9, 0, 1]] # id y1 x0 ,交换xy ->> yx
        trackers_trajectory1= trackers_cov[:, [9, 7, 6, 8, 0, 1]] # id lagel θ speed x y
        # trackers_trajectory[:,[1,2]]=trackers_trajectory[:,[1,2]]  # *3.28083989501/100 # 转换单位  米->英尺  cm    cc by zqj
        
        if self._history_frame_num == self.obs_len:
            self._history_frame_info = np.concatenate((self._history_frame_info[self.old_seq_end[0]:],
                                                      trackers_trajectory), axis=0)
            self.old_seq_end = self.old_seq_end[1:] + [trackers_trajectory.shape[0]]
            # ####==== zqj20230627 ====####
            self._history_frame_info1 = np.concatenate((self._history_frame_info1[self.old_seq_end1[0]:],
                                                      trackers_trajectory1), axis=0)
            self.old_seq_end1 = self.old_seq_end1[1:] + [trackers_trajectory1.shape[0]]
        elif self._history_frame_num > 0 and self._history_frame_num < self.obs_len:
            self._history_frame_info = np.concatenate((self._history_frame_info, trackers_trajectory), axis=0)
            self.old_seq_end.append(trackers_trajectory.shape[0])
            # ####==== zqj20230627 ====####
            self._history_frame_info1 = np.concatenate((self._history_frame_info1, trackers_trajectory1), axis=0)
            self.old_seq_end1.append(trackers_trajectory1.shape[0])
            self._history_frame_num += 1
        elif self._history_frame_num == 0:
            self._history_frame_info = trackers_trajectory
            self.old_seq_end.append(trackers_trajectory.shape[0])
            # ####==== zqj20230627 ====####
            self._history_frame_info1= trackers_trajectory1
            self.old_seq_end1.append(trackers_trajectory1.shape[0])
            self._history_frame_num += 1

    def get_obs_traj(self,trackers_cov):
        if self._history_frame_num == self.obs_len:
            objects_in_curr_seq = np.unique(self._history_frame_info[:, 0])  # 序列数据-数据内的object id
            # curr_seq_rel = np.zeros((len(objects_in_curr_seq), 2, self.obs_len+1))  # 初始化 （行人数量,2,obs序列长度）
            # curr_seq = np.zeros((len(objects_in_curr_seq), 2, self.obs_len+1))  # 初始化 （行人数量,2,obs序列长度）
            # curr_loss_mask = np.zeros((len(objects_in_curr_seq),
            #                            self.obs_len+1))  # 初始化  （行人数量,序列长度）
            num_objects_considered = 0  # 场景下考虑进行预测的object数量
            # _non_linear_ped = []
            obs_objects_considered_id = [] # 场景下考虑进行预测的object id
            # self._history_frame_info = 1
            # print(self._history_frame_info1)
            # print('curr_object_seq1:',len(self._history_frame_info1))
            self.id_label_theta= np.zeros((1,6))
            for _, object_id in enumerate(objects_in_curr_seq):
                # print(self._history_frame_info.shape)
                # ####==== zqj20230712 use插值后新的轨迹数据 ====####
                curr_object_seq = self.tracks[object_id][:,1:]
                # curr_object_seq = self._history_frame_info[self._history_frame_info[:, 0] == object_id, :]  # 某个object的序列数据  cc by zqj20230712
                curr_object_seq = np.around(curr_object_seq, decimals=4)  # 保留小数点后4位
                # ####==== zqj20230627 ====####
                curr_object_seq1 = self._history_frame_info1[self._history_frame_info1[:, 0] == object_id, :]  # 某个object的序列数据
                # curr_object_seq1 = np.around(curr_object_seq1, decimals=4)  # 保留小数点后4位
                # 判断序列长度是否完整 ,没有完全出现在观测+
                # pad_front = frames.index(curr_object_seq[0, 0]) - idx
                # pad_end = frames.index(curr_object_seq[-1, 0]) - idx + 1
                if curr_object_seq.shape[0] == self.obs_len:
                    num_objects_considered += 1
                    obs_objects_considered_id.append(object_id)
                    # ####==== zqj20230627 ====####
                    # self.id_label_theta.append(curr_object_seq1[0])
                    self.id_label_theta = np.concatenate((self.id_label_theta, curr_object_seq1[-1].reshape(1,6)), axis=0)
                    
                # else:
                #     obs_objects_considered_id.append(object_id)
            self.T_veh_ids = obs_objects_considered_id
            self.hist = {}
            self.neighbors = {}
            # grid_location = self.get_grid_location(trackers_cov)
            len_hist = self.obs_len-1  # 19 29
            for vehId in obs_objects_considered_id:
                hist = self.getHistory(vehId, len_hist,vehId,1)

                neighbors = []
                # for i in grid_location[vehId]:
                #     if self.getHistory(i.astype(int), len_hist, vehId, dsId =1).shape[0]!=0:
                #         neighbors.append(self.getHistory(i.astype(int), len_hist, vehId, dsId =1))
                self.hist[vehId] = hist
                self.neighbors[vehId] = neighbors
            self.get_mask()
            # t = 5


    def predict_trajectory(self, trackers_cov):
        timer5 = time.time()
        self.update(trackers_cov) # 调整数据结构，更新历史轨迹数据
        self.get_tracks(trackers_cov) # 对轨迹数据进行插值和平滑
        self.get_obs_traj(trackers_cov) # 目标数
        timer6 = time.time()
        # print('infer time6:', round((timer6-timer5)*1000,2))
        if self._history_frame_num ==self.obs_len and len(self.T_veh_ids) != 0:
            print('--zqj-data-process:', len(self.T_veh_ids))
            timer7 = time.time()
            hist = [] # history
            # neighbors =[]
            masks = []
            ref_pos = []
            masks = np.zeros((len(self.T_veh_ids), 3, 13, 64), dtype=np.float32)
            ref_pos = np.zeros((len(self.T_veh_ids), 2), dtype=np.float32)
            hist = np.zeros((len(self.T_veh_ids), self.obs_len, 2), dtype=np.float32)
            for idx, veh_id in enumerate(self.T_veh_ids):
                # ref_pos.append(self.tracks[veh_id][-1,[2,3]])
                # hist.append(self.hist[veh_id])
                # print('hist_shape:', self.hist[veh_id].shape)
                # if len(self.neighbors[veh_id]) !=0:
                # neighbors+=self.neighbors[veh_id]
                # print(self.masks[veh_id].shape)
                # print(self.tracks[veh_id].shape)
                # masks.append(self.masks[veh_id])
                masks[idx] = self.masks[veh_id]
                hist[idx] = self.hist[veh_id]
                ref_pos[idx] = self.tracks[veh_id][-1,[2,3]]
            # print(masks) # 空矩阵
            # print(hist) # 历史轨迹点
            # print(ref_pos) # 预测第一点真实值
            timer8 = time.time()
            # print('infer time8:', round((timer8-timer7)*1000,2))
            # ref_pos = torch.Tensor(ref_pos).cuda()
            ref_pos = torch.from_numpy(ref_pos).cuda()
            lon_enc = torch.zeros([len(self.T_veh_ids),2]).cuda()
            # lon_enc[int(self.D[idx, 7] - 1)] = 1
            lat_enc =  torch.zeros([len(self.T_veh_ids),3]).cuda()
            timer9 = time.time()
            # print('infer time9:', round((timer9-timer8)*1000,2))
            # print('hist_shape:', torch.Tensor(hist).size(),  torch.Tensor(masks).size())
            # hist = torch.Tensor(hist).permute([1,0,2]).cuda()
            print('hist shpe-------------------:', hist.shape)
            hist = torch.from_numpy(hist).cuda().permute([1, 0, 2])
            # print('infer time10:', round((time.time() - timer9) * 1000, 2))
            # masks = torch.tensor(masks)
            masks = torch.from_numpy(masks)
            # print('infer time10:', round((time.time() - timer9) * 1000, 2))
            masks = masks.cuda()
            # print('infer time10:', round((time.time() - timer9) * 1000, 2))
            masks= masks.bool()
            # neighbors = torch.Tensor(neighbors).cuda()
            timer10 = time.time()
            # print('infer time10:', round((timer10-timer9)*1000,2))
            # print(hist.shape,neighbors.shape)
            # neighbors = neighbors.permute([1, 0, 2])
            # if neighbors.shape[0] != 0:
            #     neighbors =neighbors.permute([1,0,2])
            # else:
            neighbors = torch.zeros((self.obs_len,1,2)).cuda()  # zqj20230627 todo 传参 30
            # print('hist', hist.size())
            timer0 = time.time()
            # print('infer time0:', round((timer0-timer10)*1000,2))
            # 预测的结果
            # print(11, hist) # 历史轨迹点tensor
            # print(22, masks) # 全为False矩阵tensor
            # print(33, neighbors) # 20行空tensor
            # print(44, lat_enc) # tensor([[0., 0., 0.]]
            # print(55, lon_enc) # tensor([[0., 0.]]
            fut_pred, lat_pred, lon_pred=self.generator(hist, neighbors, masks, lat_enc, lon_enc)
            # print(66, fut_pred) # 6*20*5数据点
            # print(77, lat_pred) # tensor([[1.0000e+00, 3.6407e-15, 3.6056e-15]]
            # print(88, lon_pred) # tensor([[1.0000e+00, 4.6936e-15]]
            timer1 = time.time()
            print('infer time:', round((timer1-timer0)*1000,2))
            
            fut_pred_max = torch.zeros_like(fut_pred[0])
            single_output_flag = True
            if single_output_flag:
                fut_pred_all = fut_pred_max.repeat(1,3,1)
            timer2 = time.time()
            # print('time2:', round((timer2-timer1)*1000,2))
            for k in range(lat_pred.shape[0]):
                if single_output_flag:  #  edite by zqj20230315
                    # lat_man = torch.argmax(lat_pred[k, :]).detach()
                    # lon_man = torch.argmax(lon_pred[k, :]).detach()
                    # indx = lon_man * 3 + lat_man   #  zqj? ==0
                    # fut_pred_max[:, k, :] = fut_pred[indx][:, k, :]
                    fut_pred_max[:, k, :] = fut_pred[0][:, k, :]
                else:
                    lat_man = torch.argmax(lat_pred[k, :]).detach()
                    lon_man = torch.argmax(lon_pred[k, :]).detach()
                    indx = lon_man * 3 + lat_man
                    for i in range(3):
                        fut_pred_all[:, k+i*lat_pred.shape[0], :] = fut_pred[lon_man * 3 +i][:, k, :]
            timer3 = time.time()
            # print('time3:', round((timer3-timer2)*1000,2))
            if single_output_flag:
                traj_fake = fut_pred_max[:, :,[0, 1]]+ref_pos
            else:
                traj_fake = fut_pred_all[:, :, [0, 1]] + ref_pos.repeat(3,1)
            # traj_fake = traj_fake[:, :, [1, 0]] # 交换x y位置
            traj_fake = traj_fake[:, :, [0, 1]] # 不交换x y位置
            timer4 = time.time()
            # print('time4:', round((timer4-timer3)*1000,2))
            # ####==== eval中不需要该操作  ====####
            # traj_fake[:, :, [0, 1]] =traj_fake[:, :, [0, 1]] # /3.28083989501 # 转换单位  英尺->米
            # print(traj_fake[0,:,:])
            # print('predict_trajectory:', traj_fake[-1, :, :]-traj_fake[0,:,:])
            # print("{}[:{}] - run_traj alg".format(__file__.split('/')[len(__file__.split('/')) - 1], sys._getframe().f_lineno))
            
            return traj_fake, self.T_veh_ids, self.id_label_theta[1:,:]
            # return traj_fake[:20,:,:]
        else:
            return None, None, None

        # return None  # cc zqj20230627
    def getHistory(self,vehId,t,refVehId,dsId):
        if vehId == 0:
            return np.empty([0,2])
        else:
            if vehId not in self.T_veh_ids:
                # print(vehId)
                # print(self.T_veh_ids)
                return np.empty([0,2])
        vehTrack = self.tracks[vehId] # .transpose()
        if self.tracks[vehId].shape[0]<self.obs_len or np.argwhere(vehTrack[:, 0] == t).size == 0:
            return np.empty([0, 2])
        refTrack = self.tracks[refVehId]  # .transpose()
        refPos = refTrack[-1, 2:4]
        stpt = np.maximum(0, vehTrack.shape[0]- self.obs_len)
        enpt = np.argwhere(vehTrack[:, 0] == t).item() + 1
        hist = vehTrack[stpt:enpt:self.d_s, 2:4] - refPos
        if len(hist) < self.obs_len//self.d_s:
            return np.empty([0,2])
        return hist

    def get_grid_location(self,trackers_cov, grid_parameter={"dis_range": 90, "grid_dis": 15, "lane_ids": [1, 2, 3, 4]}):
        dis_range = grid_parameter["dis_range"]
        grid_dis = grid_parameter["grid_dis"]
        # lane_ids = grid_parameter["lane_ids"]
        grid_num = round(dis_range * 2 / grid_dis) + 1
        trackers_trajectory = trackers_cov[:, [9, 0, 1, -5]]  # object_id, x, y, lane_id
        veh_ids = np.unique(trackers_trajectory[:, 0])
        grid_location = {}
        for veh_id in veh_ids:
            grid_location[veh_id] = np.zeros([grid_num * 3])
            lane_id = trackers_trajectory[trackers_trajectory[:, 0] == veh_id][0, 3]
            ref_y = trackers_trajectory[trackers_trajectory[:, 0] == veh_id][0, 2]
            # if lane_id =
            frameEgo =trackers_trajectory[trackers_trajectory[:, 3] == lane_id] if any(trackers_trajectory[:, 3] == 1) else np.zeros((0,4))
            frameL = trackers_trajectory[trackers_trajectory[:, 3] == lane_id + 1] if any(trackers_trajectory[:, 3] == lane_id + 1) else np.zeros((0,4))
            frameR = trackers_trajectory[trackers_trajectory[:, 3] == lane_id - 1] if any(trackers_trajectory[:, 3] == lane_id - 1) else np.zeros((0,4))

            if frameL.shape[0] != 0:
                for l in range(frameL.shape[0]):
                    y = frameL[l, 2] - ref_y
                    if abs(y) < 90:
                        gridInd = round((y + 90) / 15)
                        grid_location[veh_id][gridInd] = frameL[l, 0]
            for l in range(frameEgo.shape[0]):
                y = frameEgo[l, 2] - ref_y
                if abs(y) < 90 and y != 0:
                    gridInd = grid_num + round((y + 90) / 15)
                    grid_location[veh_id][gridInd] = frameEgo[l, 0]
            if frameR.shape[0] != 0:
                for l in range(frameR.shape[0]):
                    y = frameR[l, 2] - ref_y
                    if abs(y) < 90:
                        gridInd = 2 * grid_num + + round((y + 90) / 15)
                        grid_location[veh_id][gridInd] = frameR[l, 0]
        self.grid_location = grid_location
        return grid_location

    def get_mask(self):
        self.masks = {}
        if len(self.T_veh_ids) != 0:
            for veh_id in self.T_veh_ids:

                neighbors = self.neighbors[veh_id]
                pos = [0, 0]
                mask = np.zeros((self.grid_size[1], self.grid_size[0],self.enc_size))
                for id, nbr in enumerate(neighbors):
                    if len(nbr) != 0:
                        pos[0] = id % self.grid_size[0]
                        pos[1] = id // self.grid_size[0]
                        mask[pos[1], pos[0],:] = 1
                self.masks[veh_id] = mask
    def get_tracks(self,trackers_cov):
        if self._history_frame_num == self.obs_len:
            # self._history_frame_info = None
            # self.old_seq_end = []
            history_frame_info = self._history_frame_info # 3列，id y x
            # ####==== 在数据的第一列增加目标出现的帧号 用于判断缺少的帧数 ====####
            history_frame_info = np.concatenate((np.zeros((history_frame_info.shape[0], 1)), history_frame_info), axis=1)
            seq_start = 0
            for i, seq_end in enumerate(self.old_seq_end):
                if seq_end != 0:
                    seq_end = seq_start+seq_end
                    history_frame_info[seq_start:seq_end, 0] = i
                    seq_start = seq_end
            veh_ids = np.unique(history_frame_info[:, 1])
            tracks = {}
            for veh_id in veh_ids:
                veh_track = history_frame_info[history_frame_info[:, 1] == veh_id]
                # ####==== 目标帧差和出现次数 并开始插值 ====####
                if (veh_track[-1, 0] - veh_track[0, 0] + 1) > veh_track.shape[0]:
                    j = 0
                    new_veh_track = np.zeros((int(veh_track[-1, 0] - veh_track[0, 0]+1), 4))

                    for i in range(veh_track.shape[0]):
                        if i == 0:
                            new_veh_track[j, :] = veh_track[i, :]
                            j += 1
                            continue
                        step_num = int(veh_track[i, 0] - veh_track[i - 1, 0])
                        if step_num == 1:
                            new_veh_track[j, :] = veh_track[i, :]
                            j += 1
                        else:
                            # ####==== 等距离插值 ====####
                            step_XY = (veh_track[i, [2, 3]] - veh_track[i - 1, [2, 3]]) / step_num
                            for k in range(step_num):
                                new_veh_track[j, 0] = veh_track[i - 1, 0] + (k + 1)
                                new_veh_track[j, 1] = veh_track[i - 1, 1]
                                new_veh_track[j, [2, 3]] = veh_track[i - 1, [2, 3]] + step_XY * (k + 1)
                                j += 1
                else:
                    new_veh_track = veh_track
                # ####==== 轨迹数据平滑 ====####
                # new_veh_track[:,-1] = savgol_filter(new_veh_track[:,-1], 19, 8)
                # new_veh_track[:,-2] = savgol_filter(new_veh_track[:,-2], 19, 8)
                tracks[veh_id] = new_veh_track # new_veh_track[:, 1:]
            # ####==== 插值后新的轨迹数据 ====####
            self.tracks = tracks
            return None