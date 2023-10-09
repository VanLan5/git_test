# -*-coding:utf-8-*
from math import fabs
import os
# os.environ["CUDA_VISIBLE_DEVICES"] = "0"
import numpy as np
import pandas as pd
import copy
import sys
import time
import math
import matplotlib.pyplot as plt
from  numba import  jit
from scipy import stats # 计算正态分布
# ####==== 显示真实轨迹和预测值 ====####
from trajectory_predictor import TrajectoryPredictor
trk = [[0,0], [1,1], [2,2],[2,3]]

#弧度转角度
PI=3.1415926535897932
ARC=6371004
# // 椭球体长半轴, 米
__A = 6378137
# // 椭球体短半轴, 米
# __B = 6356752.3142
__B = 6356725

# // 标准纬度, 弧度（取激光雷达所在纬度）
__B0=0.0
# // 原点经度, 弧度(0°)
__L0=0.0
# //反向转换程序中的迭代初始值
__IterativeValue=10

# 地球长半轴
R_a = 6378137.00
# 地球短半轴
R_b = 6356752.3142

# 弧度转换成角度
def FG_rad2degree(rad):
    return rad*180.0/PI

# 角度转换成弧度
def FG_degree2rad(degree):
    return degree*PI/180.0

# 经纬度转换成墨卡托坐标
def LonLat2Mercator(B, L):
    # f / * 扁率 * /, e / * 第一偏心率 * /, e_ / * 第二偏心率 * /, NB0 / * 卯酉圈曲率半径 * /, K, dtemp;
    # f, e, e_, NB0, K, dtemp=0
    # print('lonlat2mercator')
    f = 0.0
    e = 0.0
    e_ = 0.0
    NB0 = 0.0
    E = 0.0
    dtemp = 0.0
    E = float(math.exp(1))
    # E = math.exp(1)
    # print(f'B:{B},L:{L}')

    # print('1')
    __B0 = B
    __L0 = 0
    if  L < -PI or L > PI or B < -PI / 2 or B > PI / 2:
        # print('2')
        return False
    if __A <= 0 or __B <= 0:
        # print('3')
        return False
    f = (__A - __B) / __A

    dtemp = 1 - (__B / __A) * (__B / __A)
    if dtemp < 0:
        # print('4')
        return False
    # print(f'dtemp1-->{dtemp}')
    e = math.sqrt(dtemp)

    dtemp = (__A / __B) * (__A / __B) - 1
    if dtemp < 0:
        # print('5')
        return False
    # print('6')
    # print(f'dtemp2-->{dtemp}')
    e_ = math.sqrt(dtemp)
    NB0 = ((__A * __A) / __B) / math.sqrt(1 + e_ * e_ * math.cos(__B0) * math.cos(__B0))
    K = NB0 * math.cos(__B0)
    x = K * (L - __L0)
    y = K * math.log(math.tan(PI / 4 + B / 2) * math.pow((1 - e * math.sin(B)) / (1 + e * math.sin(B)), e / 2))
    # print(f'x:{x},y:{y}')
    # return 0
    # print(f'__B0:{__B0}')
    # print(f'L2M:__B0:{__B0},__L0{__L0}')
    return x,y

# 墨卡托坐标转经纬度
def Mercator2LonLat(B,L,X, Y):
    # double f/*扁率*/, e/*第一偏心率*/, e_/*第二偏心率*/, NB0/*卯酉圈曲率半径*/, K, dtemp;
    # double E = exp(1);
    f = 0.0
    e = 0.0
    e_ = 0.0
    NB0 = 0.0
    E = 0.0
    dtemp = 0.0
    E = float(math.exp(1))
    __B0 = B
    __L0 = 0


    if __A <= 0 or __B <= 0:
        return False

    f = (__A - __B) / __A
    dtemp = 1 - (__B / __A) * (__B / __A)
    if dtemp < 0:
        return False
    e = math.sqrt(dtemp)

    dtemp = (__A / __B) * (__A / __B) - 1
    if dtemp < 0:
        return False
    e_ = math.sqrt(dtemp)
    NB0 = ((__A * __A) / __B) / math.sqrt(1 + e_ * e_ * math.cos(__B0) * math.cos(__B0))
    K = NB0 * math.cos(__B0)
    Object_Long = FG_rad2degree(Y / K + __L0)
    # print(f'__B0:{__B0}')
    B = 0.0
    for i in range(__IterativeValue):
        B=PI/2-2*math.atan(math.pow(E, (-X/K)) * math.pow(E, (e/2)*math.log((1-e*math.sin(B))/(1+e*math.sin(B)))))
    Object_Lat= FG_rad2degree(B)
    # print(f'm2l:__B0:{__B0},__L0{__L0}')
    # print(f'object_long {Object_Long},object_lat{Object_Lat}')
    return Object_Long,Object_Lat

def XYZ_To_BLH_batch(original_long,original_lat, box, rotaionangle):
    RadAngle = float(FG_degree2rad(rotaionangle))
    mer_x, mer_y = LonLat2Mercator(FG_degree2rad(original_lat), FG_degree2rad(original_long))
    for i in range(box.shape[0]):
        move_x = box[i][0]
        move_y = box[i][1]
        mer_move_x = move_x * math.cos(RadAngle) + move_y * math.sin(RadAngle) + mer_x
        mer_move_y = move_y * math.cos(RadAngle) - move_x * math.sin(RadAngle) +mer_y
        Object_Long,Object_Lat = Mercator2LonLat(FG_degree2rad(original_lat),FG_degree2rad(original_long),mer_move_y,mer_move_x)
        box[i][-2] = Object_Long
        box[i][-1] = Object_Lat
    return box

#
original_long = 120.6264519756185
original_lat = 31.4241176799312
rotaionangle = 106.106
RadAngle = float(FG_degree2rad(rotaionangle))
original_lat_rad = FG_degree2rad(original_lat)
original_long_rad = FG_degree2rad(original_long)
mer_x, mer_y = LonLat2Mercator(original_lat_rad, original_long_rad)

def XYZ_To_BLH(move_x, move_y):
    mer_move_x = move_x * math.cos(RadAngle) + move_y * math.sin(RadAngle) + mer_x
    mer_move_y = move_y * math.cos(RadAngle) - move_x * math.sin(RadAngle) + mer_y
    Object_Long, Object_Lat = Mercator2LonLat(original_lat_rad, original_long_rad, mer_move_y, mer_move_x)
    return Object_Long, Object_Lat


# @jit(nopython=True)
def cal_new_info(trk1, track1):
    """
    calculation speed
    :param trk1: array (sequeen, id_num, x&y)
    :param track1: array [[id label θ speed x y], ....]
    :return: list_data   [[id,x0,y0, v0, θ0, a0, score0, x1, y1], ...]
    """
    # dict_data = {}
    list_data = [[0.0]]  # assert data type for numba
    for j in range(trk1.shape[1]):  # id_num
        list_id_info = [0.0]
        v_tmp = 0.0
        for i in range(trk1.shape[0]):  # point_num:10
            if i < 1:
                dif_x = trk1[i][j][0] - track1[j][4]/100  # cm2m
                dif_y = trk1[i][j][1] - track1[j][5]/100  # cm2m
                speed1 = (dif_x ** 2 + dif_y ** 2) ** 0.5 / 0.1  # 计算距离m/s  后期可以直接计算速度功能加进来
                acc1 = (speed1 - track1[j][3]) / 0.1
                v_tmp = speed1
                dis_angle = math.atan2(dif_y, dif_x) * 180 / np.pi  # 计算角度
                # lon_new, lat_new = XYZ_To_BLH(trk1[i][j][0], trk1[i][j][1])  # xy2lon_lat
                list_id_info += [track1[j][0], trk1[i][j][0], trk1[i][j][1], speed1, dis_angle, acc1, 99.0]
                # print(list_id_info)
            else:
                dif_x = trk1[i][j][0] - trk1[i - 1][j][0]
                dif_y = trk1[i][j][1] - trk1[i - 1][j][1]

                speed1 = (dif_x ** 2 + dif_y ** 2) ** 0.5 / 0.1  # 计算距离m/s  后期可以直接计算速度功能加进来
                acc1 = (speed1 - v_tmp) / 0.1
                v_tmp = speed1
                dis_angle = math.atan2(dif_y, dif_x) * 180 / np.pi  # 计算角度
                # lon_new, lat_new = XYZ_To_BLH(trk1[i][j][0], trk1[i][j][1])  # xy2lon_lat
                list_id_info += [trk1[i][j][0], trk1[i][j][1], speed1, dis_angle, acc1, 99.0]
        # dict_data[track1[j][0]] = list_id_info
        list_data.append(list_id_info[1:])
    return list_data[1:]


def cal_angle(trk):
    '''
    arg:
        trk:
    '''
    state_list = trk
    dis_x = state_list[-1][0] - state_list[0][0]
    dis_y = state_list[-1][1] - state_list[0][1]

    if len(state_list)>=2:
        dis_angle = math.atan2(dis_y, dis_x) * 180 / np.pi
        # dis_angle = (-1 * dis_angle - 90) % 360  # 和万集主雷达对齐
    return dis_angle


def plot_csp_xy(xy_list0, xy_list, gt_real, track_his, tital):
    fig, ax= plt.subplots()
    size_scatter = 20
    # plt.scatter(0, 0,color='r', marker='x')
    # ####==== 显示真实轨迹 ====####
    ax.plot(xy_list0[:, 0], xy_list0[:, 1], 'b-', label = tital)
    plt.scatter(xy_list0[:, 0], xy_list0[:, 1], marker='>', s=size_scatter)
    
    # ####==== 显示预测轨迹 ====####
    if xy_list is not None:
        ax.plot(xy_list[:, 0], xy_list[:, 1], 'r-', label = 'pred')
        plt.scatter(xy_list[:, 0], xy_list[:, 1], marker='o', s=size_scatter)

    if gt_real is not None:
        ax.plot(gt_real[:, 0], gt_real[:, 1], 'b-', label = 'real')
        plt.scatter(gt_real[:, 0], gt_real[:, 1], marker='o', s=size_scatter)

    if track_his is not None:
        ax.plot(track_his[:, 0], track_his[:, 1], 'g-', label = 'track_his', )
        plt.scatter(track_his[:, 0], track_his[:, 1], marker='o', s=size_scatter, c='y')
    # plt.xlim(-200,200)
    # plt.ylim(-200,200)
    plt.title(tital)
    plt.xlabel('x')
    plt.ylabel('y')
    plt.legend() # 将样例显示出来
    if 0:   # save img
        save_track_path = '/de-30/'
        if not os.path.exists(save_track_path):
            os.makedirs(save_track_path)
        save_track_plot = save_track_path + str(tital) +'.png'
        fig.savefig(save_track_plot)
    else:
        plt.show()
        # plt.pause(1)
        plt.close()
        # ax.clear()


def get_pixel_location(center_x, center_y, img_corner_x, img_corner_y, limit_x, limit_y):
    dist2pixel = 10
    tempx = round((center_x - limit_x[0]) * dist2pixel) - 1
    tempy = round((center_y - limit_y[1]) * dist2pixel) * (-1) - 1

    if (tempx >= img_corner_x) or (tempy >= img_corner_y) or (tempx < 0) or (tempy < 0):
        tempx = -1000
        tempy = -1000
    return int(tempx), int(tempy)


def get_lane_num(all_objects):
    # with open('/data/Guangzhou/station1.json', 'r', encoding='utf8')as fp:
    #     json_data = json.load(fp)
    # # all_objects = np.zeros((10, 2))
    x_limit = [-100, 100]
    y_limit = [-100, 100]
    w = int((x_limit[1] - x_limit[0])*10)
    h = int((y_limit[1] - y_limit[0]) * 10)
    # x_limit = [min(all_objects[0]), max(all_objects[0])]
    # y_limit =  [min(all_objects[1]), max(all_objects[1])]
    lane_nums = []
    # global json_data
    # #####*****  遍历检测的目标(前两列是x,y)  *****#####
    for i in range(all_objects.shape[0]):
        lane_num = 0
        temp_x, temp_y = get_pixel_location(all_objects[i][0], all_objects[i][1], h, w, x_limit, y_limit)
        if w > h:
            haxi_indice = str(temp_x + w * temp_y)
        else:
            haxi_indice = str(temp_y + h * temp_x)
        # if haxi_indice in json_data:
        #     # #####*****  得到车道号和航向角  *****#####
        #     head_angle = json_data[haxi_indice]['angle'][0]
        #     lane_num = json_data[haxi_indice]['lane'][0]
        
        # ####==== added by zqj ====####
        head_angle = 1
        lane_num = 0
        # lane_num = random.randint(0, 5)
        # print(lane_num)
        lane_nums.append(lane_num)
    lane_nums = np.array(lane_nums)
    return lane_nums


def read_tnt_csv(csv_path):
    '''
    @author:zqj
    @func:read csv box for visual
        csv: [id,type,x_coordinate,y_coordinate,z_coordinate,speed,angle,length,width,height,confidence,source,time_stamp,longitude,latitude,frameid]
             [[0, 1,      2,               3,          4,       5,    6,    7,     8,    9,      10,       11,      12,       13,        14,     15]]
    @args:*.csv
    @return:None
    '''
    # csv_path = '/data/qlg_data/csv_test1/000001.csv'
    dataset = pd.read_csv(csv_path, encoding='utf-8')  # header=None 默认去除head
    pc_list = dataset.iloc[:, :].values
    return pc_list


if __name__ == '__main__':

    os.environ["CUDA_VISIBLE_DEVICES"] = "2"
    import setproctitle
    setproctitle.setproctitle('fjs_GPU2_csp_infer')
    in_len = 20 #输入序列长度
    out_len = 20 #输出序列长度
    is_real_data = True
    if is_real_data:
        # 224/450/814/61曲线很好，61和450最好，可能是因为线比较密
        # 1082直线-车速慢，对比另一个直线车速快73
        # csv_path = '/data/wx0727/61.csv'
        # 10719/10628/6601/7794
        # csv_path = '/data/tp_improve/csv_qlg/10719.csv'
        csv_path = '/data/trajectory_predict/data/30258.csv'
    else:
        csv_path = '/data/trajectory_predict/data/saveData12201.csv'    # 生成数据
    id_file = csv_path.split('/')[-1].split('.')[0]
    trackers = read_tnt_csv(csv_path) # 二维矩阵,去除表头，去除了开头一行
    len_track = trackers.shape[0] #256
    print('trackers_shape:', len_track, str(2))

    # ***** 加载trajectory_predictor信息******#
    ckpt_path = '/data/traj_prediction/result/cslstm20_1_100.tar' # 神经网络权重
    # ckpt_path = '/data/traj_prediction/result/cslstm20_m_54.tar' # 神经网络权重
    trajectory_predictor = TrajectoryPredictor(ckpt_path, in_len)
    trackers_cov_all = np.zeros((len_track, 16))
    # trackers_cov[:, [0, 1, 3, 4, 6, 2, 5, 7, 8, 9, 10, 11, 12, 13, 14, 15]] = trackers[:,[0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11,12, 13, 14, 15]]
    trackers_cov_all[:, [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15]] = trackers[:,[2, 3, 4, 8, 7, 9, 6, 1, 5, 0, 11, 10,12, 13, 14, 15]]
    ####==== 加载真实数据 ====####
    if is_real_data:
        start_idx = 0 
        len_traj = len_track-21-start_idx #256-21-0
        is_show = False
        ade_list = []
        fde_list = []
        dif_list = []
        if len_track > start_idx + len_traj: # 肯定成立呀
            for i in range(start_idx + len_traj): #256-21=235
                # ####==== 舍弃前**次 ====####
                if i >=start_idx:
                    trackers_cov = trackers_cov_all[i, :]*100    # 数据还原到cm
                    trackers_cov = trackers_cov[np.newaxis,:] # 增加一维
                    # #####*****  zly:after tracker  *****#####   6-θ 1-label 5-v
                    # #####*****  [x,y,z,w,l,h,theta(deg),label,speed,id,score,hits,...]  *****#####
                    start_time = time.time()
                    lane_nums = get_lane_num(trackers_cov[:, [0, 1]])
                    trackers_cov[:,-5] = lane_nums
                    # print('track info:', trackers_cov) # trackers_cov 包含其中一轨迹点16个参数
                    # 预测xy，轨迹id，当前预测点info
                    traj_fake, traj_id, id_r_info = trajectory_predictor.predict_trajectory(trackers_cov)  # (obs+pred序列长度, consider_objects_num, 2)
                    object_num = []
                    if traj_fake is not None:
                        # print("***************________________**********")
                        print('traj_pred_shape:', traj_fake.shape)
                        # print('traj_fake:', traj_fake[:,0,:])
                        is_show = True
                    if is_show:
                        gt_track = trackers_cov_all[:, [0, 1]][start_idx:start_idx+len_traj+out_len]*100 # 预测点后续全部原始轨迹
                        gt_pred = trackers_cov_all[:, [0, 1]][i+1:i+out_len+1]*100 # 预测对应真实点
                        track_his = trackers_cov_all[:, [0, 1]][i-in_len+1:i+1]*100 # 预测之前20个点
                        print('gt_pred+1:', gt_pred[0, :]) # 预测真实开始点
                        # 移至cpu 返回值是cpu上的Tensor，返回值为numpy.array()
                        pred_track =traj_fake[:20,0,:].detach().cpu().numpy() # 预测轨迹点
                        if 1:
                            diff_x = (pred_track[:, 0] - gt_pred[:, 0])**2
                            diff_y = (pred_track[:, 1] - gt_pred[:, 1])**2
                            diff_all = (diff_x+diff_y)**0.5/100 # 所有点的欧氏距离
                            print('ade@10:', diff_all[:10].mean()) # 预测轨迹和真值轨迹前10个点的平均欧氏距离
                            print('fde@10:', diff_all[9]) # 预测轨迹和真值轨迹最后一个点的欧氏距离
                            ade_list.append(diff_all[:10].mean())
                            fde_list.append(diff_all[9])
                            dif_list.append(diff_all[:10])
                        # plot_csp_xy(gt_track, pred_track,  gt_pred, track_his,id_file+'_'+str(i))
                        is_show = False
                    else:
                        object_num.append(0)
                    object_num.append(trackers_cov.shape[0])
                    path_predict_time = (time.time() - start_time)*1000
                    
                    print('--debug-time:', round(path_predict_time,2))
            
            # print("{}[:{}] - run_traj_alg".format(__file__.split('/')[len(__file__.split('/')) - 1], sys._getframe().f_lineno))
        else:
            print(f'out of range, and len of data is {len_track}')
        # ####==== output metric ====####
        print('--id-ade:', sum(ade_list) / len(ade_list))
        print('--id-fde:', sum(fde_list) / len(fde_list))
        dif_arr = np.array(dif_list) # 199*10
        is_norm = 0 # 指标正态分布判断
        for i in range(dif_arr.shape[1]):
            dif_arr0 = dif_arr[:,i]
            print(f'--zqj-debug diff{i}-----:', sum(dif_arr0)/dif_arr0.shape[0])
            if is_norm:
                arr0_df = pd.DataFrame(dif_arr0, columns = ['value'])
                print(f'diff{i}__norm:', (stats.kstest(arr0_df['value'], 'norm', (arr0_df['value'].mean(), arr0_df['value'].std()))))
        if is_norm:
            ade_df = pd.DataFrame(ade_list, columns = ['value'])
            fde_df = pd.DataFrame(fde_list, columns = ['value'])
            print('ade__norm:', (stats.kstest(ade_df['value'], 'norm', (ade_df['value'].mean(), ade_df['value'].std()))))
            print('fde__norm:', (stats.kstest(fde_df['value'], 'norm', (fde_df['value'].mean(), fde_df['value'].std()))))
        is_save_all_info = 0
        if is_save_all_info:
            # with open('./de.txt', 'a') as f:
            #     f.write(id_file + ','+ str(sum(ade_list) / len(ade_list))+','+str(sum(fde_list) / len(fde_list))+'\n')
            x0 = np.array(range(len(ade_list)))[np.newaxis, :].T
            x1 = np.array(ade_list)[np.newaxis, :].T
            x2 = np.array(fde_list)[np.newaxis, :].T
            ade_arr = np.concatenate((x0,x1), axis=1)
            fde_arr = np.concatenate((x0,x2), axis=1)
            plot_csp_xy(ade_arr, None, None,None, 'ade_'+id_file)
            plot_csp_xy(fde_arr, None, None, None, 'fde_'+id_file)
    # ####==== 自己生成数据模式 ====####qqqqqqqq
    else:
        trackers_cov1 = trackers_cov_all
        with open('/data/trajectory_CSP/Log/debug_time-2.txt', 'a') as f:
            f.write('---'*20+'\n')
            is_save_data = False
            fram_id = 0
            for i in range(99):
                # ####====  模拟数据变化 对于拐弯，需要舍弃前20次变化缓慢的数据 ====####
                trackers_cov = trackers_cov1
                trackers_cov[:, 1] += 50
                # trackers_cov[:, 1] += 1.1**i
                # trackers_cov[:, [0, 1]] += 50
                if i >=30:
                    print('--zqj-i:', i)
                    # #####*****  zly:after tracker  *****#####
                    #             [0,1,2,3,4,5,     6,      7,    8,   9,  10,   11,...]
                    # #####*****  [x,y,z,w,l,h,theta(deg),label,speed,id,score,hits,...]  *****#####

                    start_time = time.time()
                    time.sleep(0.1)
                    if i==60:   # zqj20230627 测试id
                        print('--'*20)
                        trackers_cov = trackers_cov[:49,:]
                        # ####====  zengji id ====####
                        # a = np.array([[-6.79700000e+03+1000, 4.96000000e+02+1000, -5.12000000e+02,  2.67000000e+02,
                        #                   9.60000000e+02,  2.82000000e+02 , 2.86100000e+03 , 6.00000000e+00,
                        #                   3.00000000e+00 , 28551 , 0.00000000e+00 , 0.00000000e+00,
                        #                   1.67150329e+09,  1.20626092e+02  ,3.14247415e+01 , 1.83800000e+03]])
                        # trackers_cov = np.concatenate((trackers_cov, a), axis=0)
                        # print(trackers_cov.shape)
                    # ####==== 获取长度为目标数据量的全零的list ====####
                    lane_nums = get_lane_num(trackers_cov[:, [0, 1]])
                    trackers_cov[:,-5] = lane_nums
                    traj_fake, traj_id, track_last_info = trajectory_predictor.predict_trajectory(trackers_cov)  # (obs+pred序列长度, consider_objects_num, 2)
                    object_num = []
                    if traj_fake is not None:
                        print(f"****____objects_num:{len(traj_id)} {track_last_info.shape}____****")
                        # print(traj_id)
                        traj_fake = traj_fake[:10, :, :].detach().cpu().numpy()/100   # cm2m

                        traj_pred = copy.deepcopy(traj_fake[:10, ...])
                        # print('traj_fake:', traj_fake[0, ...])
                        if is_save_data:
                            np_save_name = '/data/git_zqj/trajectory_predict/data/org_' + str(fram_id) + '.npy'
                            np.save(np_save_name, traj_fake)
                        timer1 = time.time()
                        num_idx, num_id = traj_fake.shape[:2]
                        add_zi = np.full(shape=[num_idx, num_id, 2], fill_value=[-6.3, 0.45])
                        traj_fake_show = np.concatenate((traj_fake, add_zi), axis=2)
                        # print('---debug-traj_pred_11:', traj_fake[0, 5, :])
                        # print('--zqj-debug-add_shape:', add_zi.shape)
                        traj_fake_show = traj_fake_show.reshape(-1, 4)
                        post_time = round((time.time() - timer1) * 1000, 2)
                        print('----zqj-debug-traj_post_time:', post_time)
                        if 1:
                            # ####==== 增加速度(需要类别)、加速度(todo)、航向角(需要初始角度)、置信度的输出 ====####
                            cal_new_info_time_start = time.time()
                            # time.sleep(0.01)
                            traj_info_list = cal_new_info(traj_pred, track_last_info)
                            cal_new_info_time = round((time.time() - cal_new_info_time_start) * 1000, 2)
                            # print('traj_info_list:', np.array(traj_info_list)[:,[1,2]])
                            print('--zqj-debug-traj-cal_new_info-time(ms):', cal_new_info_time)
                            if is_save_data:
                                np_save_name1 = '/data/git_zqj/trajectory_predict/data/new_' + str(fram_id) + '.npy'
                                np.save(np_save_name1, np.array(traj_info_list))

                        # # ####==== code 增加速度(需要类别)、加速度(todo)、航向角(需要初始角度)、置信度的输出 ====####
                        # cal_new_info_time = time.time()
                        # traj_info_list = cal_new_info(pred_track, track_last_info)
                        # # print(traj_info_list)
                        # path_predict_time = (time.time() - cal_new_info_time) * 1000
                        # print('----------------- cal_new_info-time(ms):', round(path_predict_time, 2))

                    else:
                        object_num.append(0)
                    object_num.append(trackers_cov.shape[0])
                    path_predict_time = (time.time() - start_time)*1000
                    print('----------------------------------> total-time(ms):', round(path_predict_time,2)-100)
                    f.write(str(i)+' '+str(round(path_predict_time,2))+'\n')
        print("{}[:{}] - run_traj_alg".format(__file__.split('/')[len(__file__.split('/')) - 1], sys._getframe().f_lineno))
