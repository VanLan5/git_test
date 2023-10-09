from __future__ import division
import torch
from torch.autograd import Variable
import torch.nn as nn
from utils import outputActivation

class highwayNet(nn.Module):

    ## Initialization 初始化
    def __init__(self,args): # 不固定数目参数
        super(highwayNet, self).__init__()

        ## Unpack arguments
        self.args = args

        ## Use gpu flag
        self.use_cuda = args['use_cuda']

        # Flag for maneuver based (True) vs uni-modal decoder (False) 和下lon和lat有关
        self.use_maneuvers = args['use_maneuvers']

        # Flag for train mode (True) vs test-mode (False)
        self.train_flag = args['train_flag']

        ## Sizes of network layers
        self.encoder_size = args['encoder_size']  # 64
        self.decoder_size = args['decoder_size']  # 128
        self.in_length = args['in_length']  # 30
        self.out_length = args['out_length']    # 50
        self.grid_size = args['grid_size']  # (13,3)
        self.soc_conv_depth = args['soc_conv_depth']    # 64 输入张量的channels数
        self.conv_3x1_depth = args['conv_3x1_depth']    # 16 输出张量的channels数
        self.dyn_embedding_size = args['dyn_embedding_size']    # 32
        self.input_embedding_size = args['input_embedding_size']    # 32
        self.num_lat_classes = args['num_lat_classes']  # 3
        self.num_lon_classes = args['num_lon_classes']  # 2
        self.soc_embedding_size = (((args['grid_size'][0]-4)+1)//2)*self.conv_3x1_depth  # 80
        # self.soc_embedding_size = 160 # fjs-为了适应7*7网络 2023-9-25
        ## Define network weights 定义网络权重

        # Input embedding layer 输入嵌入层
        self.ip_emb = torch.nn.Linear(2,self.input_embedding_size)

        # Encoder LSTM LSTM编码器，
        self.enc_lstm = torch.nn.LSTM(self.input_embedding_size,self.encoder_size,1)

        # Vehicle dynamics embedding 车辆动力学嵌入
        self.dyn_emb = torch.nn.Linear(self.encoder_size,self.dyn_embedding_size)

        # Convolutional social pooling layer and social embedding layer 卷积社交池化：两个卷积层和一个池化层
        self.soc_conv = torch.nn.Conv2d(self.encoder_size,self.soc_conv_depth,3) # 卷积核3*3
        self.conv_3x1 = torch.nn.Conv2d(self.soc_conv_depth, self.conv_3x1_depth, (3,1)) # 卷积核3*1
        self.soc_maxpool = torch.nn.MaxPool2d((2,1),padding = (1,0)) # 池化

        # FC social pooling layer (for comparison):
        # self.soc_fc = torch.nn.Linear(self.soc_conv_depth * self.grid_size[0] * self.grid_size[1], (((args['grid_size'][0]-4)+1)//2)*self.conv_3x1_depth)

        # Decoder LSTM 基于机动的LSTM解码器，本来是结合横向换道特征和纵向制动特征，现在是没有或没有挖掘数据
        if self.use_maneuvers:
            # LSTM(117, 128)
            self.dec_lstm = torch.nn.LSTM(self.soc_embedding_size + self.dyn_embedding_size + self.num_lat_classes + self.num_lon_classes, self.decoder_size)
        else:
            # LSTM(117, 128)
            self.dec_lstm = torch.nn.LSTM(self.soc_embedding_size + self.dyn_embedding_size, self.decoder_size)

        # Output layers: 本来是结合3个横向换道特征和2个纵向制动特征，现在是没有或没有挖掘数据
        self.op = torch.nn.Linear(self.decoder_size,5)
        self.op_lat = torch.nn.Linear(self.soc_embedding_size + self.dyn_embedding_size, self.num_lat_classes)
        self.op_lon = torch.nn.Linear(self.soc_embedding_size + self.dyn_embedding_size, self.num_lon_classes)

        # Activations: # 激活函数
        self.leaky_relu = torch.nn.LeakyReLU(0.1)
        self.relu = torch.nn.ReLU()
        self.softmax = torch.nn.Softmax(dim=1) # 非线性转换输出概率分布，第二个维度


    ## Forward Pass 
    def forward(self,hist,nbrs,masks,lat_enc,lon_enc):
        ## Forward pass hist:
        _,(hist_enc,_) = self.enc_lstm(self.leaky_relu(self.ip_emb(hist)))
        hist_enc = self.leaky_relu(self.dyn_emb(hist_enc.view(hist_enc.shape[1],hist_enc.shape[2])))

        ## Forward pass nbrs

        _, (nbrs_enc,_) = self.enc_lstm(self.leaky_relu(self.ip_emb(nbrs)))
        nbrs_enc = nbrs_enc.view(nbrs_enc.shape[1], nbrs_enc.shape[2])

        ## Masked scatter # 分散
        soc_enc = torch.zeros_like(masks).float()
        soc_enc = soc_enc.masked_scatter_(masks, nbrs_enc)
        soc_enc = soc_enc.permute(0,3,2,1)
        ## Apply convolutional social pooling # 应用卷积社会池
        soc_enc = self.soc_maxpool(self.leaky_relu(self.conv_3x1(self.leaky_relu(self.soc_conv(soc_enc)))))
        soc_enc = soc_enc.view(-1,self.soc_embedding_size)
        ## Apply fc soc pooling
        # soc_enc = soc_enc.contiguous()
        # soc_enc = soc_enc.view(-1, self.soc_conv_depth * self.grid_size[0] * self.grid_size[1])
        # soc_enc = self.leaky_relu(self.soc_fc(soc_enc))
        ## Concatenate encodings:
        enc = torch.cat((soc_enc,hist_enc),1)


        if self.use_maneuvers:
            ## Maneuver recognition:
            lat_pred = self.softmax(self.op_lat(enc))
            lon_pred = self.softmax(self.op_lon(enc))

            if self.train_flag:
                ## Concatenate maneuver encoding of the true maneuver
                enc = torch.cat((enc, lat_enc, lon_enc), 1)
                fut_pred = self.decode(enc)
                return fut_pred, lat_pred, lon_pred
            else:
                # 测试，取所有可能性的平均？
                fut_pred = []
                ## Predict trajectory distributions for each maneuver class
                for k in range(self.num_lon_classes):
                    for l in range(self.num_lat_classes):
                        lat_enc_tmp = torch.zeros_like(lat_enc)
                        lon_enc_tmp = torch.zeros_like(lon_enc)
                        lat_enc_tmp[:, l] = 1
                        lon_enc_tmp[:, k] = 1
                        enc_tmp = torch.cat((enc, lat_enc_tmp, lon_enc_tmp), 1)
                        fut_pred.append(self.decode(enc_tmp))
                return fut_pred, lat_pred, lon_pred
        else:
            fut_pred = self.decode(enc)
            return fut_pred


    def decode(self,enc):
        enc = enc.repeat(self.out_length, 1, 1)
        h_dec, _ = self.dec_lstm(enc)
        h_dec = h_dec.permute(1, 0, 2)
        fut_pred = self.op(h_dec)
        fut_pred = fut_pred.permute(1, 0, 2)
        fut_pred = outputActivation(fut_pred)
        return fut_pred





