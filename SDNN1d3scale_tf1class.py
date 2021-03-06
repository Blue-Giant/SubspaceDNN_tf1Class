"""
@author: LXA
 Date: 2021 年 10 月 31 日
"""
import os
import sys
import tensorflow as tf
import numpy as np
import matplotlib
import platform
import shutil
import time
import DNN_tools
import DNN_data
import DNN_Class_base

import MS_LaplaceEqs
import MS_BoltzmannEqs
import General_Laplace

import plotData
import saveData
import DNN_Print_Log


class SD2NN(object):
    def __init__(self, input_dim=4, out_dim=1, hidden2Normal=None, hidden2Scale1=None, hidden2Scale2=None,
                 Model_name2Normal='DNN', Model_name2Scale1='DNN', Model_name2Scale2='DNN', actIn_name2Normal='relu',
                 actHidden_name2Normal='relu', actOut_name2Normal='linear', actIn_name2Scale='relu',
                 actHidden_name2Scale='relu', actOut_name2Scale='linear', opt2regular_WB='L2', type2numeric='float32',
                 freq2Normal=None, freq2Scale1=None, freq2Scale2=None, sFourier2Normal=1.0, sFourier2Scale1=1.0,
                 sFourier2Scale2=1.0):
        super(SD2NN, self).__init__()
        if 'DNN' == str.upper(Model_name2Normal):
            self.DNN2Normal = DNN_Class_base.Pure_Dense_Net(
                indim=input_dim, outdim=out_dim, hidden_units=hidden2Normal, name2Model=Model_name2Normal,
                actName2in=actIn_name2Normal, actName=actHidden_name2Normal, actName2out=actOut_name2Normal,
                type2float=type2numeric, scope2W='W_Normal', scope2B='B_Normal')
        elif 'SCALE_DNN' == str.upper(Model_name2Normal):
            self.DNN2Normal = DNN_Class_base.Dense_ScaleNet(
                indim=input_dim, outdim=out_dim, hidden_units=hidden2Normal, name2Model=Model_name2Normal,
                actName2in=actIn_name2Normal, actName=actHidden_name2Normal, actName2out=actOut_name2Normal,
                type2float=type2numeric, scope2W='W_Normal', scope2B='B_Normal', repeat_high_freq=False)
        elif 'FOURIER_DNN' == str.upper(Model_name2Normal):
            self.DNN2Normal = DNN_Class_base.Dense_FourierNet(
                indim=input_dim, outdim=out_dim, hidden_units=hidden2Normal, name2Model=Model_name2Normal,
                actName2in=actIn_name2Normal, actName=actHidden_name2Normal, actName2out=actOut_name2Normal,
                type2float=type2numeric, scope2W='W_Normal', scope2B='B_Normal', repeat_high_freq=False)

        if 'DNN' == str.upper(Model_name2Scale1):
            self.DNN2Scale1 = DNN_Class_base.Pure_Dense_Net(
                indim=input_dim, outdim=out_dim, hidden_units=hidden2Scale1, name2Model=Model_name2Scale1,
                actName2in=actIn_name2Scale, actName=actHidden_name2Scale, actName2out=actOut_name2Scale,
                type2float=type2numeric, scope2W='W_Scale1', scope2B='B_Scale1')
        elif 'SCALE_DNN' == str.upper(Model_name2Scale1):
            self.DNN2Scale1 = DNN_Class_base.Dense_ScaleNet(
                indim=input_dim, outdim=out_dim, hidden_units=hidden2Scale1, name2Model=Model_name2Scale1,
                actName2in=actIn_name2Scale, actName=actHidden_name2Scale, actName2out=actOut_name2Scale,
                type2float=type2numeric, scope2W='W_Scale1', scope2B='B_Scale1')
        elif 'FOURIER_DNN' == str.upper(Model_name2Scale1):
            self.DNN2Scale1 = DNN_Class_base.Dense_FourierNet(
                indim=input_dim, outdim=out_dim, hidden_units=hidden2Scale1, name2Model=Model_name2Scale1,
                actName2in=actIn_name2Scale, actName=actHidden_name2Scale, actName2out=actOut_name2Scale,
                type2float=type2numeric, scope2W='W_Scale1', scope2B='B_Scale1')

        if 'DNN' == str.upper(Model_name2Scale2):
            self.DNN2Scale2 = DNN_Class_base.Pure_Dense_Net(
                indim=input_dim, outdim=out_dim, hidden_units=hidden2Scale2, name2Model=Model_name2Scale2,
                actName2in=actIn_name2Scale, actName=actHidden_name2Scale, actName2out=actOut_name2Scale,
                type2float=type2numeric, scope2W='W_Scale2', scope2B='B_Scale2')
        elif 'SCALE_DNN' == str.upper(Model_name2Scale2):
            self.DNN2Scale2 = DNN_Class_base.Dense_ScaleNet(
                indim=input_dim, outdim=out_dim, hidden_units=hidden2Scale2, name2Model=Model_name2Scale2,
                actName2in=actIn_name2Scale, actName=actHidden_name2Scale, actName2out=actOut_name2Scale,
                type2float=type2numeric, scope2W='W_Scale2', scope2B='B_Scale2')
        elif 'FOURIER_DNN' == str.upper(Model_name2Scale2):
            self.DNN2Scale2 = DNN_Class_base.Dense_FourierNet(
                indim=input_dim, outdim=out_dim, hidden_units=hidden2Scale2, name2Model=Model_name2Scale2,
                actName2in=actIn_name2Scale, actName=actHidden_name2Scale, actName2out=actOut_name2Scale,
                type2float=type2numeric, scope2W='W_Scale2', scope2B='B_Scale2')

        if type2numeric == 'float32':
            self.float_type = tf.float32
        elif type2numeric == 'float64':
            self.float_type = tf.float64
        elif type2numeric == 'float16':
            self.float_type = tf.float16

        self.freq2Normal = freq2Normal
        self.freq2Scale1 = freq2Scale1
        self.freq2Scale2 = freq2Scale2
        self.opt2regular_WB = opt2regular_WB
        self.sFourier2Normal = sFourier2Normal
        self.sFourier2Scale1 = sFourier2Scale1
        self.sFourier2Scale2 = sFourier2Scale2

    def loss_it2Laplace(self, X=None, fside=None, if_lambda2fside=True, loss_type='ritz_loss', alpha1=0.05, alpha2=0.01,
                        opt2orthogonal=1):
        assert (X is not None)
        assert (fside is not None)

        shape2X = X.get_shape().as_list()
        lenght2X_shape = len(shape2X)
        assert (lenght2X_shape == 2)
        assert (shape2X[-1] == 1)

        if if_lambda2fside:
            force_side = fside(X)
        else:
            force_side = fside

        UNN_Normal = self.DNN2Normal(X, scale=self.freq2Normal, sFourier=self.sFourier2Normal)
        UNN_Scale1 = self.DNN2Scale1(X, scale=self.freq2Scale1, sFourier=self.sFourier2Scale1)
        UNN_Scale2 = self.DNN2Scale2(X, scale=self.freq2Scale2, sFourier=self.sFourier2Scale2)

        UNN = UNN_Normal + alpha1 * UNN_Scale1 + alpha2 * UNN_Scale2

        dUNN_Normal = tf.gradients(UNN_Normal, X)[0]
        dUNN_Scale1 = tf.gradients(UNN_Scale1, X)[0]
        dUNN_Scale2 = tf.gradients(UNN_Scale1, X)[0]
        dUNN = dUNN_Normal + alpha1 * dUNN_Scale1 + alpha2 * dUNN_Scale2

        if str.lower(loss_type) == 'ritz_loss' or str.lower(loss_type) == 'variational_loss':
            dUNN_Norm = tf.reshape(tf.abs(dUNN), shape=[-1, 1])
            dUNN_2Norm = tf.square(dUNN_Norm)
            loss_it_ritz = (1.0/2)*dUNN_2Norm-tf.multiply(tf.reshape(force_side, shape=[-1, 1]), UNN)
            loss_it = tf.reduce_mean(loss_it_ritz)
        elif str.lower(loss_type) == 'l2_loss':
            ddUNN_Normal = tf.gradients(dUNN_Normal, X)[0]
            ddUNN_Scale1 = tf.gradients(dUNN_Scale1, X)[0]
            ddUNN_Scale2 = tf.gradients(dUNN_Scale2, X)[0]
            loss_it_L2 = ddUNN_Normal + alpha1*ddUNN_Scale1 + alpha2*ddUNN_Scale2 + \
                         tf.reshape(force_side, shape=[-1, 1])
            square_loss_it = tf.square(loss_it_L2)
            loss_it = tf.reduce_mean(square_loss_it)

        if opt2orthogonal == 1:
            # |Uc*Uf|^2-->0; Uc 和 Uf 是两个列向量 形状为(*,1)
            Un_dot_Us1 = tf.multiply(UNN_Normal, alpha1 * UNN_Scale1)
            Un_dot_Us2 = tf.multiply(UNN_Normal, alpha2 * UNN_Scale2)
            Us1_dot_Us2 = tf.multiply(alpha1 * UNN_Scale1, alpha2 * UNN_Scale2)
            square_Un_dot_Us1 = tf.square(Un_dot_Us1)
            square_Un_dot_Us2 = tf.square(Un_dot_Us2)
            square_Us1_dot_Us2 = tf.square(Us1_dot_Us2)
            UNN_dot_UNN = tf.reduce_mean(square_Un_dot_Us1) + tf.reduce_mean(square_Un_dot_Us2) + \
                          tf.reduce_mean(square_Us1_dot_Us2)

        return UNN, loss_it, UNN_dot_UNN

    def loss_it2pLaplace(self, X=None, Aeps=None, if_lambda2Aeps=True, fside=None, if_lambda2fside=True,
                         loss_type='ritz_loss', p_index=2, alpha1=0.05, alpha2=0.01, opt2orthogonal=1):
        assert (X is not None)
        assert (fside is not None)
        assert (Aeps is not None)

        shape2X = X.get_shape().as_list()
        lenght2X_shape = len(shape2X)
        assert (lenght2X_shape == 2)
        assert (shape2X[-1] == 1)

        if if_lambda2Aeps:
            a_eps = Aeps(X)  # * 行 1 列
        else:
            a_eps = Aeps
        if if_lambda2fside:
            force_side = fside(X)
        else:
            force_side = fside

        UNN_Normal = self.DNN2Normal(X, scale=self.freq2Normal, sFourier=self.sFourier2Normal)
        UNN_Scale1 = self.DNN2Scale1(X, scale=self.freq2Scale1, sFourier=self.sFourier2Scale1)
        UNN_Scale2 = self.DNN2Scale2(X, scale=self.freq2Scale2, sFourier=self.sFourier2Scale2)

        UNN = UNN_Normal + alpha1 * UNN_Scale1 + alpha2 * UNN_Scale2

        dUNN_Normal = tf.gradients(UNN_Normal, X)[0]
        dUNN_Scale1 = tf.gradients(UNN_Scale1, X)[0]
        dUNN_Scale2 = tf.gradients(UNN_Scale1, X)[0]
        dUNN = dUNN_Normal + alpha1 * dUNN_Scale1 + alpha2 * dUNN_Scale2

        # 变分形式的loss of interior，训练得到的 UNN 是 * 行 1 列
        if str.lower(loss_type) == 'ritz_loss' or str.lower(loss_type) == 'variational_loss':
            dUNN_Norm = tf.abs(dUNN)
            AdUNN_pNorm = tf.multiply(a_eps, tf.pow(dUNN_Norm, p_index))
            loss_it_ritz = (1.0/p_index)*AdUNN_pNorm-tf.multiply(tf.reshape(force_side, shape=[-1, 1]), UNN)
            loss_it = tf.reduce_mean(loss_it_ritz)

        if opt2orthogonal == 1:
            # |Uc*Uf|^2-->0; Uc 和 Uf 是两个列向量 形状为(*,1)
            Un_dot_Us1 = tf.multiply(UNN_Normal, alpha1 * UNN_Scale1)
            Un_dot_Us2 = tf.multiply(UNN_Normal, alpha2 * UNN_Scale2)
            Us1_dot_Us2 = tf.multiply(alpha1 * UNN_Scale1, alpha2 * UNN_Scale2)
            square_Un_dot_Us1 = tf.square(Un_dot_Us1)
            square_Un_dot_Us2 = tf.square(Un_dot_Us2)
            square_Us1_dot_Us2 = tf.square(Us1_dot_Us2)
            UNN_dot_UNN = tf.reduce_mean(square_Un_dot_Us1) + tf.reduce_mean(square_Un_dot_Us2) + \
                          tf.reduce_mean(square_Us1_dot_Us2)

        return UNN, loss_it, UNN_dot_UNN

    def loss_it2Possion_Boltzmann(self, X=None, Aeps=None, if_lambda2Aeps=True, Kappa_eps=None, if_lambda2Kappa=True,
                                  fside=None, if_lambda2fside=True, loss_type='ritz_loss', p_index=2,
                                  alpha1=0.05, alpha2=0.01, opt2orthogonal=1):
        assert (X is not None)
        assert (fside is not None)
        assert (Aeps is not None)
        assert (Kappa_eps is not None)

        shape2X = X.get_shape().as_list()
        lenght2X_shape = len(shape2X)
        assert (lenght2X_shape == 2)
        assert (shape2X[-1] == 1)

        if if_lambda2Aeps:
            a_eps = Aeps(X)  # * 行 1 列
        else:
            a_eps = Aeps

        if if_lambda2Kappa:
            Kappa = Kappa_eps(X)
        else:
            Kappa = Kappa_eps

        if if_lambda2fside:
            force_side = fside(X)
        else:
            force_side = fside

        UNN_Normal = self.DNN2Normal(X, scale=self.freq2Normal, sFourier=self.sFourier2Normal)
        UNN_Scale1 = self.DNN2Scale1(X, scale=self.freq2Scale1, sFourier=self.sFourier2Scale1)
        UNN_Scale2 = self.DNN2Scale2(X, scale=self.freq2Scale2, sFourier=self.sFourier2Scale2)

        UNN = UNN_Normal + alpha1 * UNN_Scale1 + alpha2 * UNN_Scale2

        dUNN_Normal = tf.gradients(UNN_Normal, X)[0]
        dUNN_Scale1 = tf.gradients(UNN_Scale1, X)[0]
        dUNN_Scale2 = tf.gradients(UNN_Scale1, X)[0]
        dUNN = dUNN_Normal + alpha1 * dUNN_Scale1 + alpha2 * dUNN_Scale2

        if str.lower(loss_type) == 'ritz_loss' or str.lower(loss_type) == 'variational_loss':
            dUNN_Norm = tf.reshape(tf.sqrt(tf.reduce_sum(tf.square(dUNN), axis=-1)), shape=[-1, 1])  # 按行求和
            AdUNN_pNorm = tf.multiply(a_eps, tf.pow(dUNN_Norm, p_index))
            loss_it_ritz = (1.0 / p_index) * (AdUNN_pNorm + Kappa*UNN*UNN) - \
                           tf.multiply(tf.reshape(force_side, shape=[-1, 1]), UNN)
            loss_it = tf.reduce_mean(loss_it_ritz)

        if opt2orthogonal == 1:
            # |Uc*Uf|^2-->0; Uc 和 Uf 是两个列向量 形状为(*,1)
            Un_dot_Us1 = tf.multiply(UNN_Normal, alpha1 * UNN_Scale1)
            Un_dot_Us2 = tf.multiply(UNN_Normal, alpha2 * UNN_Scale2)
            Us1_dot_Us2 = tf.multiply(alpha1 * UNN_Scale1, alpha2 * UNN_Scale2)
            square_Un_dot_Us1 = tf.square(Un_dot_Us1)
            square_Un_dot_Us2 = tf.square(Un_dot_Us2)
            square_Us1_dot_Us2 = tf.square(Us1_dot_Us2)
            UNN_dot_UNN = tf.reduce_mean(square_Un_dot_Us1) + tf.reduce_mean(square_Un_dot_Us2) + \
                          tf.reduce_mean(square_Us1_dot_Us2)

        return UNN, loss_it, UNN_dot_UNN

    def loss2Normal_bd(self, X_bd=None, Ubd_exact=None, if_lambda2Ubd=True):
        assert (X_bd is not None)
        assert (Ubd_exact is not None)

        shape2X = X_bd.get_shape().as_list()
        lenght2X_shape = len(shape2X)
        assert (lenght2X_shape == 2)
        assert (shape2X[-1] == 1)

        if if_lambda2Ubd:
            Ubd = Ubd_exact(X_bd)
        else:
            Ubd = Ubd_exact

        UNN_bd = self.DNN2Normal(X_bd, scale=self.freq2Normal, sFourier=self.sFourier2Normal)
        loss_bd_square = tf.square(UNN_bd - Ubd)
        loss_bd = tf.reduce_mean(loss_bd_square)
        return loss_bd

    def loss2Scale1_bd(self, X_bd=None, alpha=0.1):
        assert (X_bd is not None)
        shape2X = X_bd.get_shape().as_list()
        lenght2X_shape = len(shape2X)
        assert (lenght2X_shape == 2)
        assert (shape2X[-1] == 1)

        UNN_bd = self.DNN2Scale1(X_bd, scale=self.freq2Scale1, sFourier=self.sFourier2Scale1)
        loss_bd_square = tf.square(alpha*UNN_bd)
        loss_bd = tf.reduce_mean(loss_bd_square)
        return loss_bd

    def loss2Scale2_bd(self, X_bd=None, alpha=0.01):
        assert (X_bd is not None)
        shape2X = X_bd.get_shape().as_list()
        lenght2X_shape = len(shape2X)
        assert (lenght2X_shape == 2)
        assert (shape2X[-1] == 1)

        UNN_bd = self.DNN2Scale2(X_bd, scale=self.freq2Scale2, sFourier=self.sFourier2Scale2)
        loss_bd_square = tf.square(alpha*UNN_bd)
        loss_bd = tf.reduce_mean(loss_bd_square)
        return loss_bd

    def loss_bd2NormalAddScale(self, X_bd=None, Ubd_exact=None, alpha1=0.1, alpha2=0.01, if_lambda2Ubd=True):
        assert (X_bd is not None)
        assert (Ubd_exact is not None)

        shape2X = X_bd.get_shape().as_list()
        lenght2X_shape = len(shape2X)
        assert (lenght2X_shape == 2)
        assert (shape2X[-1] == 1)

        if if_lambda2Ubd:
            Ubd = Ubd_exact(X_bd)
        else:
            Ubd = Ubd_exact

        UNN_bd2Normal = self.DNN2Normal(X_bd, scale=self.freq2Normal, sFourier=self.sFourier2Normal)
        UNN_bd2Scale1 = self.DNN2Scale1(X_bd, scale=self.freq2Scale1, sFourier=self.sFourier2Scale1)
        UNN_bd2Scale2 = self.DNN2Scale2(X_bd, scale=self.freq2Scale2, sFourier=self.sFourier2Scale2)
        UNN_bd = UNN_bd2Normal + alpha1*UNN_bd2Scale1 + alpha2*UNN_bd2Scale2
        loss_bd_square = tf.square(UNN_bd - Ubd)
        loss_bd = tf.reduce_mean(loss_bd_square)
        return loss_bd

    def get_regularSum2WB(self):
        sum2WB = self.DNN2Normal.get_regular_sum2WB(self.opt2regular_WB) + \
                 self.DNN2Scale1.get_regular_sum2WB(self.opt2regular_WB) + \
                 self.DNN2Scale2.get_regular_sum2WB(self.opt2regular_WB)
        return sum2WB

    def evalue_MscaleDNN(self, X_points=None, alpha1=0.1, alpha2=0.1):
        assert (X_points is not None)
        shape2X = X_points.get_shape().as_list()
        lenght2X_shape = len(shape2X)
        assert (lenght2X_shape == 2)
        assert (shape2X[-1] == 1)

        UNN_Normal = self.DNN2Normal(X_points, scale=self.freq2Normal, sFourier=self.sFourier2Normal)
        UNN_Scale1 = alpha1 * self.DNN2Scale1(X_points, scale=self.freq2Scale1, sFourier=self.sFourier2Scale1)
        UNN_Scale2 = alpha2 * self.DNN2Scale2(X_points, scale=self.freq2Scale2, sFourier=self.sFourier2Scale2)
        UNN = UNN_Normal + UNN_Scale1 + UNN_Scale2
        return UNN_Normal, UNN_Scale1, UNN_Scale2, UNN


def solve_Multiscale_PDE(R):
    log_out_path = R['FolderName']
    if not os.path.exists(log_out_path):  # 判断路径是否已经存在
        os.mkdir(log_out_path)            # 无 log_out_path 路径，创建一个 log_out_path 路径

    outfile_name1 = '%s%s.txt' % ('log2', 'train')
    log_fileout_NN = open(os.path.join(log_out_path, outfile_name1), 'w')  # 在这个路径下创建并打开一个可写的 log_train.txt文件
    DNN_Print_Log.log_dictionary_3Scale(R, log_fileout_NN, actName2normal=R['actHidden_name2Normal'],
                                        actName2scale=R['actHidden_name2Scale'])

    batchsize_it = R['batch_size2interior']
    batchsize_bd = R['batch_size2boundary']
    init_bd_penalty = R['init_boundary_penalty']         # Regularization parameter for boundary conditions
    lr_decay = R['learning_rate_decay']
    learning_rate = R['learning_rate']

    init_UdotU_penalty = R['init_penalty2orthogonal']
    penalty2WB = R['penalty2weight_biases']              # Regularization parameter for weights and biases

    # ------- set the problem ---------
    input_dim = R['input_dim']
    out_dim = R['output_dim']
    act_func2Normal = R['actHidden_name2Normal']
    act_func2Scale = R['actHidden_name2Scale']

    region_l = 0.0
    region_r = 1.0
    if R['PDE_type'] == 'pLaplace':
        # 求解如下方程, A_eps(x) 震荡的比较厉害，具有多个尺度
        #       d      ****         d         ****
        #   -  ----   |  A_eps(x)* ---- u_eps(x) |  =f(x), x \in R^n
        #       dx     ****         dx        ****
        p_index = R['order2pLaplace_operator']
        epsilon1 = R['epsilon1']
        epsilon2 = R['epsilon2']
        region_l = 0.0
        region_r = 1.0
        u_true, f, A_eps, u_left, u_right = MS_LaplaceEqs.get_infos2pLaplace_1D_2(
            in_dim=input_dim, out_dim=out_dim, intervalL=region_l, intervalR=region_r, index2p=p_index, eps1=epsilon1,
            eps2=epsilon2)
    elif R['PDE_type'] == 'Possion_Boltzmann':
        # 求解如下方程, A_eps(x) 震荡的比较厉害，具有多个尺度
        #       d      ****         d         ****
        #   -  ----   |  A_eps(x)* ---- u_eps(x) | + K(x)u_eps(x) =f(x), x \in R^n
        #       dx     ****         dx        ****
        p_index = R['order2pLaplace_operator']
        epsilon = R['epsilon']
        region_l = 0.0
        region_r = 1.0
        A_eps, kappa, u_true, u_left, u_right, f = MS_BoltzmannEqs.get_infos2Boltzmann_1D(
            in_dim=input_dim, out_dim=out_dim, region_a=region_l, region_b=region_r, index2p=p_index, eps=epsilon,
            eqs_name=R['equa_name'])

    sd2nn = SD2NN(input_dim=R['input_dim'], out_dim=1, hidden2Normal=R['hidden2normal'],
                  hidden2Scale1=R['hidden2scale1'], hidden2Scale2=R['hidden2scale2'],
                  Model_name2Normal=R['model2Normal'], Model_name2Scale1=R['model2Scale1'],
                  Model_name2Scale2=R['model2Scale2'], actIn_name2Normal=R['actHidden_name2Normal'],
                  actHidden_name2Normal=R['actHidden_name2Normal'], actOut_name2Normal='linear',
                  actIn_name2Scale=R['actHidden_name2Scale'], actHidden_name2Scale=R['actHidden_name2Scale'],
                  actOut_name2Scale='linear', opt2regular_WB='L2', type2numeric='float32', freq2Normal=R['freq2Normal'],
                  freq2Scale1=R['freq2Scale1'], freq2Scale2=R['freq2Scale2'], sFourier2Normal=R['sFourier2Normal'],
                  sFourier2Scale1=R['sFourier2MesoScale'], sFourier2Scale2=R['sFourier2FineScale'])

    global_steps = tf.compat.v1.Variable(0, trainable=False)
    with tf.device('/gpu:%s' % (R['gpuNo'])):
        with tf.compat.v1.variable_scope('vscope', reuse=tf.compat.v1.AUTO_REUSE):
            X_it = tf.compat.v1.placeholder(tf.float32, name='X_it', shape=[None, input_dim])                # * 行 1 列
            X_left = tf.compat.v1.placeholder(tf.float32, name='X_left', shape=[None, input_dim])      # * 行 1 列
            X_right = tf.compat.v1.placeholder(tf.float32, name='X_right', shape=[None, input_dim])    # * 行 1 列
            bd_penalty = tf.compat.v1.placeholder_with_default(input=1e3, shape=[], name='bd_p')
            UdotU_penalty = tf.compat.v1.placeholder_with_default(input=1.0, shape=[], name='p_powU')
            in_learning_rate = tf.compat.v1.placeholder_with_default(input=1e-5, shape=[], name='lr')

            if R['PDE_type'] == 'general_Laplace':
                UNN2train, Loss_it2NNs, UNN_dot_UNN = sd2nn.loss_it2Laplace(
                    X=X_it, fside=f, loss_type=R['loss_type'], alpha1=R['contrib2scale1'], alpha2=R['contrib2scale2'],
                    opt2orthogonal=R['opt2orthogonal'])
            elif R['PDE_type'] == 'pLaplace':
                fx = MS_LaplaceEqs.force_sice_3scale2(X_it, eps1=R['epsilon1'], eps2=R['epsilon2'])
                UNN2train, Loss_it2NNs, UNN_dot_UNN = sd2nn.loss_it2pLaplace(
                    X=X_it, Aeps=A_eps, fside=fx, if_lambda2fside=False, loss_type=R['loss_type'],
                    alpha1=R['contrib2scale1'], alpha2=R['contrib2scale2'], opt2orthogonal=R['opt2orthogonal'])
            elif R['PDE_type'] == 'Possion_Boltzmann':
                UNN2train, Loss_it2NNs, UNN_dot_UNN = sd2nn.loss_it2Possion_Boltzmann()

            if R['opt2loss_udotu'] == 'with_orthogonal':
                Loss2UNN_dot_UNN = UdotU_penalty * UNN_dot_UNN
            else:
                Loss2UNN_dot_UNN = tf.constant(0.0)

            if R['opt2loss_bd'] == 'unified_boundary':
                loss_bd2left = sd2nn.loss_bd2NormalAddScale(X_left, Ubd_exact=u_left, alpha1=R['contrib2scale1'],
                                                            alpha2=R['contrib2scale2'])
                loss_bd2right = sd2nn.loss_bd2NormalAddScale(X_right, Ubd_exact=u_right, alpha1=R['contrib2scale1'],
                                                             alpha2=R['contrib2scale2'])
                Loss_bd2NNs = bd_penalty * (loss_bd2left + loss_bd2right)
            else:
                loss_bd2Normal_left = sd2nn.loss2Normal_bd(X_left, Ubd_exact=u_left)
                loss_bd2Normal_right = sd2nn.loss2Normal_bd(X_right, Ubd_exact=u_right)
                loss_bd2Normal = loss_bd2Normal_left + loss_bd2Normal_right

                loss_bd2Scale1_left = sd2nn.loss2Scale1_bd(X_left, alpha=R['contrib2scale1'])
                loss_bd2Scale1_right = sd2nn.loss2Scale1_bd(X_right, alpha=R['contrib2scale1'])
                loss_bd2Scale1 = loss_bd2Scale1_left + loss_bd2Scale1_right

                loss_bd2Scale2_left = sd2nn.loss2Scale2_bd(X_left, alpha=R['contrib2scale2'])
                loss_bd2Scale2_right = sd2nn.loss2Scale2_bd(X_right, alpha=R['contrib2scale2'])
                loss_bd2Scale2 = loss_bd2Scale2_left + loss_bd2Scale2_right

                Loss_bd2NNs = bd_penalty*(loss_bd2Normal + loss_bd2Scale1 + loss_bd2Scale2)

            regularSum2WB = sd2nn.get_regularSum2WB()
            PWB = penalty2WB * regularSum2WB

            Loss2NN = Loss_it2NNs + Loss_bd2NNs + Loss2UNN_dot_UNN + PWB

            my_optimizer = tf.train.AdamOptimizer(in_learning_rate)
            if R['loss_type'] == 'variational_loss' or R['loss_type'] == 'variational_loss2':
                if R['train_model'] == 'training_group4_1':
                    train_op1 = my_optimizer.minimize(Loss_it2NNs, global_step=global_steps)
                    train_op2 = my_optimizer.minimize(Loss_bd2NNs, global_step=global_steps)
                    train_op3 = my_optimizer.minimize(Loss2UNN_dot_UNN, global_step=global_steps)
                    train_op4 = my_optimizer.minimize(Loss2NN, global_step=global_steps)
                    train_Loss2NN = tf.group(train_op1, train_op2, train_op3, train_op4)
                elif R['train_model'] == 'training_group3':
                    train_op1 = my_optimizer.minimize(Loss_it2NNs, global_step=global_steps)
                    train_op2 = my_optimizer.minimize(Loss_bd2NNs, global_step=global_steps)
                    train_op3 = my_optimizer.minimize(Loss2NN, global_step=global_steps)
                    train_Loss2NN = tf.group(train_op1, train_op2, train_op3)
                elif R['train_model'] == 'training_group4_2':
                    train_op1 = my_optimizer.minimize(Loss_it2NNs, global_step=global_steps)
                    train_op2 = my_optimizer.minimize(Loss_bd2NNs, global_step=global_steps)
                    train_op3 = my_optimizer.minimize(Loss2UNN_dot_UNN, global_step=global_steps)
                    train_op4 = my_optimizer.minimize(Loss2NN, global_step=global_steps)
                    train_Loss2NN = tf.group(train_op1, train_op2, train_op3, train_op4)
                elif R['train_model'] == 'training_group2':
                    train_op3 = my_optimizer.minimize(Loss2UNN_dot_UNN, global_step=global_steps)
                    train_op4 = my_optimizer.minimize(Loss2NN, global_step=global_steps)
                    train_Loss2NN = tf.group(train_op3, train_op4)
                else:
                    train_Loss2NN = my_optimizer.minimize(Loss2NN, global_step=global_steps)
            # 训练上的真解值和训练结果的误差
            U_true = u_true(X_it)
            train_mse_NN = tf.reduce_mean(tf.square(U_true - UNN2train))
            train_rel_NN = train_mse_NN / tf.reduce_mean(tf.square(U_true))

            UNN_Normal2test, UNN_Scale12test, UNN_Scale22test, UNN2test = \
                sd2nn.evalue_MscaleDNN(X_points=X_it, alpha1=R['contrib2scale1'], alpha2=R['contrib2scale2'])

    t0 = time.time()
    loss_it_all, loss_bd_all, loss_all, loss_udu_all, train_mse_all, train_rel_all = [], [], [], [], [], []
    test_mse_all, test_rel_all = [], []
    test_epoch = []

    test_batch_size = 1000
    test_x_bach = np.reshape(np.linspace(region_l, region_r, num=test_batch_size), [-1, 1])
    saveData.save_testData_or_solus2mat(test_x_bach, dataName='testx', outPath=R['FolderName'])

    # ConfigProto 加上allow_soft_placement=True就可以使用 gpu 了
    config = tf.compat.v1.ConfigProto(allow_soft_placement=True)  # 创建sess的时候对sess进行参数配置
    config.gpu_options.allow_growth = True              # True是让TensorFlow在运行过程中动态申请显存，避免过多的显存占用。
    config.allow_soft_placement = True                  # 当指定的设备不存在时，允许选择一个存在的设备运行。比如gpu不存在，自动降到cpu上运行
    with tf.compat.v1.Session(config=config) as sess:
        sess.run(tf.compat.v1.global_variables_initializer())
        tmp_lr = learning_rate

        for i_epoch in range(R['max_epoch'] + 1):
            x_it_batch = DNN_data.rand_it(batchsize_it, input_dim, region_a=region_l, region_b=region_r)
            xl_bd_batch, xr_bd_batch = DNN_data.rand_bd_1D(batchsize_bd, input_dim, region_a=region_l, region_b=region_r)
            tmp_lr = tmp_lr * (1 - lr_decay)
            if R['activate_penalty2bd_increase'] == 1:
                if i_epoch < int(R['max_epoch'] / 10):
                    temp_penalty_bd = init_bd_penalty
                elif i_epoch < int(R['max_epoch'] / 5):
                    temp_penalty_bd = 10 * init_bd_penalty
                elif i_epoch < int(R['max_epoch'] / 4):
                    temp_penalty_bd = 50 * init_bd_penalty
                elif i_epoch < int(R['max_epoch'] / 2):
                    temp_penalty_bd = 100 * init_bd_penalty
                elif i_epoch < int(3 * R['max_epoch'] / 4):
                    temp_penalty_bd = 200 * init_bd_penalty
                else:
                    temp_penalty_bd = 500 * init_bd_penalty
            else:
                temp_penalty_bd = init_bd_penalty

            if R['activate_powSolus_increase'] == 1:
                if i_epoch < int(R['max_epoch'] / 10):
                    temp_penalty_powU = init_UdotU_penalty
                elif i_epoch < int(R['max_epoch'] / 5):
                    temp_penalty_powU = 10* init_UdotU_penalty
                elif i_epoch < int(R['max_epoch'] / 4):
                    temp_penalty_powU = 50*init_UdotU_penalty
                elif i_epoch < int(R['max_epoch'] / 2):
                    temp_penalty_powU = 100*init_UdotU_penalty
                elif i_epoch < int(3 * R['max_epoch'] / 4):
                    temp_penalty_powU = 200*init_UdotU_penalty
                else:
                    temp_penalty_powU = 500*init_UdotU_penalty
            else:
                temp_penalty_powU = init_UdotU_penalty

            _, loss_it_nn, loss_bd_nn, loss_nn, udu_nn, train_mse_nn, train_rel_nn, pwb = sess.run(
                [train_Loss2NN, Loss_it2NNs, Loss_bd2NNs, Loss2NN, UNN_dot_UNN, train_mse_NN, train_rel_NN, PWB],
                feed_dict={X_it: x_it_batch, X_left: xl_bd_batch, X_right: xr_bd_batch,
                           in_learning_rate: tmp_lr, bd_penalty: temp_penalty_bd, UdotU_penalty: temp_penalty_powU})
            loss_it_all.append(loss_it_nn)
            loss_bd_all.append(loss_bd_nn)
            loss_all.append(loss_nn)
            loss_udu_all.append(udu_nn)
            train_mse_all.append(train_mse_nn)
            train_rel_all.append(train_rel_nn)

            if i_epoch % 1000 == 0:
                run_times = time.time() - t0
                DNN_tools.print_and_log_train_one_epoch(
                    i_epoch, run_times, tmp_lr, temp_penalty_bd, temp_penalty_powU, pwb, loss_it_nn, loss_bd_nn, loss_nn,
                    udu_nn, train_mse_nn, train_rel_nn, log_out=log_fileout_NN)

                # ---------------------------   test network ----------------------------------------------
                test_epoch.append(i_epoch / 1000)
                u_true2test, utest_nn, unn_normal, unn_scale1, unn_scale2 = sess.run(
                    [U_true, UNN2test, UNN_Normal2test, UNN_Scale12test, UNN_Scale22test],
                    feed_dict={X_it: test_x_bach})
                test_mse2nn = np.mean(np.square(u_true2test - utest_nn))
                test_mse_all.append(test_mse2nn)
                test_rel2nn = test_mse2nn / np.mean(np.square(u_true2test))
                test_rel_all.append(test_rel2nn)

                DNN_tools.print_and_log_test_one_epoch(test_mse2nn, test_rel2nn, log_out=log_fileout_NN)

    # -----------------------  save training results to mat files, then plot them ---------------------------------
    saveData.save_trainLoss2mat_1actFunc(loss_it_all, loss_bd_all, loss_all, actName=act_func2Normal,
                                         outPath=R['FolderName'])

    saveData.save_train_MSE_REL2mat(train_mse_all, train_rel_all, actName=act_func2Normal, outPath=R['FolderName'])

    plotData.plotTrain_loss_1act_func(loss_it_all, lossType='loss_it', seedNo=R['seed'], outPath=R['FolderName'])
    plotData.plotTrain_loss_1act_func(loss_bd_all, lossType='loss_bd', seedNo=R['seed'], outPath=R['FolderName'],
                                      yaxis_scale=True)
    plotData.plotTrain_loss_1act_func(loss_all, lossType='loss', seedNo=R['seed'], outPath=R['FolderName'])

    plotData.plotTrain_MSE_REL_1act_func(train_mse_all, train_rel_all, actName=act_func2Scale, seedNo=R['seed'],
                                         outPath=R['FolderName'], yaxis_scale=True)

    # ----------------------  save testing results to mat files, then plot them --------------------------------
    saveData.save_testData_or_solus2mat(u_true2test, dataName='Utrue', outPath=R['FolderName'])
    saveData.save_testData_or_solus2mat(utest_nn, dataName=act_func2Normal, outPath=R['FolderName'])
    saveData.save_testData_or_solus2mat(unn_normal, dataName='normal', outPath=R['FolderName'])
    saveData.save_testData_or_solus2mat(unn_scale1, dataName='scale1', outPath=R['FolderName'])
    saveData.save_testData_or_solus2mat(unn_scale2, dataName='scale2', outPath=R['FolderName'])

    saveData.save_testMSE_REL2mat(test_mse_all, test_rel_all, actName=act_func2Scale, outPath=R['FolderName'])
    plotData.plotTest_MSE_REL(test_mse_all, test_rel_all, test_epoch, actName=act_func2Scale, seedNo=R['seed'],
                              outPath=R['FolderName'], yaxis_scale=True)


if __name__ == "__main__":
    R = {}
    R['gpuNo'] = 0
    if platform.system() == 'Windows':
        os.environ["CDUA_VISIBLE_DEVICES"] = "%s" % (R['gpuNo'])
    else:
        print('-------------------------------------- linux -----------------------------------------------')
        # Linux终端没有GUI, 需要添加如下代码，而且必须添加在 import matplotlib.pyplot 之前，否则无效。
        matplotlib.use('Agg')

        if tf.test.is_gpu_available():
            os.environ["CUDA_VISIBLE_DEVICES"] = "0,1,2,3"  # 设置当前使用的GPU设备仅为第 0,1,2,3 块GPU, 设备名称为'/gpu:0'
        else:
            os.environ["CUDA_VISIBLE_DEVICES"] = "1"

    # ------------------------------------------- 文件保存路径设置 ----------------------------------------
    store_file = 'pLaplace1D'
    # store_file = 'Boltzmann1D'
    # store_file = 'Convection1D'
    BASE_DIR = os.path.dirname(os.path.abspath(__file__))
    sys.path.append(BASE_DIR)
    OUT_DIR = os.path.join(BASE_DIR, store_file)
    if not os.path.exists(OUT_DIR):
        print('---------------------- OUT_DIR ---------------------:', OUT_DIR)
        os.mkdir(OUT_DIR)

    R['seed'] = np.random.randint(1e5)
    seed_str = str(R['seed'])  # int 型转为字符串型
    FolderName = os.path.join(OUT_DIR, seed_str)  # 路径连接
    R['FolderName'] = FolderName
    if not os.path.exists(FolderName):
        print('--------------------- FolderName -----------------:', FolderName)
        os.mkdir(FolderName)

    # ----------------------------------------  复制并保存当前文件 -----------------------------------------
    if platform.system() == 'Windows':
        tf.compat.v1.reset_default_graph()
        shutil.copy(__file__, '%s/%s' % (FolderName, os.path.basename(__file__)))
    else:
        shutil.copy(__file__, '%s/%s' % (FolderName, os.path.basename(__file__)))

    # ---------------------------- Setup of laplace equation ------------------------------
    # if the value of step_stop_flag is not 0, it will activate stop condition of step to kill program
    step_stop_flag = input('please input an  integer number to activate step-stop----0:no---!0:yes--:')
    R['activate_stop'] = int(step_stop_flag)
    # if the value of step_stop_flag is not 0, it will activate stop condition of step to kill program
    R['max_epoch'] = 200000
    if 0 != R['activate_stop']:
        epoch_stop = input('please input a stop epoch:')
        R['max_epoch'] = int(epoch_stop)

    if store_file == 'Laplace1D':
        R['PDE_type'] = 'general_Laplace'
        R['equa_name'] = 'PDE1'
        # R['equa_name'] = 'PDE2'
        # R['equa_name'] = 'PDE3'
        # R['equa_name'] = 'PDE4'
        # R['equa_name'] = 'PDE5'
        # R['equa_name'] = 'PDE6'
        # R['equa_name'] = 'PDE7'
    elif store_file == 'pLaplace1D':
        R['PDE_type'] = 'pLaplace'
        R['equa_name'] = '3scale2'
        # R['equa_name'] = '3scale3'
    elif store_file == 'Boltzmann1D':
        R['PDE_type'] = 'Possion_Boltzmann'
        # R['equa_name'] = 'Boltzmann1'
        R['equa_name'] = 'Boltzmann2'

    if R['PDE_type'] == 'pLaplace' or R['PDE_type'] == 'Possion_Boltzmann':
        # 尺度设置
        epsilon1 = input('please input epsilon1 =')  # 由终端输入的会记录为字符串形式
        R['epsilon1'] = float(epsilon1)  # 字符串转为浮点

        epsilon2 = input('please input epsilon2 =')  # 由终端输入的会记录为字符串形式
        R['epsilon2'] = float(epsilon2)  # 字符串转为浮点

        # 问题幂次
        order2pLaplace = input('please input the order(a int number) to p-laplace:')
        order = float(order2pLaplace)
        R['order2pLaplace_operator'] = order

    R['input_dim'] = 1  # 输入维数，即问题的维数(几元问题)
    R['output_dim'] = 1  # 输出维数
    R['loss_type'] = 'variational_loss'  # PDE变分
    # R['loss_type'] = 'variational_loss2'  # PDE变分
    # R['loss_type'] = 'L2_loss'                # L2 loss

    # R['opt2orthogonal'] = 0                    # 0: L2 opt2orthogonal+energy    1: opt2orthogonal    2:energy
    R['opt2orthogonal'] = 1                      # 0: L2 opt2orthogonal+energy    1: opt2orthogonal    2:energy
    # R['opt2orthogonal'] = 2                    # 0: L2 opt2orthogonal+energy    1: opt2orthogonal    2:energy

    # ---------------------------- Setup of DNN -------------------------------
    R['batch_size2interior'] = 3000  # 内部训练数据的批大小
    R['batch_size2boundary'] = 500  # 边界训练数据大小

    R['regular_wb_model'] = 'L0'
    # R['regular_wb_model'] = 'L1'
    # R['regular_wb_model'] = 'L2'
    R['penalty2weight_biases'] = 0.000  # Regularization parameter for weights
    # R['penalty2weight_biases'] = 0.001   # Regularization parameter for weights
    # R['penalty2weight_biases'] = 0.0025  # Regularization parameter for weights

    R['activate_penalty2bd_increase'] = 1
    R['init_boundary_penalty'] = 100  # Regularization parameter for boundary conditions

    R['activate_powSolus_increase'] = 0
    if R['activate_powSolus_increase'] == 1:
        R['init_penalty2orthogonal'] = 5.0
    elif R['activate_powSolus_increase'] == 2:
        R['init_penalty2orthogonal'] = 10000.0
    else:
        R['init_penalty2orthogonal'] = 20.0
        # R['init_penalty2orthogonal'] = 15.0
        # R['init_penalty2orthogonal'] = 10.0

    R['learning_rate'] = 2e-4  # 学习率
    R['learning_rate_decay'] = 5e-5  # 学习率 decay
    R['optimizer_name'] = 'Adam'  # 优化器
    R['train_model'] = 'training_union'  # 训练模式, 一个 loss 联结训练
    # R['train_model'] = 'training_group1'          # 训练模式, 多个 loss 组团训练
    # R['train_model'] = 'training_group2'
    # R['train_model'] = 'training_group3'
    # R['train_model'] = 'training_group4'

    # R['model2Normal'] = 'DNN'  # 使用的网络模型
    # R['model2Normal'] = 'Scale_DNN'
    R['model2Normal'] = 'Fourier_DNN'
    # R['model2Normal'] = 'Sin+Cos_DNN'

    # R['model2Scale1'] = 'DNN'                    # 使用的网络模型
    # R['model2Scale1'] = 'Scale_DNN'
    R['model2Scale1'] = 'Fourier_DNN'

    # R['model2Scale2'] = 'DNN'                    # 使用的网络模型
    # R['model2Scale2'] = 'Scale_DNN'
    R['model2Scale2'] = 'Fourier_DNN'

    # normal 和 scale 网络的总参数数目:12520 + 29360 = 41880
    if R['model2Normal'] == 'Fourier_DNN':
        R['hidden2normal'] = (50, 80, 60, 60, 40)  # 1*50+100*80+80*60+60*60+60*40+40*1 = 18890个参数
    else:
        R['hidden2normal'] = (100, 80, 60, 60, 40)  # 1*100+100*80+80*60+60*60+60*40+40*1 = 18940个参数
        # R['hidden2normal'] = (200, 100, 100, 80, 80, 50)
        # R['hidden2normal'] = (300, 200, 200, 100, 100, 50)
        # R['hidden2normal'] = (500, 400, 300, 200, 100)

    if R['model2Scale1'] == 'Fourier_DNN':
        if R['epsilon1'] == 0.1:
            R['hidden2scale1'] = (125, 80, 60, 60, 40)  # 1*125+250*80+80*60+60*60+60*40+40*1=30925 个参数
        else:
            R['hidden2scale1'] = (225, 200, 150, 150, 100)  # 1*225+450*200+200*150+150*150+150*100+100*1=157825 个参数
    else:
        if R['epsilon1'] == 0.1:
            R['hidden2scale1'] = (160, 100, 80, 80, 60)  # 1*200+200*60+60*60+60*50+50*40+40*1=20840 个参数
        else:
            R['hidden2scale1'] = (250, 200, 150, 150, 100)  # 1*250+250*60+60*60+60*60+60*50+50*1=25500 个参数

    if R['model2Scale2'] == 'Fourier_DNN':
        if R['epsilon2'] == 0.1:
            R['hidden2scale2'] = (125, 80, 60, 60, 40)  # 1*125+250*80+80*60+60*60+60*40+40*1=30925 个参数
        else:
            R['hidden2scale2'] = (225, 200, 150, 150, 100)  # 1*225+450*200+200*150+150*150+150*100+100*1=157825 个参数
    else:
        if R['epsilon2'] == 0.1:
            R['hidden2scale2'] = (160, 100, 80, 80, 60)  # 1*200+200*60+60*60+60*50+50*40+40*1=20840 个参数
        else:
            R['hidden2scale2'] = (250, 200, 150, 150, 100)  # 1*250+250*60+60*60+60*60+60*50+50*1=25500 个参数

    # 激活函数的选择
    # R['actHidden_name2Normal'] = 'relu'
    R['actHidden_name2Normal'] = 'tanh'
    # R['actHidden_name2Normal'] = 'srelu'
    # R['actHidden_name2Normal'] = 'sin'
    # R['actHidden_name2Normal'] = 's2relu'

    if R['model2Normal'] == 'Fourier_DNN':
        # R['actHidden_name2Normal'] = 's2relu'
        R['actHidden_name2Normal'] = 'tanh'

    # R['actHidden_name2Scale'] = 'relu'
    # R['actHidden_name2Scale']' = leaky_relu'
    # R['actHidden_name2Scale'] = 'srelu'
    R['actHidden_name2Scale'] = 's2relu'
    # R['actHidden_name2Scale'] = 'tanh'
    # R['actHidden_name2Scale'] = 'elu'
    # R['actHidden_name2Scale'] = 'phi'

    if R['loss_type'] == 'L2_loss':
        R['actHidden_name2Scale'] = 'tanh'

    R['plot_ongoing'] = 0
    R['subfig_type'] = 0
    # R['freq2Normal'] = np.concatenate(
    # ([1, 1, 1, 2, 2, 2, 3, 3, 3, 4, 4, 4, 5, 5, 5, 6, 6, 6, 7, 7, 7, 8, 8, 8, 9, 9, 9], np.arange(10, 21)), axis=0)
    # R['freq2Normal'] = np.concatenate(
    #    ([1, 1, 1, 2, 2, 2, 3, 3, 3, 4, 4, 4, 5, 5, 5, 6, 6, 6, 7, 7, 7, 8, 8, 8, 9, 9, 9], np.arange(10, 26)), axis=0)
    R['freq2Normal'] = np.concatenate(
        ([1, 1, 1, 2, 2, 2, 3, 3, 3, 4, 4, 4, 5, 5, 5, 6, 6, 6, 7, 7, 7, 8, 8, 8, 9, 9, 9], np.arange(10, 31)), axis=0)
    # R['freqs'] = np.arange(5, 100)
    R['freq2Scale1'] = np.arange(10, 60)
    R['freq2Scale2'] = np.arange(50, 100)
    # R['freq2Scale2'] = np.arange(60, 111)

    if R['loss_type'] == 'variational_loss':
        # R['contrib2scale'] = 0.01
        R['contrib2scale1'] = 0.05
        # R['contrib2scale1'] = 0.06
        # R['contrib2scale1'] = 0.07
        # R['contrib2scale1'] = 0.075
        R['contrib2scale2'] = 0.01
        # R['contrib2scale2'] = 0.0075
        # R['contrib2scale2'] = 0.008
    elif R['loss_type'] == 'variational_loss2':
        # R['contrib2scale'] = 0.01
        R['contrib2scale1'] = 0.25
        # R['contrib2scale1'] = 0.06
        # R['contrib2scale1'] = 0.07
        # R['contrib2scale1'] = 0.075
        R['contrib2scale2'] = 0.075
        # R['contrib2scale2'] = 0.0075
        # R['contrib2scale2'] = 0.008

    R['opt2loss_udotu'] = 'with_orthogonal'
    R['opt2loss_bd'] = 'unified_boundary'

    R['sFourier2Normal'] = 1.0
    R['sFourier2MesoScale'] = 0.5
    R['sFourier2FineScale'] = 0.5

    solve_Multiscale_PDE(R)


