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
import DNN_Class_base

import MS_LaplaceEqs
import MS_BoltzmannEqs
import General_Laplace
import MS_ConvectionEqs

import DNN_data
import Load_data2Mat

import plotData
import saveData
import DNN_Print_Log


class SD2NN(object):
    def __init__(self, input_dim=3, out_dim=1, hidden2Normal=None, hidden2Scale=None, Model_name2Normal='DNN',
                 Model_name2Scale='DNN', actIn_name2Normal='relu', actHidden_name2Normal='relu',
                 actOut_name2Normal='linear', actIn_name2Scale='relu', actHidden_name2Scale='relu',
                 actOut_name2Scale='linear', opt2regular_WB='L2', type2numeric='float32', freq2Normal=None,
                 freq2Scale=None):
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

        if 'DNN' == str.upper(Model_name2Scale):
            self.DNN2Scale = DNN_Class_base.Pure_Dense_Net(
                indim=input_dim, outdim=out_dim, hidden_units=hidden2Scale, name2Model=Model_name2Scale,
                actName2in=actIn_name2Scale, actName=actHidden_name2Scale, actName2out=actOut_name2Scale,
                type2float=type2numeric, scope2W='W_Scale', scope2B='B_Scale')
        elif 'SCALE_DNN' == str.upper(Model_name2Scale):
            self.DNN2Scale = DNN_Class_base.Dense_ScaleNet(
                indim=input_dim, outdim=out_dim, hidden_units=hidden2Scale, name2Model=Model_name2Scale,
                actName2in=actIn_name2Scale, actName=actHidden_name2Scale, actName2out=actOut_name2Scale,
                type2float=type2numeric, scope2W='W_Scale', scope2B='B_Scale')
        elif 'FOURIER_DNN' == str.upper(Model_name2Scale):
            self.DNN2Scale = DNN_Class_base.Dense_FourierNet(
                indim=input_dim, outdim=out_dim, hidden_units=hidden2Scale, name2Model=Model_name2Scale,
                actName2in=actIn_name2Scale, actName=actHidden_name2Scale, actName2out=actOut_name2Scale,
                type2float=type2numeric, scope2W='W_Scale', scope2B='B_Scale')

        if type2numeric == 'float32':
            self.float_type = tf.float32
        elif type2numeric == 'float64':
            self.float_type = tf.float64
        elif type2numeric == 'float16':
            self.float_type = tf.float16

        self.freq2Normal = freq2Normal
        self.freq2Scale = freq2Scale
        self.opt2regular_WB = opt2regular_WB

    def loss_it2Laplace(self, X=None, fside=None, if_lambda2fside=True, loss_type='ritz_loss', alpha=0.1,
                        opt2use_orthogonal=True, opt2orthogonal=1):
        assert(X is not None)
        assert (fside is not None)

        shape2X = X.get_shape().as_list()
        lenght2X_shape = len(shape2X)
        assert(lenght2X_shape == 2)
        assert(shape2X[-1] == 3)
        x1 = tf.reshape(X[:, 0], shape=[-1, 1])
        x2 = tf.reshape(X[:, 1], shape=[-1, 1])
        x3 = tf.reshape(X[:, 2], shape=[-1, 1])

        if if_lambda2fside:
            force_side = fside(x1, x2, x3)
        else:
            force_side = fside

        UNN_Normal = self.DNN2Normal(X, scale=self.freq2Normal)
        UNN_Scale = self.DNN2Scale(X, scale=self.freq2Scale)

        UNN = UNN_Normal + alpha * UNN_Scale

        dUNN_Normal = tf.gradients(UNN_Normal, X)[0]
        dUNN_Scale = tf.gradients(UNN_Scale, X)[0]
        dUNN = tf.add(dUNN_Normal, alpha * dUNN_Scale)

        if str.lower(loss_type) == 'ritz_loss' or str.lower(loss_type) == 'variational_loss':
            dUNN_Norm = tf.reshape(tf.sqrt(tf.reduce_sum(tf.square(dUNN), axis=-1)), shape=[-1, 1])
            dUNN_2Norm = tf.square(dUNN_Norm)
            loss_it_ritz = (1.0/2)*dUNN_2Norm-tf.multiply(tf.reshape(force_side, shape=[-1, 1]), UNN)
            loss_it = tf.reduce_mean(loss_it_ritz)
        elif str.lower(loss_type) == 'l2_loss':
            dUNN_Normal2x = tf.gather(dUNN_Normal, [0], axis=-1)
            dUNN_Normal2y = tf.gather(dUNN_Normal, [1], axis=-1)
            ddUNN_Normal2xxy = tf.gradients(dUNN_Normal2x, X)[0]
            ddUNN_Normal2yxy = tf.gradients(dUNN_Normal2y, X)[0]
            ddUNN_Normal2xx = tf.gather(ddUNN_Normal2xxy, [0], axis=-1)
            ddUNN_Normal2yy = tf.gather(ddUNN_Normal2yxy, [1], axis=-1)

            dUNN_Scale2x = tf.gather(dUNN_Scale, [0], axis=-1)
            dUNN_Scale2y = tf.gather(dUNN_Scale, [1], axis=-1)
            ddUNN_Scale2xxy = tf.gradients(dUNN_Scale2x, X)[0]
            ddUNN_Scale2yxy = tf.gradients(dUNN_Scale2y, X)[0]
            ddUNN_Scale2xx = tf.gather(ddUNN_Scale2xxy, [0], axis=-1)
            ddUNN_Scale2yy = tf.gather(ddUNN_Scale2yxy, [1], axis=-1)

            loss_it_L2 = ddUNN_Normal2xx + ddUNN_Normal2yy + alpha*ddUNN_Scale2xx + alpha*ddUNN_Scale2yy + \
                         tf.reshape(force_side, shape=[-1, 1])
            square_loss_it = tf.square(loss_it_L2)
            loss_it = tf.reduce_mean(square_loss_it)

        if opt2use_orthogonal == 'with_orthogonal':
            if opt2orthogonal == 0:              # L2 正交
                Un_dot_Us = tf.multiply(UNN_Normal, alpha * UNN_Scale)
                UNN_dot_UNN = tf.square(tf.reduce_mean(Un_dot_Us))
            elif opt2orthogonal == 1:
                # |Uc*Uf|^2-->0; Uc 和 Uf 是两个列向量 形状为(*,1)
                Un_dot_Us = tf.multiply(UNN_Normal, alpha * UNN_Scale)
                square_Un_dot_Us = tf.square(Un_dot_Us)
                UNN_dot_UNN = tf.reduce_mean(square_Un_dot_Us)
            elif opt2orthogonal == 2:
                # |a(x)*(grad Uc)*(grad Uf)|^2-->0 a(x) 是 (*,1)的；(grad Uc)*(grad Uf)是向量相乘(*,2)·(*,2)
                dU_dot_dU = tf.multiply(dUNN_Normal, alpha * dUNN_Scale)
                sum2dUdU = tf.reshape(tf.reduce_sum(dU_dot_dU, axis=-1), shape=[-1, 1])
                norm2AdUdU = tf.square(sum2dUdU)
                UNN_dot_UNN = tf.reduce_mean(norm2AdUdU)
            else:  # |Uc*Uf|^2-->0 + |a(x)*(grad Uc)*(grad Uf)|^2-->0
                U_dot_U = tf.reshape(tf.square(tf.multiply(UNN_Normal, alpha * UNN_Scale)), shape=[-1, 1])
                dU_dot_dU = tf.multiply(dUNN_Normal, alpha * dUNN_Scale)
                sum2dUdU = tf.reshape(tf.reduce_sum(dU_dot_dU, axis=-1), shape=[-1, 1])
                norm2AdUdU = tf.square(sum2dUdU)
                UNN_dot_UNN = tf.reduce_mean(norm2AdUdU) + tf.reduce_mean(U_dot_U)
        else:
            UNN_dot_UNN = tf.constant(0.0)

        return UNN, loss_it, UNN_dot_UNN

    def loss_it2pLaplace(self, X=None, Aeps=None, if_lambda2Aeps=True, fside=None, if_lambda2fside=True,
                         loss_type='ritz_loss', p_index=2, alpha=0.1, opt2use_orthogonal=True, opt2orthogonal=1):
        assert (X is not None)
        assert (fside is not None)
        assert (Aeps is not None)

        shape2X = X.get_shape().as_list()
        lenght2X_shape = len(shape2X)
        assert (lenght2X_shape == 2)
        assert (shape2X[-1] == 3)
        x1 = tf.reshape(X[:, 0], shape=[-1, 1])
        x2 = tf.reshape(X[:, 1], shape=[-1, 1])
        x3 = tf.reshape(X[:, 2], shape=[-1, 1])

        if if_lambda2Aeps:
            a_eps = Aeps(x1, x2, x3)  # * 行 1 列
        else:
            a_eps = Aeps
        if if_lambda2fside:
            force_side = fside(x1, x2, x3)
        else:
            force_side = fside

        UNN_Normal = self.DNN2Normal(X, scale=self.freq2Normal)
        UNN_Scale = self.DNN2Scale(X, scale=self.freq2Scale)

        UNN = UNN_Normal + alpha * UNN_Scale

        dUNN_Normal = tf.gradients(UNN_Normal, X)[0]
        dUNN_Scale = tf.gradients(UNN_Scale, X)[0]
        dUNN = tf.add(dUNN_Normal, alpha * dUNN_Scale)

        # 变分形式的loss of interior，训练得到的 UNN 是 * 行 1 列
        if str.lower(loss_type) == 'ritz_loss' or str.lower(loss_type) == 'variational_loss':
            dUNN_Norm = tf.reshape(tf.sqrt(tf.reduce_sum(tf.square(dUNN), axis=-1)), shape=[-1, 1])
            AdUNN_pNorm = tf.multiply(a_eps, tf.pow(dUNN_Norm, p_index))
            loss_it_ritz = (1.0/p_index)*AdUNN_pNorm-tf.multiply(tf.reshape(force_side, shape=[-1, 1]), UNN)
            loss_it = tf.reduce_mean(loss_it_ritz)

        if opt2use_orthogonal == 'with_orthogonal':
            if opt2orthogonal == 0:              # L2 正交
                Un_dot_Us = tf.multiply(UNN_Normal, alpha * UNN_Scale)
                UNN_dot_UNN = tf.square(tf.reduce_mean(Un_dot_Us))
            elif opt2orthogonal == 1:
                # |Uc*Uf|^2-->0; Uc 和 Uf 是两个列向量 形状为(*,1)
                Un_dot_Us = tf.multiply(UNN_Normal, alpha * UNN_Scale)
                square_Un_dot_Us = tf.square(Un_dot_Us)
                UNN_dot_UNN = tf.reduce_mean(square_Un_dot_Us)
            elif opt2orthogonal == 2:
                # |a(x)*(grad Uc)*(grad Uf)|^2-->0 a(x) 是 (*,1)的；(grad Uc)*(grad Uf)是向量相乘(*,2)·(*,2)
                dU_dot_dU = tf.multiply(dUNN_Normal, alpha * dUNN_Scale)
                sum2dUdU = tf.reshape(tf.reduce_sum(dU_dot_dU, axis=-1), shape=[-1, 1])
                norm2AdUdU = tf.square(sum2dUdU)
                UNN_dot_UNN = tf.reduce_mean(norm2AdUdU)
            else:  # |Uc*Uf|^2-->0 + |a(x)*(grad Uc)*(grad Uf)|^2-->0
                U_dot_U = tf.reshape(tf.square(tf.multiply(UNN_Normal, alpha * UNN_Scale)), shape=[-1, 1])
                dU_dot_dU = tf.multiply(dUNN_Normal, alpha * dUNN_Scale)
                sum2dUdU = tf.reshape(tf.reduce_sum(dU_dot_dU, axis=-1), shape=[-1, 1])
                norm2AdUdU = tf.square(sum2dUdU)
                UNN_dot_UNN = tf.reduce_mean(norm2AdUdU) + tf.reduce_mean(U_dot_U)
        else:
            UNN_dot_UNN = tf.constant(0.0)

        return UNN, loss_it, UNN_dot_UNN

    def loss_it2Possion_Boltzmann(self, X=None, Aeps=None, if_lambda2Aeps=True, Kappa_eps=None, if_lambda2Kappa=True,
                                  fside=None, if_lambda2fside=True, loss_type='ritz_loss', p_index=2, alpha=0.1,
                                  opt2use_orthogonal=True, opt2orthogonal=1):
        assert (X is not None)
        assert (fside is not None)
        assert (Aeps is not None)
        assert (Kappa_eps is not None)

        shape2X = X.get_shape().as_list()
        lenght2X_shape = len(shape2X)
        assert (lenght2X_shape == 2)
        assert (shape2X[-1] == 3)
        x1 = tf.reshape(X[:, 0], shape=[-1, 1])
        x2 = tf.reshape(X[:, 1], shape=[-1, 1])
        x3 = tf.reshape(X[:, 2], shape=[-1, 1])

        if if_lambda2Aeps:
            a_eps = Aeps(x1, x2, x3)  # * 行 1 列
        else:
            a_eps = Aeps

        if if_lambda2Kappa:
            Kappa = Kappa_eps(x1, x2, x3)
        else:
            Kappa = Kappa_eps

        if if_lambda2fside:
            force_side = fside(x1, x2, x3)
        else:
            force_side = fside

        UNN_Normal = self.DNN2Normal(X, scale=self.freq2Normal)
        UNN_Scale = self.DNN2Scale(X, scale=self.freq2Scale)

        UNN = UNN_Normal + alpha * UNN_Scale

        dUNN_Normal = tf.gradients(UNN_Normal, X)[0]  # * 行 2 列
        dUNN_Scale = tf.gradients(UNN_Scale, X)[0]
        dUNN = tf.add(dUNN_Normal, alpha * dUNN_Scale)

        if str.lower(loss_type) == 'ritz_loss' or str.lower(loss_type) == 'variational_loss':
            dUNN_Norm = tf.reshape(tf.sqrt(tf.reduce_sum(tf.square(dUNN), axis=-1)), shape=[-1, 1])
            AdUNN_pNorm = tf.multiply(a_eps, tf.pow(dUNN_Norm, p_index))
            loss_it_ritz = (1.0 / p_index) * (AdUNN_pNorm + Kappa*UNN*UNN) - \
                           tf.multiply(tf.reshape(force_side, shape=[-1, 1]), UNN)
            loss_it = tf.reduce_mean(loss_it_ritz)

        if opt2use_orthogonal == 'with_orthogonal':
            if opt2orthogonal == 0:              # L2 正交
                Un_dot_Us = tf.multiply(UNN_Normal, alpha * UNN_Scale)
                UNN_dot_UNN = tf.square(tf.reduce_mean(Un_dot_Us))
            elif opt2orthogonal == 1:
                # |Uc*Uf|^2-->0; Uc 和 Uf 是两个列向量 形状为(*,1)
                Un_dot_Us = tf.multiply(UNN_Normal, alpha * UNN_Scale)
                square_Un_dot_Us = tf.square(Un_dot_Us)
                UNN_dot_UNN = tf.reduce_mean(square_Un_dot_Us)
            elif opt2orthogonal == 2:
                # |a(x)*(grad Uc)*(grad Uf)|^2-->0 a(x) 是 (*,1)的；(grad Uc)*(grad Uf)是向量相乘(*,2)·(*,2)
                dU_dot_dU = tf.multiply(dUNN_Normal, alpha * dUNN_Scale)
                sum2dUdU = tf.reshape(tf.reduce_sum(dU_dot_dU, axis=-1), shape=[-1, 1])
                norm2AdUdU = tf.square(sum2dUdU)
                UNN_dot_UNN = tf.reduce_mean(norm2AdUdU)
            else:  # |Uc*Uf|^2-->0 + |a(x)*(grad Uc)*(grad Uf)|^2-->0
                U_dot_U = tf.reshape(tf.square(tf.multiply(UNN_Normal, alpha * UNN_Scale)), shape=[-1, 1])
                dU_dot_dU = tf.multiply(dUNN_Normal, alpha * dUNN_Scale)
                sum2dUdU = tf.reshape(tf.reduce_sum(dU_dot_dU, axis=-1), shape=[-1, 1])
                norm2AdUdU = tf.square(sum2dUdU)
                UNN_dot_UNN = tf.reduce_mean(norm2AdUdU) + tf.reduce_mean(U_dot_U)
        else:
            UNN_dot_UNN = tf.constant(0.0)

        return UNN, loss_it, UNN_dot_UNN

    def loss2Normal_bd(self, X_bd=None, Ubd_exact=None, if_lambda2Ubd=True):
        x1_bd = tf.reshape(X_bd[:, 0], shape=[-1, 1])
        x2_bd = tf.reshape(X_bd[:, 1], shape=[-1, 1])
        x3_bd = tf.reshape(X_bd[:, 2], shape=[-1, 1])

        if if_lambda2Ubd:
            Ubd = Ubd_exact(x1_bd, x2_bd, x3_bd)
        else:
            Ubd = Ubd_exact

        UNN_bd = self.DNN2Normal(X_bd, scale=self.freq2Normal)
        loss_bd_square = tf.square(UNN_bd - Ubd)
        loss_bd = tf.reduce_mean(loss_bd_square)
        return loss_bd

    def loss2Scale_bd(self, X_bd=None, alpha=0.1):
        UNN_bd = self.DNN2Scale(X_bd, scale=self.freq2Scale)
        loss_bd_square = tf.square(alpha*UNN_bd)
        loss_bd = tf.reduce_mean(loss_bd_square)
        return loss_bd

    def loss_bd2NormalAddScale(self, X_bd=None, Ubd_exact=None, alpha=0.1, if_lambda2Ubd=True):
        x1_bd = tf.reshape(X_bd[:, 0], shape=[-1, 1])
        x2_bd = tf.reshape(X_bd[:, 1], shape=[-1, 1])
        x3_bd = tf.reshape(X_bd[:, 2], shape=[-1, 1])

        if if_lambda2Ubd:
            Ubd = Ubd_exact(x1_bd, x2_bd, x3_bd)
        else:
            Ubd = Ubd_exact

        UNN_bd2Normal = self.DNN2Normal(X_bd, scale=self.freq2Normal)
        UNN_bd2Scale = self.DNN2Scale(X_bd, scale=self.freq2Scale)
        UNN_bd = tf.add(UNN_bd2Normal, alpha*UNN_bd2Scale)
        loss_bd_square = tf.square(UNN_bd - Ubd)
        loss_bd = tf.reduce_mean(loss_bd_square)
        return loss_bd

    def get_regularSum2WB(self):
        sum2WB = self.DNN2Normal.get_regular_sum2WB(self.opt2regular_WB) + \
                 self.DNN2Scale.get_regular_sum2WB(self.opt2regular_WB)
        return sum2WB

    def evalue_MscaleDNN(self, X_points=None, alpha=0.1):
        UNN_Normal = self.DNN2Normal(X_points, scale=self.freq2Normal)
        UNN_Scale = alpha*self.DNN2Scale(X_points, scale=self.freq2Scale)
        UNN = tf.add(UNN_Normal, UNN_Scale)
        return UNN_Normal, UNN_Scale, UNN


def solve_Multiscale_PDE(R):
    log_out_path = R['FolderName']
    if not os.path.exists(log_out_path):  # 判断路径是否已经存在
        os.mkdir(log_out_path)            # 无 log_out_path 路径，创建一个 log_out_path 路径

    outfile_name1 = '%s%s.txt' % ('log2', 'train')
    log_fileout_NN = open(os.path.join(log_out_path, outfile_name1), 'w')  # 在这个路径下创建并打开一个可写的 log_train.txt文件
    DNN_Print_Log.dictionary_out2file(R, log_fileout_NN, actName2normal=R['actHidden_name2Normal'],
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

    if R['contrib_scale2orthogonal'] == 'with_contrib':
        using_scale2orthogonal = R['contrib2scale']
    else:
        using_scale2orthogonal = 1.0

    if R['opt2loss_bd'] != 'unified_boundary' and R['contrib_scale2boundary'] == 'with_contrib':
        using_scale2boundary = R['contrib2scale']
    else:
        using_scale2boundary = 1.0

    # p laplace 问题需要的额外设置, 先预设一下
    p_index = 2
    epsilon = 0.1
    mesh_number = 2

    # 问题区域，每个方向设置为一样的长度。等网格划分，对于二维是方形区域
    region_lb = 0.0
    region_rt = 1.0
    if R['PDE_type'] == 'Possion_Boltzmann':
        p_index = R['order2pLaplace_operator']
        epsilon = R['epsilon']
        mesh_number = R['mesh_number']
        if R['equa_name'] == 'Boltzmann1':
            region_lb = -1.0
            region_rt = 1.0
        else:
            region_lb = 0.0
            region_rt = 1.0
        A_eps, kappa, f, u_true, u00, u01, u10, u11, u20, u21 = MS_BoltzmannEqs.get_infos2Boltzmann_3D(
            input_dim=input_dim, out_dim=out_dim, mesh_number=R['mesh_number'], intervalL=region_lb,
            intervalR=region_rt, equa_name=R['equa_name'])
    elif R['PDE_type'] == 'pLaplace':
        # 求解如下方程, A_eps(x) 震荡的比较厉害，具有多个尺度
        #       d      ****         d         ****
        #   -  ----   |  A_eps(x)* ---- u_eps(x) |  =f(x), x \in R^n
        #       dx     ****         dx        ****
        p_index = R['order2pLaplace_operator']
        epsilon = R['epsilon']
        mesh_number = R['mesh_number']
        region_lb = 0.0
        region_rt = 1.0
        u_true, f, A_eps, u00, u01, u10, u11, u20, u21 = MS_LaplaceEqs.get_infos2pLaplace_3D(
            input_dim=input_dim, out_dim=out_dim, mesh_number=R['mesh_number'], intervalL=0.0, intervalR=1.0,
            equa_name=R['equa_name'])
    elif R['PDE_type'] == 'Convection_diffusion':
        region_lb = -1.0
        region_rt = 1.0
        p_index = R['order2pLaplace_operator']
        epsilon = R['epsilon']
        mesh_number = R['mesh_number']
        A_eps, Bx, By, u_true, u_left, u_right, u_top, u_bottom, f = MS_ConvectionEqs.get_infos2Convection_2D(
            equa_name=R['equa_name'], eps=epsilon, region_lb=0.0, region_rt=1.0)

    sd2nn = SD2NN(input_dim=R['input_dim'], out_dim=1, hidden2Normal=R['hidden2normal'], hidden2Scale=R['hidden2scale'],
                  Model_name2Normal=R['model2Normal'], Model_name2Scale=R['model2Scale'],
                  actIn_name2Normal=R['actHidden_name2Normal'], actHidden_name2Normal=R['actHidden_name2Normal'],
                  actOut_name2Normal='linear', actIn_name2Scale=R['actHidden_name2Scale'],
                  actHidden_name2Scale=R['actHidden_name2Scale'], actOut_name2Scale='linear', opt2regular_WB='L2',
                  type2numeric='float32', freq2Normal=R['freq2Normal'], freq2Scale=R['freq2Scale'])

    global_steps = tf.compat.v1.Variable(0, trainable=False)
    with tf.device('/gpu:%s' % (R['gpuNo'])):
        with tf.compat.v1.variable_scope('vscope', reuse=tf.compat.v1.AUTO_REUSE):
            XYZ_it = tf.compat.v1.placeholder(tf.float32, name='XYZ_it', shape=[None, input_dim])           # * 行 3 列
            XYZ_bottom = tf.compat.v1.placeholder(tf.float32, name='XYZ_bottom', shape=[None, input_dim])   # * 行 3 列
            XYZ_top = tf.compat.v1.placeholder(tf.float32, name='XYZ_top', shape=[None, input_dim])         # * 行 3 列
            XYZ_left = tf.compat.v1.placeholder(tf.float32, name='XYZ_left', shape=[None, input_dim])       # * 行 3 列
            XYZ_right = tf.compat.v1.placeholder(tf.float32, name='XYZ_right', shape=[None, input_dim])     # * 行 3 列
            XYZ_front = tf.compat.v1.placeholder(tf.float32, name='XYZ_front', shape=[None, input_dim])     # * 行 3 列
            XYZ_behind = tf.compat.v1.placeholder(tf.float32, name='XYZ_behind', shape=[None, input_dim])   # * 行 3 列
            bd_penalty = tf.compat.v1.placeholder_with_default(input=1e3, shape=[], name='bd_p')
            UdotU_penalty = tf.compat.v1.placeholder_with_default(input=1.0, shape=[], name='p_powU')
            in_learning_rate = tf.compat.v1.placeholder_with_default(input=1e-5, shape=[], name='lr')

            if R['PDE_type'] == 'general_Laplace':
                UNN2train, Loss_it2NNs, UNN_dot_UNN = sd2nn.loss_it2Laplace(
                    X=XYZ_it, fside=f, loss_type=R['loss_type'], alpha=using_scale2orthogonal,
                    opt2orthogonal=R['opt2orthogonal'])
            elif R['PDE_type'] == 'pLaplace' or R['PDE_type'] == 'pLaplace_implicit' or \
                    R['PDE_type'] == 'pLaplace_explicit':
                UNN2train, Loss_it2NNs, UNN_dot_UNN = sd2nn.loss_it2pLaplace(
                    X=XYZ_it, Aeps=A_eps, fside=f, loss_type=R['loss_type'], alpha=using_scale2orthogonal,
                    opt2orthogonal=R['opt2orthogonal'])
            elif R['PDE_type'] == 'Possion_Boltzmann':
                UNN2train, Loss_it2NNs, UNN_dot_UNN = sd2nn.loss_it2Possion_Boltzmann(
                    X=XYZ_it, Aeps=A_eps, fside=f, Kappa_eps=kappa, loss_type=R['loss_type'],
                    alpha=using_scale2orthogonal, opt2orthogonal=R['opt2orthogonal'])

            Loss2UNN_dot_UNN = UdotU_penalty * UNN_dot_UNN

            if R['opt2loss_bd'] == 'unified_boundary':
                loss_bd2left = sd2nn.loss_bd2NormalAddScale(XYZ_left, Ubd_exact=u00, alpha=R['contrib2scale'])
                loss_bd2right = sd2nn.loss_bd2NormalAddScale(XYZ_right, Ubd_exact=u00, alpha=R['contrib2scale'])
                loss_bd2bottom = sd2nn.loss_bd2NormalAddScale(XYZ_bottom, Ubd_exact=u00, alpha=R['contrib2scale'])
                loss_bd2top = sd2nn.loss_bd2NormalAddScale(XYZ_top, Ubd_exact=u00, alpha=R['contrib2scale'])
                loss_bd2front = sd2nn.loss_bd2NormalAddScale(XYZ_front, Ubd_exact=u00, alpha=R['contrib2scale'])
                loss_bd2behind = sd2nn.loss_bd2NormalAddScale(XYZ_behind, Ubd_exact=u00, alpha=R['contrib2scale'])
                Loss_bd2NNs = bd_penalty * (loss_bd2left + loss_bd2right + loss_bd2bottom + loss_bd2top +
                                            loss_bd2front + loss_bd2behind)
            else:
                loss_bd2Normal_left = sd2nn.loss2Normal_bd(XYZ_left, Ubd_exact=u00)
                loss_bd2Normal_right = sd2nn.loss2Normal_bd(XYZ_right, Ubd_exact=u00)
                loss_bd2Normal_bottom = sd2nn.loss2Normal_bd(XYZ_bottom, Ubd_exact=u00)
                loss_bd2Normal_top = sd2nn.loss2Normal_bd(XYZ_top, Ubd_exact=u00)
                loss_bd2Normal_front = sd2nn.loss2Normal_bd(XYZ_front, Ubd_exact=u00)
                loss_bd2Normal_behind = sd2nn.loss2Normal_bd(XYZ_behind, Ubd_exact=u00)
                loss_bd2Normal = loss_bd2Normal_left + loss_bd2Normal_right + loss_bd2Normal_bottom + \
                                 loss_bd2Normal_top + loss_bd2Normal_front + loss_bd2Normal_behind

                loss_bd2Scale_left = sd2nn.loss2Scale_bd(XYZ_left, alpha=using_scale2boundary)
                loss_bd2Scale_right = sd2nn.loss2Scale_bd(XYZ_right, alpha=using_scale2boundary)
                loss_bd2Scale_bottom = sd2nn.loss2Scale_bd(XYZ_bottom, alpha=using_scale2boundary)
                loss_bd2Scale_top = sd2nn.loss2Scale_bd(XYZ_top, alpha=using_scale2boundary)
                loss_bd2Scale_front = sd2nn.loss2Scale_bd(XYZ_front, alpha=using_scale2boundary)
                loss_bd2Scale_behind = sd2nn.loss2Scale_bd(XYZ_behind, alpha=using_scale2boundary)
                loss_bd2Scale = loss_bd2Scale_left + loss_bd2Scale_right + loss_bd2Scale_bottom + loss_bd2Scale_top + \
                                loss_bd2Scale_front + loss_bd2Scale_behind

                Loss_bd2NNs = bd_penalty*(loss_bd2Normal + loss_bd2Scale)

            regularSum2WB = sd2nn.get_regularSum2WB()
            PWB = penalty2WB * regularSum2WB

            Loss2NN = Loss_it2NNs + Loss_bd2NNs + Loss2UNN_dot_UNN + PWB

            my_optimizer = tf.compat.v1.train.AdamOptimizer(in_learning_rate)
            if R['loss_type'] == 'variational_loss':
                if R['train_model'] == 'training_group2':
                    train_op1 = my_optimizer.minimize(Loss2NN, global_step=global_steps)
                    train_op2 = my_optimizer.minimize(Loss_bd2NNs, global_step=global_steps)
                    train_Loss2NN = tf.group(train_op1, train_op2)
                elif R['train_model'] == 'training_group3':
                    train_op1 = my_optimizer.minimize(Loss_it2NNs, global_step=global_steps)
                    train_op2 = my_optimizer.minimize(Loss_bd2NNs, global_step=global_steps)
                    train_op3 = my_optimizer.minimize(Loss2NN, global_step=global_steps)
                    train_Loss2NN = tf.group(train_op1, train_op2, train_op3)
                elif R['train_model'] == 'training_union':
                    train_Loss2NN = my_optimizer.minimize(Loss2NN, global_step=global_steps)
            elif R['loss_type'] == 0 or R['loss_type'] == 'variational_loss2':
                if R['train_model'] == 'training_group2':
                    train_op3 = my_optimizer.minimize(Loss2UNN_dot_UNN, global_step=global_steps)
                    train_op4 = my_optimizer.minimize(Loss2NN, global_step=global_steps)
                    train_Loss2NN = tf.group(train_op3, train_op4)
                elif R['train_model'] == 'training_group3':
                    train_op1 = my_optimizer.minimize(Loss_it2NNs, global_step=global_steps)
                    train_op2 = my_optimizer.minimize(Loss_bd2NNs, global_step=global_steps)
                    train_op3 = my_optimizer.minimize(Loss2NN, global_step=global_steps)
                    train_Loss2NN = tf.group(train_op1, train_op2, train_op3)
                elif R['train_model'] == 'training_group4':
                    train_op1 = my_optimizer.minimize(Loss_it2NNs, global_step=global_steps)
                    train_op2 = my_optimizer.minimize(Loss_bd2NNs, global_step=global_steps)
                    train_op3 = my_optimizer.minimize(Loss2UNN_dot_UNN, global_step=global_steps)
                    train_op4 = my_optimizer.minimize(Loss2NN, global_step=global_steps)
                    train_Loss2NN = tf.group(train_op1, train_op2, train_op3, train_op4)
                elif R['train_model'] == 'training_group4_1':
                    train_op1 = my_optimizer.minimize(Loss_it2NNs, global_step=global_steps)
                    train_op2 = my_optimizer.minimize(Loss_bd2NNs, global_step=global_steps)
                    train_op3 = my_optimizer.minimize(Loss2UNN_dot_UNN, global_step=global_steps)
                    train_op4 = my_optimizer.minimize(Loss2NN, global_step=global_steps)
                    train_Loss2NN = tf.group(train_op1, train_op2, train_op3, train_op4)
                elif R['train_model'] == 'training_union':
                    train_Loss2NN = my_optimizer.minimize(Loss2NN, global_step=global_steps)

            # 训练上的真解值和训练结果的误差
            if R['PDE_type'] == 'general_Laplace' or R['PDE_type'] == 'pLaplace_explicit' \
                    or R['PDE_type'] == 'Possion_Boltzmann' or R['PDE_type'] == 'Convection_diffusion':
                X_it = tf.reshape(XYZ_it[:, 0], shape=[-1, 1])
                Y_it = tf.reshape(XYZ_it[:, 1], shape=[-1, 1])
                Z_it = tf.reshape(XYZ_it[:, 1], shape=[-1, 1])
                U_true = u_true(X_it, Y_it, Z_it)
                train_mse_NN = tf.reduce_mean(tf.square(U_true - UNN2train))
                train_rel_NN = train_mse_NN / tf.reduce_mean(tf.square(U_true))
            else:
                train_mse_NN = tf.constant(0.0)
                train_rel_NN = tf.constant(0.0)

            UNN_Normal2test, UNN_Scale2test, UNN2test = sd2nn.evalue_MscaleDNN(X_points=XYZ_it, alpha=R['contrib2scale'])

    t0 = time.time()
    loss_it_all, loss_bd_all, loss_all, loss_udu_all, train_mse_all, train_rel_all = [], [], [], [], [], []
    test_mse_all, test_rel_all = [], []
    test_epoch = []

    # 画网格解图
    if R['testData_model'] == 'random_generate':
        # 生成测试数据，用于测试训练后的网络
        # test_bach_size = 400
        # size2test = 20
        # test_bach_size = 900
        # size2test = 30
        test_bach_size = 1600
        size2test = 40
        # test_bach_size = 4900
        # size2test = 70
        # test_bach_size = 10000
        # size2test = 100
        test_xyz_bach = DNN_data.rand_it(test_bach_size, input_dim, region_lb, region_rt)
        saveData.save_testData_or_solus2mat(test_xyz_bach, dataName='testXYZ', outPath=R['FolderName'])
    elif R['testData_model'] == 'loadData':
        test_bach_size = 1600
        size2test = 40
        mat_data_path = 'dataMat_highDim'
        test_xyz_bach = Load_data2Mat.get_randomData2mat(dim=input_dim, data_path=mat_data_path)
        saveData.save_testData_or_solus2mat(test_xyz_bach, dataName='testXYZ', outPath=R['FolderName'])

    # ConfigProto 加上allow_soft_placement=True就可以使用 gpu 了
    config = tf.compat.v1.ConfigProto(allow_soft_placement=True)  # 创建sess的时候对sess进行参数配置
    config.gpu_options.allow_growth = True              # True是让TensorFlow在运行过程中动态申请显存，避免过多的显存占用。
    config.allow_soft_placement = True                  # 当指定的设备不存在时，允许选择一个存在的设备运行。比如gpu不存在，自动降到cpu上运行
    with tf.compat.v1.Session(config=config) as sess:
        sess.run(tf.global_variables_initializer())
        tmp_lr = learning_rate

        for i_epoch in range(R['max_epoch'] + 1):
            xyz_it_batch = DNN_data.rand_it(batchsize_it, input_dim, region_a=region_lb, region_b=region_rt)
            xyz_bottom_batch, xyz_top_batch, xyz_left_batch, xyz_right_batch, xyz_front_batch, xyz_behind_batch = \
                DNN_data.rand_bd_3D(batchsize_bd, input_dim, region_a=region_lb, region_b=region_rt)
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
            elif R['activate_penalty2bd_increase'] == 2:
                if i_epoch < int(R['max_epoch'] / 10):
                    temp_penalty_bd = 5 * init_bd_penalty
                elif i_epoch < int(R['max_epoch'] / 5):
                    temp_penalty_bd = 1 * init_bd_penalty
                elif i_epoch < int(R['max_epoch'] / 4):
                    temp_penalty_bd = 0.5 * init_bd_penalty
                elif i_epoch < int(R['max_epoch'] / 2):
                    temp_penalty_bd = 0.1 * init_bd_penalty
                elif i_epoch < int(3 * R['max_epoch'] / 4):
                    temp_penalty_bd = 0.05 * init_bd_penalty
                else:
                    temp_penalty_bd = 0.02 * init_bd_penalty
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
                feed_dict={XYZ_it: xyz_it_batch, XYZ_left: xyz_left_batch, XYZ_right: xyz_right_batch,
                           XYZ_bottom: xyz_bottom_batch, XYZ_top: xyz_top_batch, XYZ_front: xyz_front_batch,
                           XYZ_behind: xyz_behind_batch, in_learning_rate: tmp_lr, bd_penalty: temp_penalty_bd,
                           UdotU_penalty: temp_penalty_powU})
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
                if R['PDE_type'] == 'general_Laplace' or R['PDE_type'] == 'pLaplace_explicit' or \
                        R['PDE_type'] == 'Possion_Boltzmann' or R['PDE_type'] == 'Convection_diffusion':
                    u_true2test, utest_nn, unn_normal, unn_scale = sess.run(
                        [U_true, UNN2test, UNN_Normal2test, UNN_Scale2test], feed_dict={XYZ_it: test_xyz_bach})
                else:
                    u_true2test = u_true
                    utest_nn, unn_normal, unn_scale = sess.run(
                        [UNN2test, UNN_Normal2test, UNN_Scale2test], feed_dict={XYZ_it: test_xyz_bach})
                point_ERR2NN = np.square(u_true2test - utest_nn)
                test_mse2nn = np.mean(point_ERR2NN)
                test_mse_all.append(test_mse2nn)
                test_rel2nn = test_mse2nn / np.mean(np.square(u_true2test))
                test_rel_all.append(test_rel2nn)

                DNN_tools.print_and_log_test_one_epoch(test_mse2nn, test_rel2nn, log_out=log_fileout_NN)

    # ------------------- save the testing results into mat file and plot them -------------------------
    saveData.save_trainLoss2mat_1actFunc(loss_it_all, loss_bd_all, loss_all, actName=act_func2Normal,
                                         outPath=R['FolderName'])
    saveData.save_train_MSE_REL2mat(train_mse_all, train_rel_all, actName=act_func2Normal, outPath=R['FolderName'])

    plotData.plotTrain_loss_1act_func(loss_it_all, lossType='loss_it', seedNo=R['seed'],
                                      outPath=R['FolderName'])
    plotData.plotTrain_loss_1act_func(loss_bd_all, lossType='loss_bd', seedNo=R['seed'],
                                      outPath=R['FolderName'],
                                      yaxis_scale=True)
    plotData.plotTrain_loss_1act_func(loss_all, lossType='loss', seedNo=R['seed'], outPath=R['FolderName'])
    plotData.plotTrain_loss_1act_func(loss_udu_all, lossType='udu', seedNo=R['seed'], outPath=R['FolderName'])

    plotData.plotTrain_MSE_REL_1act_func(train_mse_all, train_rel_all, actName=act_func2Normal, seedNo=R['seed'],
                                         outPath=R['FolderName'], yaxis_scale=True)

    # ----------------- save test data to mat file and plot the testing results into figures -----------------------
    if R['PDE_type'] == 'general_laplace' or R['PDE_type'] == 'pLaplace_explicit':
        saveData.save_testData_or_solus2mat(u_true2test, dataName='Utrue', outPath=R['FolderName'])

    saveData.save_testData_or_solus2mat(utest_nn, dataName='test', outPath=R['FolderName'])
    saveData.save_testData_or_solus2mat(unn_normal, dataName='normal', outPath=R['FolderName'])
    saveData.save_testData_or_solus2mat(unn_scale, dataName='scale', outPath=R['FolderName'])

    if R['hot_power'] == 1:
        plotData.plot_Hot_solution2test(u_true2test, size_vec2mat=size2test, actName='Utrue', seedNo=R['seed'],
                                        outPath=R['FolderName'])
        plotData.plot_Hot_solution2test(utest_nn, size_vec2mat=size2test, actName=act_func2Normal, seedNo=R['seed'],
                                        outPath=R['FolderName'])

    saveData.save_testMSE_REL2mat(test_mse_all, test_rel_all, actName=act_func2Normal, outPath=R['FolderName'])
    saveData.save_test_point_wise_err2mat(point_ERR2NN, actName=act_func2Normal, outPath=R['FolderName'])

    plotData.plotTest_MSE_REL(test_mse_all, test_rel_all, test_epoch, actName=act_func2Normal, seedNo=R['seed'],
                              outPath=R['FolderName'], yaxis_scale=True)
    plotData.plot_Hot_point_wise_err(point_ERR2NN, size_vec2mat=size2test, actName=act_func2Normal,
                                     seedNo=R['seed'], outPath=R['FolderName'])


if __name__ == "__main__":
    R = {}
    # -------------------------------------- CPU or GPU 选择 -----------------------------------------------
    R['gpuNo'] = 0
    # 默认使用 GPU，这个标记就不要设为-1，设为0,1,2,3,4....n（n指GPU的数目，即电脑有多少块GPU）
    # os.environ["CUDA_VISIBLE_DEVICES"] = "-1"              # -1代表使用 CPU 模式
    # os.environ["CUDA_VISIBLE_DEVICES"] = "0"               # 设置当前使用的GPU设备仅为第 0 块GPU, 设备名称为'/gpu:0'
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

    # 文件保存路径设置
    # store_file = 'Laplace3D'
    # store_file = 'pLaplace3D'
    store_file = 'Boltzmann3D'
    # store_file = 'Convection3D'
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

    # if the value of step_stop_flag is not 0, it will activate stop condition of step to kill program
    step_stop_flag = input('please input an  integer number to activate step-stop----0:no---!0:yes--:')
    R['activate_stop'] = int(step_stop_flag)
    # if the value of step_stop_flag is not 0, it will activate stop condition of step to kill program
    R['max_epoch'] = 200000
    if 0 != R['activate_stop']:
        epoch_stop = input('please input a stop epoch:')
        R['max_epoch'] = int(epoch_stop)

    # ---------------------------- Setup of multi-scale problem-------------------------------
    R['input_dim'] = 3  # 输入维数，即问题的维数(几元问题)
    R['output_dim'] = 1  # 输出维数

    if store_file == 'Laplace3D':
        R['PDE_type'] = 'general_Laplace'
        R['equa_name'] = 'PDE1'
        # R['equa_name'] = 'PDE2'
        # R['equa_name'] = 'PDE3'
        # R['equa_name'] = 'PDE4'
        # R['equa_name'] = 'PDE5'
        # R['equa_name'] = 'PDE6'
        # R['equa_name'] = 'PDE7'
    elif store_file == 'pLaplace3D':
        R['PDE_type'] = 'pLaplace'
        # R['equa_name'] = 'multi_scale3D_1'
        # R['equa_name'] = 'multi_scale3D_2'
        # R['equa_name'] = 'multi_scale3D_3'
        # R['equa_name'] = 'multi_scale3D_5'
        # R['equa_name'] = 'multi_scale3D_6'
        R['equa_name'] = 'multi_scale3D_7'
    elif store_file == 'Boltzmann3D':
        R['PDE_type'] = 'Possion_Boltzmann'
        # R['equa_name'] = 'Boltzmann1'
        # R['equa_name'] = 'Boltzmann2'
        # R['equa_name'] = 'Boltzmann3'
        R['equa_name'] = 'Boltzmann4'
        # R['equa_name'] = 'Boltzmann5'
        # R['equa_name'] = 'Boltzmann6'
        # R['equa_name'] = 'Boltzmann7'

    if R['PDE_type'] == 'general_laplace':
        R['mesh_number'] = 2
        R['epsilon'] = 0.1
        R['order2pLaplace_operator'] = 2
        R['batch_size2interior'] = 6000  # 内部训练数据的批大小
        R['batch_size2boundary'] = 1000
    elif R['PDE_type'] == 'pLaplace' or R['PDE_type'] == 'Possion_Boltzmann':
        R['mesh_number'] = 2
        R['epsilon'] = 0.1
        R['order2pLaplace_operator'] = 2
        R['batch_size2interior'] = 6000  # 内部训练数据的批大小
        R['batch_size2boundary'] = 1000

    # ---------------------------- Setup of DNN -------------------------------
    # R['loss_type'] = 'L2_loss'                 # PDE变分 1: grad U = grad Uc + grad Uf; 2: 变分形式是分开的
    R['loss_type'] = 'variational_loss'  # PDE变分 1: grad U = grad Uc + grad Uf; 2: 变分形式是分开的
    # R['loss_type'] = 'variational_loss2'       # PDE变分 1: grad U = grad Uc + grad Uf; 2: 变分形式是分开的

    R['opt2orthogonal'] = 0  # 0: integral L2-orthogonal   1: point-wise L2-orthogonal    2:energy
    # R['opt2orthogonal'] = 1                    # 0: integral L2-orthogonal   1: point-wise L2-orthogonal    2:energy
    # R['opt2orthogonal'] = 2                    # 0: integral L2-orthogonal   1: point-wise L2-orthogonal    2:energy

    R['hot_power'] = 1
    R['testData_model'] = 'loadData'

    R['regular_wb_model'] = 'L0'
    # R['regular_wb_model'] = 'L1'
    # R['regular_wb_model'] = 'L2'

    R['penalty2weight_biases'] = 0.000  # Regularization parameter for weights
    # R['penalty2weight_biases'] = 0.001        # Regularization parameter for weights
    # R['penalty2weight_biases'] = 0.0025       # Regularization parameter for weights

    R['activate_penalty2bd_increase'] = 1
    R['init_boundary_penalty'] = 100  # Regularization parameter for boundary conditions

    R['activate_powSolus_increase'] = 0
    if R['activate_powSolus_increase'] == 1:
        R['init_penalty2orthogonal'] = 5.0
    elif R['activate_powSolus_increase'] == 2:
        R['init_penalty2orthogonal'] = 10000.0
    else:
        R['init_penalty2orthogonal'] = 20.0
        # R['init_penalty2orthogonal'] = 25.0

    R['optimizer_name'] = 'Adam'  # 优化器
    R['learning_rate'] = 2e-4  # 学习率
    R['learning_rate_decay'] = 5e-5  # 学习率 decay
    R['train_model'] = 'training_union'  # 训练模式, 一个 loss 联结训练
    # R['train_model'] = 'training_group1'                # 训练模式, 多个 loss 组团训练
    # R['train_model'] = 'training_group2'
    # R['train_model'] = 'training_group3'
    # R['train_model'] = 'training_group4'

    # R['model2Normal'] = 'DNN'                           # 使用的网络模型
    # R['model2Normal'] = 'DNN_scale'
    # R['model2Normal'] = 'DNN_adapt_scale'
    R['model2Normal'] = 'Fourier_DNN'

    # R['model2Scale'] = 'DNN'                            # 使用的网络模型
    # R['model2Scale'] = 'DNN_scale'
    # R['model2Scale'] = 'DNN_adapt_scale'
    R['model2Scale'] = 'Fourier_DNN'

    # 单纯的 MscaleDNN 网络 FourierBase(250,400,400,200,200,150)  250+500*400+400*400+400*200+200*200+200*150+150 = 510400
    # 单纯的 MscaleDNN 网络 GeneralBase(500,400,400,200,200,150) 500+500*400+400*400+400*200+200*200+200*150+150 = 510650
    # FourierBase normal 和 FourierBase scale 网络的总参数数目:143220 + 365400 = 508870
    # GeneralBase normal 和 FourierBase scale 网络的总参数数目:143290 + 365650 = 508940
    if R['model2Normal'] == 'Fourier_DNN':
        R['hidden2normal'] = (70, 200, 200, 150, 150, 150)  # 70+140*200+200*200+200*150+150*150+150*150+150=143220
    else:
        R['hidden2normal'] = (140, 200, 200, 150, 150, 150)  # 140+140*200+200*200+200*150+150*150+150*150+150=143290
        # R['hidden2normal'] = (300, 200, 200, 100, 100, 50)
        # R['hidden2normal'] = (500, 400, 300, 200, 100)
        # R['hidden2normal'] = (500, 400, 300, 300, 200, 100)

    if R['model2Scale'] == 'Fourier_DNN':
        R['hidden2scale'] = (250, 300, 290, 200, 200, 150)  # 1*250+500*300+300*290+290*200+200*200+200*150+150 = 365400
    else:
        R['hidden2scale'] = (500, 300, 280, 200, 200, 150)  # 1*500+500*300+300*290+290*200+200*150+150*150+150 = 365650
        # R['hidden2scale'] = (300, 200, 200, 100, 100, 50)
        # R['hidden2scale'] = (500, 400, 300, 200, 100)
        # R['hidden2scale'] = (500, 400, 300, 300, 200, 100)

    # R['freq2Normal'] = np.concatenate(([1, 1, 1, 2, 2, 2, 3, 3, 3, 4, 4, 4, 5, 5], np.arange(5, 31)), axis=0)
    R['freq2Normal'] = np.arange(1, 41) * 0.5
    if R['model2Scale'] == 'Fourier_DNN':
        R['freq2Scale'] = np.arange(21, 121)
        # R['freq2Scale'] = np.arange(16, 101)
        # R['freq2Scale'] = np.arange(21, 101)
        # R['freq2Scale'] = np.arange(6, 105)
        # R['freq2Scale'] = np.arange(1, 101)
    else:
        R['freq2Scale'] = np.arange(21, 121)
        # R['freq2Scale'] = np.arange(21, 101)
        # R['freq2Scale'] = np.arange(6, 105)
        # R['freq2Scale'] = np.arange(1, 101)

    # 激活函数的选择
    # R['act_in2Normal'] = 'relu'
    R['act_in2Normal'] = 'tanh'

    # R['actName2Normal'] = 'relu'
    R['actName2Normal'] = 'tanh'
    # R['actName2Normal'] = 'srelu'
    # R['actName2Normal'] = 'sin'
    # R['actName2Normal'] = 's2relu'

    R['act_out2Normal'] = 'linear'

    # R['act_in2Scale'] = 'relu'
    R['act_in2Scale'] = 'tanh'

    # R['actName2Scale'] = 'relu'
    # R['actName2Scale']' = leaky_relu'
    # R['actName2Scale'] = 'srelu'
    R['actName2Scale'] = 's2relu'
    # R['actName2Scale'] = 'tanh'
    # R['actName2Scale'] = 'elu'
    # R['actName2Scale'] = 'phi'

    R['act_out2Scale'] = 'linear'

    if R['model2Normal'] == 'Fourier_DNN' and R['actName2Normal'] == 'tanh':
        R['sFourier2Normal'] = 1.0
    elif R['model2Normal'] == 'Fourier_DNN' and R['actName2Normal'] == 's2relu':
        R['sFourier2Normal'] = 0.5

    if R['model2Scale'] == 'Fourier_DNN' and R['actName2Scale'] == 'tanh':
        R['sFourier2Scale'] = 1.0
    elif R['model2Scale'] == 'Fourier_DNN' and R['actName2Scale'] == 's2relu':
        R['sFourier2Scale'] = 0.5

    if R['loss_type'] == 'variational_loss2':
        # R['init_penalty2orthogonal'] = 1.0
        # R['init_penalty2orthogonal'] = 10.0
        R['init_penalty2orthogonal'] = 20.0
        # R['init_penalty2orthogonal'] = 25.0
        # R['contrib2scale'] = 0.1
        R['contrib2scale'] = 0.05
        # R['contrib2scale'] = 0.01
        # R['contrib2scale'] = 0.005
    elif R['loss_type'] == 'variational_loss3':
        # R['init_penalty2orthogonal'] = 1.0
        # R['init_penalty2orthogonal'] = 10.0
        R['init_penalty2orthogonal'] = 20.0
        # R['init_penalty2orthogonal'] = 25.0
        # R['contrib2scale'] = 0.1
        R['contrib2scale'] = 0.05
        # R['contrib2scale'] = 0.01
        # R['contrib2scale'] = 0.005
    elif R['loss_type'] == 'variational_loss4':
        # R['init_penalty2orthogonal'] = 1.0
        # R['init_penalty2orthogonal'] = 10.0
        R['init_penalty2orthogonal'] = 20.0
        # R['init_penalty2orthogonal'] = 25.0
        # R['contrib2scale'] = 0.1
        R['contrib2scale'] = 0.05
        # R['contrib2scale'] = 0.01
        # R['contrib2scale'] = 0.005
    else:
        R['init_penalty2orthogonal'] = 20.0
        # R['init_penalty2orthogonal'] = 25.0
        # R['contrib2scale'] = 0.1
        R['contrib2scale'] = 0.05
        # R['contrib2scale'] = 0.025
        # R['contrib2scale'] = 0.01
        # R['contrib2scale'] = 0.005

    R['opt2loss_udotu'] = 'with_orthogonal'
    # R['opt2loss_udotu'] = 'without_orthogonal'

    # R['opt2loss_bd'] = 'unified_boundary'
    R['opt2loss_bd'] = 'individual_boundary'

    R['contrib_scale2orthogonal'] = 'with_contrib'
    # R['contrib_scale2orthogonal'] = 'without_contrib'

    R['contrib_scale2boundary'] = 'with_contrib'
    # R['contrib_scale2boundary'] = 'without_contrib'

    solve_Multiscale_PDE(R)

