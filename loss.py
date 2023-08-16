'''
quan xueliang
2021.3.1
MMD distance in domain adaptation
reference: Long, Mingsheng, et al. "Learning transferable features with
deep adaptation networks." International conference on machine learning. PMLR, 2015.
output: 基于 MMD kernel 返回一个损失函数项
'''

import torch
import torch.nn.functional as F
import torch.nn as nn
from torch.autograd import Function
import numpy as np
import contextlib

class GradientReverseFunction(Function):
    @staticmethod
    def forward(ctx, x, coeff=1.0):
        ctx.coeff = coeff
        output = x * 1.0
        return output
    @staticmethod
    def backward(ctx, grad_output):
        return grad_output.neg() * ctx.coeff, None


def gaussian_kernel(source, target, kernel_mul=2.0, kernel_num=0.5, fix_sigma=None):
    '''
    gaussian_kernel function
    :param source: source data, n*m
    :param target: target data, n*m
    :param kernel_mul: construct tuple, z_i = {xs_(2i-1), xs_2i, xt_(2i-1), xt_2i}
    :param kernel_num: number of gaussian kernels, multi kernel MMD
    :param fix_sigma:
    :return: kernel_val
    '''
    n_samples = int(source.size()[0]) + int(target.size()[0])
    total = torch.cat([source, target], dim=0)  # 将两个tensor拼接在一起
    # torch.unsqueeze()  矩阵维度扩充
    # torch.expand()  每个维度上扩充为指定的个数
    total0 = total.unsqueeze(0).expand(int(total.size(0)), int(total.size(0)), int(total.size(1)))
    total1 = total.unsqueeze(1).expand(int(total.size(0)), int(total.size(0)), int(total.size(1)))
    L2_distance = ((total0-total1)**2).sum(2)
    if fix_sigma:
        bandwidth = fix_sigma
    else:
        bandwidth = torch.sum(L2_distance.data) / (n_samples**2-n_samples)
    bandwidth /= kernel_mul**(kernel_mul//2)
    # 通过改变 bandwidth 来改变 gaussian kernel
    # compute the bandwidth of each gaussian kernel
    bandwidth_list = [bandwidth * (kernel_mul**i) for i in range(kernel_num)]
    # compute the value of each gaussian kernel
    kernel_val = [torch.exp(-L2_distance / bandwidth_temp) for bandwidth_temp in bandwidth_list]
    return sum(kernel_val)  # len(kernel_val)


def mmd_rbf_accelerate(source, target, kernel_mul=2.0, kernel_num=5, fix_sigma=None):
    '''
    unbiased estimate of MK-MMD loss
    :param source:
    :param target:
    :param kernel_mul:
    :param kernel_num:
    :param fix_sigma:
    :return: domain discrepancy MMD loss
    '''
    batch_size = int(source.size()[0])
    kernels = gaussian_kernel(source, target, kernel_mul=kernel_mul,
                              kernel_num=kernel_num, fix_sigma=fix_sigma)
    loss = 0
    for i in range(batch_size):
        s1, s2 = i, (i+1)%batch_size
        t1, t2 = s1+batch_size, s2+batch_size
        loss += (kernels[s1, s2] + kernels[t1, t2] - kernels[s1, t2] - kernels[s2, t1])
        # paper DAN, section 3.2, equation g(z)
    return loss / float(batch_size)


def mmd_rbf_noaccelerate(source, target, kernel_mul=2.0, kernel_num=5, fix_sigma=None):
    '''
    MK-MMD loss
    :param source:
    :param target:
    :param kernel_mul:
    :param kernel_num:
    :param fix_sigma:
    :return: domain discrepancy MMD loss
    '''
    batch_size = int(source.size()[0])
    kernels = gaussian_kernel(source, target, kernel_mul=kernel_mul,
                              kernel_num=kernel_num, fix_sigma=fix_sigma)
    loss = 0
    XX = kernels[:batch_size, :batch_size]
    YY = kernels[batch_size:, batch_size:]
    XY = kernels[:batch_size, batch_size:]
    YX = kernels[batch_size:, :batch_size]
    loss = torch.mean(XX + YY - XY - YX)
    return loss


def Entropy(predict):
    epsilon = 1e-5
    H = -predict * torch.log(predict + epsilon)
    H = H.sum(dim=1)
    return H



def entropy(out_t1, lamda=1.0):
    out_t1 = F.softmax(out_t1, dim=1)
    loss_ent = -lamda * torch.mean(torch.sum(out_t1 * (torch.log(out_t1 + 1e-5)), 1))
    return loss_ent



def _mix_rbf_kernel(X, Y, sigma_list):
    assert(X.size(0) == Y.size(0))
    m = X.size(0)

    Z = torch.cat((X, Y), 0)
    ZZT = torch.mm(Z, Z.t())
    diag_ZZT = torch.diag(ZZT).unsqueeze(1)
    Z_norm_sqr = diag_ZZT.expand_as(ZZT)
    exponent = Z_norm_sqr - 2 * ZZT + Z_norm_sqr.t()

    K = 0.0
    for sigma in sigma_list:
        gamma = 1.0 / (2 * sigma**2)
        K += torch.exp(-gamma * exponent)

    return K[:m, :m], K[:m, m:], K[m:, m:], len(sigma_list)


def mix_rbf_mmd2(X, Y, sigma_list=[1, 2, 5, 10], biased=True):
    K_XX, K_XY, K_YY, d = _mix_rbf_kernel(X, Y, sigma_list)
    # return _mmd2(K_XX, K_XY, K_YY, const_diagonal=d, biased=biased)
    return _mmd2(K_XX, K_XY, K_YY, const_diagonal=False, biased=biased)

def _mmd2(K_XX, K_XY, K_YY, const_diagonal=False, biased=False):
    m = K_XX.size(0)    # assume X, Y are same shape

    # Get the various sums of kernels that we'll use
    # Kts drop the diagonal, but we don't need to compute them explicitly
    if const_diagonal is not False:
        diag_X = diag_Y = const_diagonal
        sum_diag_X = sum_diag_Y = m * const_diagonal
    else:
        diag_X = torch.diag(K_XX)                       # (m,)
        diag_Y = torch.diag(K_YY)                       # (m,)
        sum_diag_X = torch.sum(diag_X)
        sum_diag_Y = torch.sum(diag_Y)

    Kt_XX_sums = K_XX.sum(dim=1) - diag_X             # \tilde{K}_XX * e = K_XX * e - diag_X
    Kt_YY_sums = K_YY.sum(dim=1) - diag_Y             # \tilde{K}_YY * e = K_YY * e - diag_Y
    K_XY_sums_0 = K_XY.sum(dim=0)                     # K_{XY}^T * e

    Kt_XX_sum = Kt_XX_sums.sum()                       # e^T * \tilde{K}_XX * e
    Kt_YY_sum = Kt_YY_sums.sum()                       # e^T * \tilde{K}_YY * e
    K_XY_sum = K_XY_sums_0.sum()                       # e^T * K_{XY} * e

    if biased:
        mmd2 = ((Kt_XX_sum + sum_diag_X) / (m * m)
            + (Kt_YY_sum + sum_diag_Y) / (m * m)
            - 2.0 * K_XY_sum / (m * m))
    else:
        mmd2 = (Kt_XX_sum / (m * (m - 1))
            + Kt_YY_sum / (m * (m - 1))
            - 2.0 * K_XY_sum / (m * m))

    return mmd2



class MultipleKernelMaximumMeanDiscrepancy(nn.Module):
    def __init__(self, linear=False):
        super(MultipleKernelMaximumMeanDiscrepancy, self).__init__()
        self.kernels = [GaussianKernel(alpha=2 ** k) for k in range(-3, 2)]
        self.index_matrix = None
        self.linear = linear

    def forward(self, z_s, z_t):
        features = torch.cat([z_s, z_t], dim=0)
        batch_size = int(z_s.size(0))
        self.index_matrix = self._update_index_matrix(batch_size, self.index_matrix, self.linear).to(z_s.device)
        kernel_matrix = sum([kernel(features) for kernel in self.kernels])  # Add up the matrix of each kernel
        # Add 2 / (n-1) to make up for the value on the diagonal
        # to ensure loss is positive in the non-linear version
        loss = (kernel_matrix * self.index_matrix).sum() + 2. / float(batch_size - 1)
        return loss

    def _update_index_matrix(self, batch_size, index_matrix=None, linear=True):
        if index_matrix is None or index_matrix.size(0) != batch_size * 2:
            index_matrix = torch.zeros(2 * batch_size, 2 * batch_size)
            if linear:
                for i in range(batch_size):
                    s1, s2 = i, (i + 1) % batch_size
                    t1, t2 = s1 + batch_size, s2 + batch_size
                    index_matrix[s1, s2] = 1. / float(batch_size)
                    index_matrix[t1, t2] = 1. / float(batch_size)
                    index_matrix[s1, t2] = -1. / float(batch_size)
                    index_matrix[s2, t1] = -1. / float(batch_size)
            else:
                for i in range(batch_size):
                    for j in range(batch_size):
                        if i != j:
                            index_matrix[i][j] = 1. / float(batch_size * (batch_size - 1))
                            index_matrix[i + batch_size][j + batch_size] = 1. / float(batch_size * (batch_size - 1))
                for i in range(batch_size):
                    for j in range(batch_size):
                        index_matrix[i][j + batch_size] = -1. / float(batch_size * batch_size)
                        index_matrix[i + batch_size][j] = -1. / float(batch_size * batch_size)
        return index_matrix


class GaussianKernel(nn.Module):
    def __init__(self, sigma=None, track_running_stats=True, alpha=1.):
        super(GaussianKernel, self).__init__()
        assert track_running_stats or sigma is not None
        self.sigma_square = torch.tensor(sigma * sigma) if sigma is not None else None
        self.track_running_stats = track_running_stats
        self.alpha = alpha

    def forward(self, X):
        l2_distance_square = ((X.unsqueeze(0) - X.unsqueeze(1)) ** 2).sum(2)

        if self.track_running_stats:
            self.sigma_square = self.alpha * torch.mean(l2_distance_square.detach())

        return torch.exp(-l2_distance_square / (2 * self.sigma_square))