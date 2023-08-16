import os
import torch
import torch.nn as nn
import torch.nn.functional as F

import numpy as np
import random
import matplotlib.pyplot as plt
import argparse
import time


from Attention import TransformerBase2
from model import Classifier

from data_loader import get_dataset
from torch.utils.data import DataLoader

from loss import adentropy, entropy

from sklearn.metrics import confusion_matrix, accuracy_score, f1_score, recall_score, cohen_kappa_score

import warnings

warnings.filterwarnings('ignore')

from torch.autograd import Function

net_psd = {'raw': False,
           'psd': True}


class GradientReverseFunction(Function):
    @staticmethod
    def forward(ctx, x, coeff):
        ctx.coeff = coeff
        output = x * 1.0
        return output
    @staticmethod
    def backward(ctx, grad_output):
        return grad_output.neg() * ctx.coeff, None


class Solver(object):
    def __init__(self, args) -> None:
        super().__init__()
        self.epochs = args.epoch
        self.early_stop = args.early_stop
        self.learning_rate = args.lr
        self.batch_size = args.batch_size
        self.gamma = args.gamma
        self.need_weight = args.weight
        self.exp_info = args.info
        self.seed = args.seed

        self.cur = 0 # current exp time
        self.repeate = args.repeate
        self.args = args


        self.dataset = get_dataset(ch=[0], source_size=39, class_num=5, semi_ratio=0.05, psd=args.psd, sval=args.sval)

        self.source_loader = DataLoader(self.dataset['source_dataset'], batch_size=args.batch_size, shuffle=True, num_workers=2, drop_last=False)
        self.tr_unl_loader = DataLoader(self.dataset['target_unl_set'], batch_size=args.batch_size, shuffle=True, num_workers=2, drop_last=False)
        self.tr_lab_loader = DataLoader(self.dataset['target_lab_set'], batch_size=args.batch_size, shuffle=True, num_workers=2, drop_last=False)

        self.valida_loader = DataLoader(self.dataset['validation_set'], batch_size=args.batch_size, shuffle=False, num_workers=0, drop_last=False)
        self.target_loader = DataLoader(self.dataset['target_unl_set'], batch_size=args.batch_size, shuffle=False, num_workers=0, drop_last=False)

        kwargs = {'embed_dim': 128, 'hidden_dim': 128*4, 'num_heads': 8, 'layer_nums': 3, 'psd': args.psd}
        self.netF = TransformerBase2(**kwargs).cuda()
        self.netC = Classifier(self.netF.out_dim, 5).cuda()
        self.optimizer = torch.optim.Adam(params=list(self.netF.parameters()) + list(self.netC.parameters()), lr=self.learning_rate, weight_decay=0.001)

        if self.need_weight:
            self.CEw = torch.from_numpy(self.dataset['weight'][0]).float().cuda()
        else:
            self.CEw = None

        
        self.class_criterion = nn.CrossEntropyLoss(weight=self.CEw)

        self.train_dict = {'train_acc':[], 'train_loss':[], 'trans_loss':[], 'valid_acc':[],'valid_mf1':[], 'valid_loss':[]}

        self.save_path = '/data1/zst/ModelSave/DA/' + self.exp_info
        self.model_save_path = self.save_path + '/best_model'
        self.visul_save_path = './exp/' + self.exp_info
        self.resul_save_path = './exp/' + self.exp_info

        self.check_dir()
        # self.show()


    def show(self):
        print("====================================model summary=============================================")
        # print(summary(self.netF, (1, 3000), batch_size=1))
        print("====================================model summary=============================================")


    def reset(self):
        del self.netF
        del self.netC
        del self.optimizer
        
        kwargs = {'embed_dim': 128, 'hidden_dim': 128*4, 'num_heads': 8, 'layer_nums': 3, 'psd': args.psd}
        self.netF = TransformerBase2(**kwargs).cuda()
        self.netC = Classifier(self.netF.out_dim, 5).cuda()
        self.optimizer = torch.optim.Adam(params=list(self.netF.parameters()) + list(self.netC.parameters()), lr=self.learning_rate, weight_decay=0.001)

        self.train_dict = {'train_acc':[], 'ce_loss':[], 'dc_loss':[], 'valid_acc':[],'valid_mf1':[], 'valid_loss':[]}
        # self.show()

    def train(self):
        self.set_seed()
        self.cur += 1
        
        self.reset()

        self.netF.train()
        self.netC.train()

        best_acc = 0.0
        best_mf1 = 0.0

        source_len = len(self.source_loader)
        target_len = len(self.target_loader)
        max_iter = max([source_len, target_len])
        cur_iter = 0

        for epoch in range(self.epochs):

            ce_loss_epoch_s = 0.0
            ce_loss_epoch_t = 0.0

            source_cor = 0
            source_num = 0
            target_cor = 0
            target_num = 0

            i = 0

            source_iter = self.source_loader.__iter__()
            tr_unl_iter = self.tr_unl_loader.__iter__()
            tr_lab_iter = self.tr_lab_loader.__iter__()

            while i < max_iter:
                try:
                    tr_unl_x, _____________ = tr_unl_iter.__next__()
                except:
                    tr_unl_iter = self.tr_unl_loader.__iter__()
                    tr_unl_x, _____________ = tr_unl_iter.__next__()
                
                try:
                    tr_lab_x, tr_lab_y = tr_lab_iter.__next__()
                except:
                    tr_lab_iter = self.tr_lab_loader.__iter__()
                    tr_lab_x, tr_lab_y = tr_lab_iter.__next__()


                try:
                    source_x, source_y = source_iter.__next__()
                except:
                    source_iter = self.source_loader.__iter__()
                    source_x, source_y = source_iter.__next__()
                i += 1

                source_x, source_y, tr_unl_x = source_x.cuda(), source_y.cuda(), tr_unl_x.cuda()

                tr_lab_x, tr_lab_y = tr_lab_x.cuda(), tr_lab_y.cuda()

                lab_data = torch.cat([source_x, tr_lab_x], dim=0)

                feat_lab = self.netF(lab_data)

                outputs = self.netC(feat_lab[:source_x.shape[0], 0, :])
                outputt = self.netC(feat_lab[source_x.shape[0]:, -1, :])

                ce_loss_s = self.class_criterion(outputs, source_y)
                ce_loss_t = self.class_criterion(outputt, tr_lab_y)

                ce_loss_epoch_s += ce_loss_s.data.cpu().numpy()
                ce_loss_epoch_t += ce_loss_t.data.cpu().numpy()

                ce_loss = ce_loss_s + ce_loss_t

                self.optimizer.zero_grad()
                ce_loss.backward()
                self.optimizer.step()

                # for train_acc
                source_pre = torch.max(outputs, 1)[1]
                target_pre = torch.max(outputt, 1)[1]

                source_cor += torch.sum(source_pre==source_y)
                target_cor += torch.sum(target_pre==tr_lab_y)

                source_num += source_x.shape[0]
                target_num += tr_lab_x.shape[0]

    
            ce_loss_epoch_s /= i
            ce_loss_epoch_t /= i

            source_acc = source_cor / source_num
            target_acc = target_cor / target_num

            valid_acc, valid_mf1, valid_loss = self.eval(ensemble=args.ens)
            test_acc, test_mf1 = self.test(ensemble=args.ens)

            timeinfo = time.strftime('%Y.%m.%d-%H:%M:%S',time.localtime(time.time()))

            print("{} epoch={} source_acc={:.5f} target_acc={:.5f} ce_loss_s={:.5f} ce_loss_t={:.5f} valid_acc={:.5f} valid_loss={:.5f} F1={:.5f} t_acc={:.5f} t_mf1={:.5f}"
                .format(timeinfo, epoch, source_acc, target_acc, ce_loss_epoch_s, ce_loss_epoch_t, valid_acc, valid_loss, valid_mf1, test_acc, test_mf1))


            if valid_mf1 > best_mf1:
                best_mf1 = valid_mf1
                best_acc = valid_acc
                final_test_acc = test_acc
                final_test_mf1 = test_mf1
                torch.save(self.netF, os.path.join(self.model_save_path, 'netF'+str(self.cur)+'.pt'))
                torch.save(self.netC, os.path.join(self.model_save_path, 'netC'+str(self.cur)+'.pt'))
                last_best_epoch = epoch

            # early_stopping
            if epoch - last_best_epoch >= self.early_stop:
                print('early_stpping! curr epoch: {} last_best_epoch: {}'.format(epoch, last_best_epoch))
                print('best mf1={} acc={}'.format(best_mf1, best_acc))
                break

        # self.plot_save()
        print('test acc={} test mf1={}'.format(final_test_acc, final_test_mf1))
        return [final_test_acc, final_test_mf1]

    def eval(self, ensemble=1):
        self.netF.eval()
        self.netC.eval()
        predict = []
        groundtruth = []
        loss_fuc = nn.CrossEntropyLoss()
        loss = 0.0
        with torch.no_grad():
            for (batch_id, data) in enumerate(self.valida_loader):
                x_data = data[0].cuda()
                y_data = data[1]

                feat = self.netF(x_data)
                outs = self.netC(feat[:, 0, :])
                outt = self.netC(feat[:, -1, :])
                if ensemble==0:
                    outputs = outs
                elif ensemble==1:
                    # outputs = (outs + outt) / 2.0
                    outputs = self.ensemdyn(outs, outt)
                elif ensemble==2:
                    outputs = outt

                loss += loss_fuc(outputs, y_data.cuda()).data.cpu().numpy()

                result = torch.max(outputs, 1)[1].data.cpu().numpy()
                result = np.reshape(result,newshape=-1)
                y_data = np.reshape(y_data,newshape=-1)

                predict.append(result)
                groundtruth.append(y_data)
        
        loss /= (batch_id+1)

        predict = np.concatenate(predict,axis=0)
        groundtruth = np.concatenate(groundtruth,axis=0)

        acc = accuracy_score(groundtruth, predict)
        MF1 = f1_score(groundtruth, predict, average='macro')
        # k = cohen_kappa_score(groundtruth, predict)
        # cm = confusion_matrix(ground_truth, predict, labels=class_labels)
        self.netF.train()
        self.netC.train()
        return acc, MF1, loss

    def test(self, ensemble=1):
        self.netF.eval()
        self.netC.eval()
        predict = []
        groundtruth = []
        with torch.no_grad():
            for (batch_id, data) in enumerate(self.target_loader):
                x_data = data[0].cuda()
                y_data = data[1]

                feat = self.netF(x_data)
                outs = self.netC(feat[:, 0, :])
                outt = self.netC(feat[:, -1, :])
                if ensemble==0:
                    outputs = outs
                elif ensemble==1:
                    # outputs = (outs + outt) / 2.0
                    outputs = self.ensemdyn(outs, outt)
                elif ensemble==2:
                    outputs = outt

                result = torch.max(outputs, 1)[1].data.cpu().numpy()
                result = np.reshape(result, newshape=-1)
                y_data = np.reshape(y_data, newshape=-1)

                predict.append(result)
                groundtruth.append(y_data)


        predict = np.concatenate(predict, axis=0)
        groundtruth = np.concatenate(groundtruth, axis=0)

        acc = accuracy_score(groundtruth, predict)
        MF1 = f1_score(groundtruth, predict, average='macro')
        # k = cohen_kappa_score(groundtruth, predict)
        # cm = confusion_matrix(ground_truth, predict, labels=class_labels)
        self.netF.train()
        self.netC.train()
        return acc, MF1
    


    def ensemble(self, outs, outt, method=0):
        out_s = F.softmax(outs, dim=1)
        out_t = F.softmax(outt, dim=1)
        ent_s = torch.sum(-out_s * (torch.log(out_s + 1e-5)), dim=1)
        ent_t = torch.sum(-out_t * (torch.log(out_t + 1e-5)), dim=1)
        ent_ms = ent_s.mean()
        ent_mt = ent_t.mean()

        threshold = ent_ms * args.a
        index = torch.where(ent_s.squeeze() > threshold)[0]
        if index.shape[0]>0:
            outs[index] = outt[index]
        return outs
    

    def ensemdyn(self, outs, outt, method=0):
        out_s = F.softmax(outs, dim=1)
        out_t = F.softmax(outt, dim=1)
        ent_s = torch.sum(-out_s * (torch.log(out_s + 1e-5)), dim=1)
        ent_t = torch.sum(-out_t * (torch.log(out_t + 1e-5)), dim=1)

        ent = torch.stack([ent_s, ent_t], dim=1)

        # weight version 1
        # ent = F.softmax(ent, dim=1)
        # weight = 1.0 - ent

        # weight version 2
        weight = 1.0 + torch.exp(-ent)
        # print(weight.shape)
        weight = weight / torch.sum(weight, dim=1, keepdim=True) * 2


        out = torch.stack([out_s, out_t], dim=1)
        out = weight.unsqueeze(-1) * out
        fus = out.sum(dim=1)
        return fus


    def set_seed(self):
        seed = self.seed + self.cur * 100
        random.seed(seed)
        torch.manual_seed(seed)
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        np.random.seed(seed)
        torch.backends.cudnn.deterministic = True
        print('--------------------------------------set seed {}!---------------------------'.format(seed))


    def plot_save(self):
        plt.subplot(2, 3, 1)
        plt.plot(self.train_dict['ce_loss'], label='source ce loss - ad loss')
        plt.plot(self.train_dict['valid_loss'], label='target ce loss')
        plt.legend()
        plt.subplot(2, 3, 2)
        plt.plot(self.train_dict['dc_loss'], label='ad loss')
        plt.legend()

        # plt.subplot(2, 3, 3)
        # plt.plot(np.array(self.train_dict['train_loss'])+np.array(self.train_dict['trans_loss']), label='total loss')
        # plt.legend()

        plt.subplot(2, 3, 4)
        plt.plot(self.train_dict['train_acc'], label='source acc')
        plt.legend()

        plt.subplot(2, 3, 5)
        plt.plot(self.train_dict['valid_acc'], label='target acc')
        plt.legend()

        plt.subplot(2, 3, 6)
        plt.plot(self.train_dict['valid_mf1'], label='target MF1')
        plt.legend()

        plt.savefig(self.visul_save_path + '/' + str(self.cur) + '.png')
    

    def repeate_exp(self):
        acc, mf1 = [], []
        for t in range(self.repeate):
            a, m = self.train()
            acc.append(a), mf1.append(m)
        acc = np.array(acc)
        mf1 = np.array(mf1)
        log_str = 'avg acc: {}/{} avg mf1: {}/{} \n'.format(acc.mean(), acc.std(), mf1.mean(), mf1.std())
        print(log_str)

        with open(os.path.join(self.resul_save_path, 'result.txt'), 'a') as f:
            f.write('args: {}\n'.format(self.args))
            f.write(log_str)
            f.write('acc: {} mf1: {} \n'.format(acc, mf1))


    def check_dir(self):
        if not os.path.exists(self.save_path):
            os.mkdir(self.save_path)
        if not os.path.exists(self.model_save_path):
            os.mkdir(self.model_save_path)
        if not os.path.exists(self.visul_save_path):
            os.mkdir(self.visul_save_path)
        if not os.path.exists(self.resul_save_path):
            os.mkdir(self.resul_save_path)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Args for training parameters")
   
    parser.add_argument('--gpu_id', type=str, default='0', help='gpu_id')
    parser.add_argument('--epoch', type=int, default=100, help='The number of epochs to run')
    parser.add_argument('--early_stop', type=int, default=10, help='The number of epochs to run')
    parser.add_argument('--net', type=str, default='raw', help='network')
    parser.add_argument('--lr', type=float, default=0.0001)
    parser.add_argument('--batch_size', type=int, default=64, help='The size of batch')
    parser.add_argument('--seed', type=int, default=15789, help='seed')
    parser.add_argument('--repeate', type=int, default=3, help='repeate exp')
    parser.add_argument('--weight', type=int, default=1, help='need weight?')
    parser.add_argument('--sval', type=int, default=1, help='source valid')
    parser.add_argument('--info', type=str, default='MME_tiny_stval', help='Experiment info')
    parser.add_argument('--gamma', type=float, default=0.1, help='Experiment info')
    parser.add_argument('--num_class', type=int, default=5, help='class num')
    parser.add_argument('--entropy', type=int, default=0, help='entropy')
    parser.add_argument('--T', type=float, default=0.05, help='Temp')
    parser.add_argument('--ens', type=int, default=1, help='ensemble')
    parser.add_argument('--a', type=float, default=1.5, help='ensemble')

    args = parser.parse_args()
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu_id

    args.psd = net_psd[args.net]

    print('args: ', args)

    SC = Solver(args)

    SC.repeate_exp()