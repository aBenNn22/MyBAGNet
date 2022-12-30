import torch.nn as nn
import torch
import torch.nn.functional as F
from models.BAGNet_utils import boundary_estimation, BAGNetAttentionPoolingLayer, PointwiseMLP, BAGLayer


class get_model(nn.Module):
    def __init__(self, num_classes=50, normal_channel=True, K=32, D=256):
        super(get_model, self).__init__()
        if normal_channel:
            channel = 6
        else:
            channel = 3
        self.num_classes = num_classes
        self.k = K
        self.pointwiseMLP = PointwiseMLP()
        self.attnPooling = BAGNetAttentionPoolingLayer()
        self.BAGLayer = BAGLayer()
        self.convs1 = nn.Conv1d(768, 512, 1)
        self.convs2 = nn.Conv1d(512, 256, 1)
        self.convs3 = nn.Conv1d(256, 128, 1)
        self.convs4 = nn.Conv1d(128, num_classes, 1)
        # self.bns1 = nn.BatchNorm1d(512)
        # self.bns2 = nn.BatchNorm1d(256)
        # self.bns3 = nn.BatchNorm1d(128)
        self.convs5 = nn.Conv1d(1024, 512, 1)
        self.convs6 = nn.Conv1d(512, 256, 1)
        self.convs7 = nn.Conv1d(256, 128, 1)
        self.convs8 = nn.Conv1d(128, num_classes, 1)
        # self.bns4 = nn.BatchNorm1d(512)
        # self.bns5 = nn.BatchNorm1d(256)
        # self.bns6 = nn.BatchNorm1d(128)

    def forward(self, xyz, cls_label):
        B, C, N = xyz.size()
        device = xyz.device
        output_scores = []
        # print(xyz.shape)
        for b in range(B):
            points = xyz[b:b+1, :, :] # (1, C, N)
            # print(points.shape)
            bound_points, non_bound_points, bound_map, non_bound_map = boundary_estimation(points, self.k)
            print(f'bound_points {bound_points.shape}, non_bound {non_bound_points.shape}')
            # (1, C, Nb)   (1, C, Nb)      list(Nb)    list(Nn)
            if bound_points.shape[2] > 0:
                bound_feature = self.BAGLayer(bound_points, points) # (1, Nb, D)
            non_bound_feature = self.pointwiseMLP(non_bound_points) # (1, Nn, 512)
            global_feature = self.attnPooling(points) # (1, N, 512)
            Nb = bound_points.shape[1]
            bound_global = global_feature[:, bound_map, :] # (1, Nb, 512)
            non_bound_global = global_feature[:, non_bound_map, :] # (1, Nn, 512)

            # print(f'non_bound_feature {non_bound_feature.shape}, non_bound_global {non_bound_global.shape}')
            # print(f'bound_feature {bound_feature.shape}, bound_global {bound_global.shape}')
            if bound_points.shape[2] > 0:
                bound_cat = torch.cat((bound_feature, bound_global), dim=-1).transpose(2,1)
            non_bound_cat = torch.cat((non_bound_feature, non_bound_global), dim=-1).transpose(2,1)

            # bound_cat = F.relu(self.bns1(self.convs1(bound_cat)))
            # bound_cat = F.relu(self.bns2(self.convs2(bound_cat)))
            # bound_cat = F.relu(self.bns3(self.convs3(bound_cat)))
            if bound_points.shape[2] > 0:
                bound_cat = F.relu(self.convs1(bound_cat))
                bound_cat = F.relu(self.convs2(bound_cat))
                bound_cat = F.relu(self.convs3(bound_cat))
                bound_cat = self.convs4(bound_cat)
                bound_cat = bound_cat.transpose(2,1)
            # non_bound_cat = F.relu(self.bns4(self.convs5(non_bound_cat)))
            # non_bound_cat = F.relu(self.bns5(self.convs6(non_bound_cat)))
            # non_bound_cat = F.relu(self.bns6(self.convs7(non_bound_cat)))
            non_bound_cat = F.relu(self.convs5(non_bound_cat))
            non_bound_cat = F.relu(self.convs6(non_bound_cat))
            non_bound_cat = F.relu(self.convs7(non_bound_cat))
            non_bound_cat = self.convs8(non_bound_cat)
            non_bound_cat = non_bound_cat.transpose(2,1)
            output_score = torch.zeros((1, N, self.num_classes), dtype=float).to(device)
            for i, idx in enumerate(bound_map):
                # print(f'output_score[:, idx, :] {output_score[:, idx, :].shape}')
                output_score[:, idx, :] = bound_cat[:, i, :]
            for i, idx in enumerate(non_bound_map):
                output_score[:, idx, :] = non_bound_cat[:, i, :]
            output_score = output_score.transpose(2, 1).contiguous()
            output_score = F.log_softmax(output_score.view(-1, self.num_classes), dim=-1)
            output_score = output_score.view(1, N, self.num_classes) # [1, N, 50]
            output_scores.append(output_score)
        res_score = torch.cat(output_scores, dim=0)
        return res_score


class get_loss(nn.Module):
    def __init__(self):
        super(get_loss, self).__init__()

    def forward(self, pred, target):
        total_loss = F.nll_loss(pred, target)

        return total_loss