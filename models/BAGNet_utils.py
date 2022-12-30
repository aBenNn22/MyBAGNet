import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
from time import time
import numpy as np
import math
from pclpy import pcl

def timeit(tag, t):
    print("{}: {}s".format(tag, time() - t))
    return time()

def pc_normalize(pc):
    l = pc.shape[0]
    centroid = np.mean(pc, axis=0)
    pc = pc - centroid
    m = np.max(np.sqrt(np.sum(pc**2, axis=1)))
    pc = pc / m
    return pc

def square_distance(src, dst):
    """
    Calculate Euclid distance between each two points.

    src^T * dst = xn * xm + yn * ym + zn * zm；
    sum(src^2, dim=-1) = xn*xn + yn*yn + zn*zn;
    sum(dst^2, dim=-1) = xm*xm + ym*ym + zm*zm;
    dist = (xn - xm)^2 + (yn - ym)^2 + (zn - zm)^2
         = sum(src**2,dim=-1)+sum(dst**2,dim=-1)-2*src^T*dst

    Input:
        src: source points, [B, N, C]
        dst: target points, [B, M, C]
    Output:
        dist: per-point square distance, [B, N, M]
    """
    B, N, _ = src.shape
    _, M, _ = dst.shape
    dist = -2 * torch.matmul(src, dst.permute(0, 2, 1))
    dist += torch.sum(src ** 2, -1).view(B, N, 1)
    dist += torch.sum(dst ** 2, -1).view(B, 1, M)
    return dist


def index_points(points, idx):
    """

    Input:
        points: input points data, [B, N, C]
        idx: sample index data, [B, S]
    Return:
        new_points:, indexed points data, [B, S, C]
    """
    device = points.device
    B = points.shape[0]
    view_shape = list(idx.shape)
    view_shape[1:] = [1] * (len(view_shape) - 1)
    repeat_shape = list(idx.shape)
    repeat_shape[0] = 1
    batch_indices = torch.arange(B, dtype=torch.long).to(device).view(view_shape).repeat(repeat_shape)
    new_points = points[batch_indices, idx, :]
    return new_points



def query_ball_point(radius, nsample, xyzn, new_xyzn):
    """
    Input:
        radius: local region radius
        nsample: max sample number in local region
        xyzn: all points, [B, N, C]
        new_xyzn: query points, [B, S, C]
    Return:
        group_idx: grouped points index, [B, S, nsample]
    """
    device = xyzn.device
    B, N, C = xyzn.shape
    _, S, _ = new_xyzn.shape
    group_idx = torch.arange(N, dtype=torch.long).to(device).view(1, 1, N).repeat([B, S, 1])
    # print(group_idx.shape)
    # print(f'xyz {xyzn.shape}, new_xyzn {new_xyzn.shape}')
    sqrdists = square_distance(new_xyzn, xyzn)
    group_idx[sqrdists > radius ** 2] = N
    group_idx = group_idx.sort(dim=-1)[0][:, :, :nsample]
    # print(group_idx.shape)
    group_first = group_idx[:, :, 0].view(B, S, 1).repeat([1, 1, nsample])
    # print(group_first.shape)
    mask = group_idx == N
    # print(f'group_idx {group_idx.shape}, group_first {group_first.shape}')
    group_idx[mask] = group_first[mask]
    return group_idx


def boundary_estimation(xyzn, nsample, radius=10000):
    """
    Input:
        radius: local region radius
        nsample: max sample number in local region
        xyzn: all points, [B, C, N], B = 1 cause the number of boundary point of each batch is different
    Return:
        bound_points: all boundary points, [B, 6, Nb], B = 1
        non_bound_points: all non-boundary points, [B, 6, Nn], B = 1
        bound_map: idx of boundary points => idx of all points
        non_bound_map: idx of non-boundary points => idx of all points
    """
    # 加载点云数据
    xyzn = xyzn.transpose(1,2)
    points = xyzn[:,:,0:3].squeeze(0)
    # print(points.shape)
    cloud = pcl.PointCloud.PointXYZ.from_array(points.numpy())
    # 法向量估计
    n = pcl.features.NormalEstimationOMP.PointXYZ_Normal()
    tree = pcl.search.KdTree.PointXYZ()
    n.setInputCloud(cloud)
    n.setSearchMethod(tree)    # 设置近邻搜索的方式
    # n.setNumberOfThreads(6)
    n.setKSearch(nsample)           # 点云法向计算时，需要所搜的近邻点大小
    normals = pcl.PointCloud.Normal()
    n.compute(normals)         # 开始进行法向计
    # 边界提取
    boundEst = pcl.features.BoundaryEstimation.PointXYZ_Normal_Boundary()
    boundEst.setInputCloud(cloud)          # 输入点云
    boundEst.setInputNormals(normals)      # 输入法线
    boundEst.setRadiusSearch(radius)         # 半径阈值
    boundEst.setAngleThreshold(np.pi / 2)  # 夹角阈值
    boundEst.setSearchMethod(tree)         # 设置近邻搜索方式
    boundaries = pcl.PointCloud.Boundary()
    boundEst.compute(boundaries)           # 获取边界索引
    cloud_boundary = pcl.PointCloud.PointXYZ()
    cloud_non_bound = pcl.PointCloud.PointXYZ()
    normal_boundary = pcl.PointCloud.Normal()
    normal_non_bound = pcl.PointCloud.Normal()
    bound_map = []
    non_bound_map = []
    # 获取边界点
    for i in range(cloud.size()):
        if boundaries.boundary_point[i] > 0:
            cloud_boundary.push_back(cloud.points[i])
            normal_boundary.push_back(normals.points[i])
            bound_map.append(i)
        else:
            cloud_non_bound.push_back(cloud.points[i])
            normal_non_bound.push_back(normals.points[i])
            non_bound_map.append(i)
    # print(torch.cat((torch.from_numpy(cloud_boundary.xyz).unsqueeze(0).transpose(1,2), torch.from_numpy(normal_boundary.normals).unsqueeze(0).transpose(1,2)), dim=1).shape)
    return torch.cat((torch.from_numpy(cloud_boundary.xyz).unsqueeze(0).transpose(1,2), torch.from_numpy(normal_boundary.normals).unsqueeze(0).transpose(1,2)), dim=1), torch.cat((torch.from_numpy(cloud_non_bound.xyz).unsqueeze(0).transpose(1,2), torch.from_numpy(normal_non_bound.normals).unsqueeze(0).transpose(1,2)), dim=1), bound_map, non_bound_map


    # device = xyzn.device
    # B, C, N = xyzn.shape
    # xyzn = xyzn.transpose(2, 1)
    # #求得每一个点的邻域点
    # nei_idx = query_ball_point(radius, nsample, xyzn, xyzn) # (B, N, nsample) B = 1
    # nei_xyzn = index_points(xyzn, nei_idx) # (B, N, nsample, C)
    
    # point_angle = torch.zeros(N, nsample, dtype=torch.float).to(device)
    # # for b in range(xyzn.shape[0]):
    # bound_list = []
    # bound_map = []
    # non_bound_list = []
    # non_bound_map = []
    # points_normal = xyzn[0, :, 3:6].reshape((N, 3))
    # neis_normal = nei_xyzn[0, :, :, 3:6].reshape((N, nsample, 3))
    # for n in range(N):
    #     point_normal = points_normal[n].flatten()
    #     l_point = torch.sqrt(point_normal.dot(point_normal))
    #     nei_normal = neis_normal[n, :, :].reshape((nsample,3))
    #     bound = False
    #     for s in range(nsample):
    #         neiN = nei_normal[s].flatten()
    #         l_nei = torch.sqrt(neiN.dot(neiN))
    #         dian = point_normal.dot(neiN)
    #         cos_ = dian / (l_point * l_nei)
    #         angle_s = torch.arccos(cos_)
    #         point_angle[n, s] = angle_s
    #         if s > 0 and point_angle[n, s] - point_angle[n, s-1] > math.pi / 2:
    #             new_bound = xyzn[0, n, :].reshape((1, 1, 6))
    #             bound_map.append(n)
    #             bound_list.append(new_bound)
    #             bound = True
    #             break
    #     if not bound:
    #         new_non_bound = xyzn[0, n, :].reshape((1, 1, 6))
    #         non_bound_map.append(n)
    #         non_bound_list.append(new_non_bound)
    # bound_points = torch.cat(bound_list, dim=1)
    # bound_points = bound_points.transpose(1, 2)
    # non_bound_points = torch.cat(non_bound_list, dim=1)
    # non_bound_points = non_bound_points.transpose(1, 2)
    # return bound_points, non_bound_points, bound_map, non_bound_map


class BAGNetAttentionPoolingLayer(nn.Module):
    def __init__(self, radius=10000, K=32):
        super(BAGNetAttentionPoolingLayer, self).__init__()
        self.radius = radius
        self.K = K
        self.conv1 = nn.Conv2d(6, 16, 1)
        self.conv2 = nn.Conv2d(16, 64, 1)
        self.conv3 = nn.Conv2d(64, 128, 1)
        self.conv4 = nn.Conv2d(128, 512, 1)
        # self.bn1 = nn.BatchNorm2d(16)
        # self.bn2 = nn.BatchNorm2d(64)
        # self.bn3 = nn.BatchNorm2d(128)
        # self.bn4 = nn.BatchNorm2d(512)

    def forward(self, points):
        B, C, N = points.size()
        points = points.transpose(2, 1)
        k_points_idx = query_ball_point(self.radius, self.K, points, points)
        k_points = index_points(points, k_points_idx)
        k_points = k_points.permute(0, 3, 1, 2)
        # k_points = F.relu(self.bn1(self.conv1(k_points)))
        # k_points = F.relu(self.bn2(self.conv2(k_points)))
        # k_points = F.relu(self.bn3(self.conv3(k_points)))
        # k_points = F.relu(self.bn4(self.conv4(k_points)))
        k_points = F.relu(self.conv1(k_points))
        k_points = F.relu(self.conv2(k_points))
        k_points = F.relu(self.conv3(k_points))
        k_points = F.relu(self.conv4(k_points))
        k_points = k_points.permute(0, 2, 3, 1)
        k_points = torch.max(k_points, dim=2)[0]
        return k_points


class STN3d(nn.Module):
    def __init__(self, channel):
        super(STN3d, self).__init__()
        self.conv1 = torch.nn.Conv1d(channel, 64, 1)
        self.conv2 = torch.nn.Conv1d(64, 128, 1)
        self.conv3 = torch.nn.Conv1d(128, 1024, 1)
        self.fc1 = nn.Linear(1024, 512)
        self.fc2 = nn.Linear(512, 256)
        self.fc3 = nn.Linear(256, 9)
        self.relu = nn.ReLU()

        # self.bn1 = nn.BatchNorm1d(64)
        # self.bn2 = nn.BatchNorm1d(128)
        # self.bn3 = nn.BatchNorm1d(1024)
        # self.bn4 = nn.BatchNorm1d(512)
        # self.bn5 = nn.BatchNorm1d(256)

    def forward(self, x):
        batchsize = x.size()[0]
        # x = F.relu(self.bn1(self.conv1(x)))
        # x = F.relu(self.bn2(self.conv2(x)))
        # x = F.relu(self.bn3(self.conv3(x)))
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))
        x = torch.max(x, 2, keepdim=True)[0]
        x = x.view(-1, 1024)

        # x = F.relu(self.bn4(self.fc1(x)))
        # x = F.relu(self.bn5(self.fc2(x)))
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)

        iden = Variable(torch.from_numpy(np.array([1, 0, 0, 0, 1, 0, 0, 0, 1]).astype(np.float32))).view(1, 9).repeat(
            batchsize, 1)
        if x.is_cuda:
            iden = iden.cuda()
        x = x + iden
        x = x.view(-1, 3, 3)
        return x


class STNkd(nn.Module):
    def __init__(self, k=64):
        super(STNkd, self).__init__()
        self.conv1 = torch.nn.Conv1d(k, 64, 1)
        self.conv2 = torch.nn.Conv1d(64, 128, 1)
        self.conv3 = torch.nn.Conv1d(128, 1024, 1)
        self.fc1 = nn.Linear(1024, 512)
        self.fc2 = nn.Linear(512, 256)
        self.fc3 = nn.Linear(256, k * k)
        self.relu = nn.ReLU()

        # self.bn1 = nn.BatchNorm1d(64)
        # self.bn2 = nn.BatchNorm1d(128)
        # self.bn3 = nn.BatchNorm1d(1024)
        # self.bn4 = nn.BatchNorm1d(512)
        # self.bn5 = nn.BatchNorm1d(256)

        self.k = k

    def forward(self, x):
        batchsize = x.size()[0]
        # x = F.relu(self.bn1(self.conv1(x)))
        # x = F.relu(self.bn2(self.conv2(x)))
        # x = F.relu(self.bn3(self.conv3(x)))
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))
        x = torch.max(x, 2, keepdim=True)[0]
        x = x.view(-1, 1024)

        # x = F.relu(self.bn4(self.fc1(x)))
        # x = F.relu(self.bn5(self.fc2(x)))
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)

        iden = Variable(torch.from_numpy(np.eye(self.k).flatten().astype(np.float32))).view(1, self.k * self.k).repeat(
            batchsize, 1)
        if x.is_cuda:
            iden = iden.cuda()
        x = x + iden
        x = x.view(-1, self.k, self.k)
        return x


class PointwiseMLP(nn.Module):
    def __init__(self, global_feat=True, feature_transform=False, channel=6):
        super(PointwiseMLP, self).__init__()
        self.stn = STN3d(channel)
        self.conv1 = torch.nn.Conv1d(channel, 64, 1)
        self.conv2 = torch.nn.Conv1d(64, 128, 1)
        self.conv3 = torch.nn.Conv1d(128, 512, 1)
        # self.bn1 = nn.BatchNorm1d(64)
        # self.bn2 = nn.BatchNorm1d(128)
        # self.bn3 = nn.BatchNorm1d(512)
        self.global_feat = global_feat
        self.feature_transform = feature_transform
        if self.feature_transform:
            self.fstn = STNkd(k=64)
    
    def forward(self, x):
        B, D, N = x.size()
        # x = x.repeat(16, 1, 1)
        # print(f'x.shape {x.shape}')
        trans = self.stn(x)
        x = x.transpose(2, 1)
        if D > 3:
            feature = x[:, :, 3:]
            x = x[:, :, :3]
        x = torch.bmm(x, trans)
        if D > 3:
            x = torch.cat([x, feature], dim=2)
        x = x.transpose(2, 1)
        # x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.conv1(x))

        if self.feature_transform:
            trans_feat = self.fstn(x)
            x = x.transpose(2, 1)
            x = torch.bmm(x, trans_feat)
            x = x.transpose(2, 1)
        else:
            trans_feat = None

        # pointfeat = x
        # x = F.relu(self.bn2(self.conv2(x)))
        # x = F.relu(self.bn3(self.conv3(x)))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))
        x = x.transpose(2,1)
        return x


class BAGLayer(nn.Module):
    def __init__(self, radius=10000, D=256, K=32, channel=6):
        super(BAGLayer, self).__init__()
        self.radius = radius
        self.K = K
        self.center_conv1 = nn.Conv2d(channel, D, 1)
        self.center_conv2 = nn.Conv2d(D, K, 1)
        self.edge_conv = nn.Conv2d(channel, D, 1)
        self.nei_conv = nn.Conv2d(channel, D, 1)
        # self.bn1 = nn.BatchNorm2d(D)
        # self.bn2 = nn.BatchNorm2d(K)

    def forward(self, x, allpoints):
        B, C, N = x.size()
        # print("BAGLayer", x.shape)
        # 获得center point features, edge features, neighbor points features
        x = x.transpose(2, 1)
        allpoints = allpoints.transpose(2, 1)
        nei_idx = query_ball_point(self.radius, self.K, allpoints, x)
        nei_points = index_points(allpoints, nei_idx) # (B, N, K, C)
        # print(f'nei points shape {nei_points.shape}')
        x = torch.unsqueeze(x, dim=2)
        x_repeat = x.repeat(1, 1, self.K, 1) # (B, N, K, C)
        # print(f'x_repeat shape {x_repeat.shape}')
        edge_features = torch.log(x_repeat - nei_points) # (B, N, K, C)

        x_before = x + torch.sum(edge_features, dim=2, keepdim=True) # (B, N, 1, C)
        # print(f'x {x.shape}')

        x_before = x_before.permute(0, 3, 1, 2)
        # print(f'x_before shape {x_before.shape}')
        # x_after = F.relu(self.bn1(self.conv1(x_before))) # (B, N, 1, D)
        x_after = F.relu(self.center_conv1(x_before)) # (B, N, 1, D)
        # print(f'x_after shape {x_after.shape}')
        x_after = x_after.permute(0, 2, 3, 1)
        # print(f'x_after shape {x_after.shape}')

        edge_vertex_feature = edge_features + nei_points
        
        edge_features = edge_features.permute(0, 3, 1, 2)
        # edge_features = F.relu(self.bn1(self.conv1(edge_features))) # (B, N, K, D)
        edge_features = F.relu(self.edge_conv(edge_features)) # (B, N, K, D)
        edge_features = edge_features.permute(0, 2, 3, 1)
        
        edge_vertex_feature = edge_vertex_feature.permute(0, 3, 1, 2)
        # edge_vertex_feature = F.relu(self.bn1(self.conv1(edge_vertex_feature))) # (B, N, K, D)
        edge_vertex_feature = F.relu(self.nei_conv(edge_vertex_feature)) # (B, N, K, D)
        edge_vertex_feature = edge_vertex_feature.permute(0, 2, 3, 1)

        # print(x_after.shape, edge_vertex_feature.shape, edge_features.shape)
        x_after = x_after + torch.sum(edge_vertex_feature, dim=2, keepdim=True) - torch.sum(edge_features, dim=2, keepdim=True)
        # print(f'x_after {x_after.shape}')

        x_after = x_after.permute(0, 3, 1, 2)
        # x_after = F.relu(self.bn2(self.conv2(x_after)))
        x_after = F.relu(self.center_conv2(x_after))
        x_after = x_after.permute(0, 2, 3, 1)
        attention_coff = F.softmax(x_after, dim=-1)

        # print(f'attention_coff {attention_coff.shape}, edge_vertex {edge_vertex_feature.shape}')
        bound_features = torch.bmm(torch.squeeze(attention_coff, 0), torch.squeeze(edge_vertex_feature, 0))
        # print(f'bound_features {bound_features.shape}')

        return bound_features.unsqueeze(0).squeeze(2)