import numpy as np
import torch


def k_adjacency(A, k, with_self=False, self_factor=1):
    # A is a 2D square array
    if isinstance(A, torch.Tensor):
        A = A.data.cpu().numpy()
    assert isinstance(A, np.ndarray)
    Iden = np.eye(len(A), dtype=A.dtype)
    if k == 0:
        return Iden
    Ak = np.minimum(np.linalg.matrix_power(A + Iden, k), 1) - np.minimum(np.linalg.matrix_power(A + Iden, k - 1), 1)
    if with_self:
        Ak += (self_factor * Iden)
    return Ak


def edge2mat(link, num_node):
    A = np.zeros((num_node, num_node))
    for i, j in link:
        A[j, i] = 1
    return A


def normalize_digraph(A, dim=0): # 右乘对角阵，相当于列归一化
    # A is a 2D square array
    Dl = np.sum(A, dim) # dimension为0表示对每一列求和，结果为1*width；为1表示对每一行求和，结果为1*height
    h, w = A.shape
    Dn = np.zeros((w, w))

    for i in range(w):
        if Dl[i] > 0:
            Dn[i, i] = Dl[i] ** (-1) #无向图的邻接矩阵一定是对称矩阵

    AD = np.dot(A, Dn)
    return AD

def normalize_digraph_2(A): # 行列归一化
    # A is a 2D square array
    D1 = np.sum(A, 1) #行和
    D0 = np.sum(A, 0) #列和
    h, w = A.shape
    Dn = np.zeros((w, w))

    for i in range(w):
        if D1[i] > 0:
            Dn[i, i] = D1[i] ** (-1/2)
    AD = np.dot(Dn,A)

    Dn = np.zeros((w, w))
    for i in range(w):
        if D0[i] > 0:
            Dn[i, i] = D0[i] ** (-1/2)
    AD = np.dot(AD,Dn)

    return AD


def get_hop_distance(num_node, edge, max_hop=1): # 计算结点之间的距离，最大距离为max_hop，超过此距离设为inf不可达
    A = np.eye(num_node)

    for i, j in edge:
        A[i, j] = 1
        A[j, i] = 1

    # compute hop steps
    hop_dis = np.zeros((num_node, num_node)) + np.inf
    transfer_mat = [
        np.linalg.matrix_power(A, d) for d in range(max_hop + 1) # 生成一个(max_hop + 1)*num_node*num_node的矩阵
    ]
    arrive_mat = (np.stack(transfer_mat) > 0) #使用np.stack()转换之后可以直接计算>0，返回的矩阵中值为true或false。
    for d in range(max_hop, -1, -1):
        hop_dis[arrive_mat[d]] = d
    return hop_dis


class Graph:
    """The Graph to model the skeletons.

    Args:
        layout (str): must be one of the following candidates: 'openpose', 'nturgb+d', 'coco', 'handmp', 'openpose_25'. Default: 'coco'.
        mode (str): must be one of the following candidates: 'stgcn_spatial', 'spatial'. Default: 'spatial'.
        max_hop (int): the maximal distance between two connected nodes.
            Default: 1
    """

    def __init__(self,
                 layout='coco',
                 mode='spatial',
                 max_hop=1,
                 nx_node=1,
                 num_filter=3,
                 init_std=0.02,
                 init_off=0.04):

        self.max_hop = max_hop
        self.layout = layout
        self.mode = mode
        self.num_filter = num_filter
        self.init_std = init_std
        self.init_off = init_off
        self.nx_node = nx_node

        assert nx_node == 1 or mode == 'random', "nx_node can be > 1 only if mode is 'random'"
        assert layout in ['openpose', 'nturgb+d', 'coco', 'handmp', 'openpose_25']

        self.get_layout(layout)
        self.hop_dis = get_hop_distance(self.num_node, self.inward, max_hop)

        assert hasattr(self, mode), f'Do Not Exist This Mode: {mode}'
        self.A = getattr(self, mode)()

    def __str__(self):
        return self.A

    def get_layout(self, layout):#inward表示从四肢到中心点的连接，outward表示中心点到四肢的连接
        if layout == 'openpose':
            self.num_node = 18
            self.inward = [
                (4, 3), (3, 2), (7, 6), (6, 5), (13, 12), (12, 11), (10, 9),
                (9, 8), (11, 5), (8, 2), (5, 1), (2, 1), (0, 1), (15, 0),
                (14, 0), (17, 15), (16, 14)
            ]
            self.center = 1
        elif layout == 'nturgb+d':
            self.num_node = 25
            neighbor_base = [
                (1, 2), (2, 21), (3, 21), (4, 3), (5, 21), (6, 5), (7, 6),
                (8, 7), (9, 21), (10, 9), (11, 10), (12, 11), (13, 1),
                (14, 13), (15, 14), (16, 15), (17, 1), (18, 17), (19, 18),
                (20, 19), (22, 8), (23, 8), (24, 12), (25, 12)
            ]
            self.inward = [(i - 1, j - 1) for (i, j) in neighbor_base]
            self.center = 21 - 1
        elif layout == 'coco':
            self.num_node = 17
            self.inward = [
                (15, 13), (13, 11), (16, 14), (14, 12), (11, 5), (12, 6),
                (9, 7), (7, 5), (10, 8), (8, 6), (5, 0), (6, 0),
                (1, 0), (3, 1), (2, 0), (4, 2)
            ]
            self.center = 0
        elif layout == 'handmp':
            self.num_node = 21
            self.inward = [
                (1, 0), (2, 1), (3, 2), (4, 3), (5, 0), (6, 5), (7, 6), (8, 7),
                (9, 0), (10, 9), (11, 10), (12, 11), (13, 0), (14, 13),
                (15, 14), (16, 15), (17, 0), (18, 17), (19, 18), (20, 19)
            ]
            self.center = 0
        elif layout == 'openpose_25':
            self.num_node = 25
            self.inward = [
                (4, 3), (3, 2), (2, 1), (7, 6), (6, 5), (5, 1),
                (17, 15), (15, 0), (18, 16), (16, 0), (0, 1), (1, 8),
                (23, 22), (22, 11), (24, 11), (11, 10), (10, 9), (9, 8), 
                (20, 19), (19, 14), (21, 14), (14, 13), (13, 12), (12, 8)
            ]
            self.center = 8
        else:
            raise ValueError(f'Do Not Exist This Layout: {layout}')
        self.self_link = [(i, i) for i in range(self.num_node)]
        self.outward = [(j, i) for (i, j) in self.inward]
        self.neighbor = self.inward + self.outward

    def stgcn_spatial(self):# stgcn中提出的空间关键点划分策略
        adj = np.zeros((self.num_node, self.num_node))
        adj[self.hop_dis <= self.max_hop] = 1
        normalize_adj = normalize_digraph(adj)
        hop_dis = self.hop_dis
        center = self.center

        A = []
        for hop in range(self.max_hop + 1):
            a_close = np.zeros((self.num_node, self.num_node)) # 近心
            a_further = np.zeros((self.num_node, self.num_node)) # 远心
            for i in range(self.num_node):
                for j in range(self.num_node):
                    if hop_dis[j, i] == hop:
                        if hop_dis[j, center] >= hop_dis[i, center]: # (j,i)是近心边，从j邻接i是在靠近center
                            a_close[j, i] = normalize_adj[j, i]
                        else:
                            a_further[j, i] = normalize_adj[j, i] # (j,i)是远心边，从j邻接到i是在远离center
            A.append(a_close) # 当hop等于0时，a_close描述的是根结点。# 当hop大于1时，a_close描述的是近心邻接关系。
            if hop > 0:
                A.append(a_further) # 当hop大于1时，a_further描述的是远心邻接关系。
        return np.stack(A)

    def spatial(self):
        Iden = edge2mat(self.self_link, self.num_node)
        In = normalize_digraph(edge2mat(self.inward, self.num_node))
        Out = normalize_digraph(edge2mat(self.outward, self.num_node))
        A = np.stack((Iden, In, Out))
        return A

    def binary_adj(self):
        A = edge2mat(self.inward + self.outward, self.num_node)
        return A[None]

    def random(self):
        num_node = self.num_node * self.nx_node
        return np.random.randn(self.num_filter, num_node, num_node) * self.init_std + self.init_off
