import torch
import torch.nn.functional as F
import torch.nn as nn



from torch_geometric.nn import GCNConv


class GCN(torch.nn.Module):
    def __init__(self, num_features, hidden_channels):
        super().__init__()
        self.conv1 = GCNConv(num_features, hidden_channels)
        self.conv2 = GCNConv(hidden_channels, num_features)


    def forward(self, x, edge_index):
        x = self.conv1(x, edge_index)
        x = x.relu()
        # x = F.dropout(x, p=0.5, training=self.training)
        x = self.conv2(x, edge_index)
        # x = x.relu()
        return x


class GCN_Encoder(torch.nn.Module):
    def __init__(self, num_features, middle_features ,hidden_channels):
        super().__init__()
        self.conv1 = GCNConv(num_features, middle_features)
        self.conv2 = GCNConv(middle_features, hidden_channels)


    def forward(self, x, edge_index):
        x = self.conv1(x, edge_index)
        x = x.relu()
        # x = F.dropout(x, p=0.5, training=self.training)
        x = self.conv2(x, edge_index)
        # x = x.relu()
        return x


class SelfExpression(nn.Module):
    def __init__(self, n):
        super(SelfExpression, self).__init__()
        self.Coefficient = nn.Parameter(1.0e-8 * torch.ones(n, n, dtype=torch.float32), requires_grad=True)

    def forward(self, x):  # shape=[n, d]
        y = torch.matmul(self.Coefficient, x)
        return y


class GCNCluster(torch.nn.Module):
    def __init__(self, features, hidden_channels, num_sample):
        super(GCNCluster, self).__init__()
        # encoder: (Linear:raw->100  GCN:100 -> 100 -> 10)
        # decoder: (GCN:10 -> 100 -> 100  Linear:100->raw)
        self.W_encoder = nn.Linear(features[-1].shape[-1], 100)
        self.W_deocder = nn.Linear(100, features[-1].shape[-1])


        self.gcnconv_encoder = GCN_Encoder(100, 100, 10)
        self.gcnconv_decoder = GCN_Encoder(10, 100, 100)

        self.attribute_expression = SelfExpression(num_sample)
        self.graph_expression = SelfExpression(num_sample)

        self.W_e = nn.Linear(features[-1].shape[-1], 100)
        self.W_d = nn.Linear(100, features[-1].shape[-1])


    def forward(self, datas):
        # 输入features list中的的数据
        # 输出融合后的矩阵C_F
        # 在主函数中需要对融合后的矩阵C_F计算目标矩阵P
        # 损失函数是KL(C_F, P)
        # breakpoint()
        attribute_features = datas.x

        gcn_raw_features = attribute_features

        gcn_features = self.gcnconv_encoder(self.W_e(gcn_raw_features), datas.edge_index)
        gcn_reconstruction = self.W_d(self.gcnconv_decoder(gcn_features, datas.edge_index))

        return attribute_features, gcn_features, gcn_raw_features, gcn_reconstruction, self.attribute_expression.Coefficient, self.graph_expression.Coefficient








def target_distribution(batch: torch.Tensor) -> torch.Tensor:
    """
    Compute the target distribution p_ij, given the batch (q_ij), as in 3.1.3 Equation 3 of
    Xie/Girshick/Farhadi; this is used the KL-divergence loss function.

    :param batch: [batch size, number of clusters] Tensor of dtype float
    :return: [batch size, number of clusters] Tensor of dtype float
    """
    weight = (batch ** 2) / torch.sum(batch, 0)
    return (weight.t() / torch.sum(weight, 1)).t()



def sim(CF):
    # AASSC-Net method return mat4
    # mat1 = 0.5*(CF + CF.T)
    # mat2 = mat1 - torch.diag(torch.diag(mat1))
    # c_sum = torch.sum(mat2, dim=1, keepdim=True)
    # mat3 = torch.div(mat2, c_sum)
    # mat4 = mat3 + torch.eye(CF.shape[0]).to(CF.device)

    # our method return mat_sim
    safe_CF = CF.abs() + 1e-3
    s = torch.sum(safe_CF, dim=1)
    mat_sim = safe_CF/s
    return mat_sim


class Encoder_conv(nn.Module):

    def __init__(self, data_shape, k=3, NUM=1):
        super(Encoder_conv, self).__init__()
        self.k = k
        self.NUM = NUM
        self.image_width = data_shape[-1]

        # Encoder Network
        self.Encoder_Conv_layers = nn.Sequential(
            nn.Conv2d(1,16 * self.NUM, kernel_size=3, padding=1),
            nn.BatchNorm2d(16 * self.NUM),
            nn.GELU(),
            nn.Conv2d(16 * self.NUM, 16 * self.NUM, kernel_size=3, padding=1),
            nn.BatchNorm2d(16 * self.NUM),
            nn.GELU(),
            nn.MaxPool2d(kernel_size=2),
            nn.Conv2d(16 * self.NUM, 32 * self.NUM, kernel_size=3, padding=1),
            nn.BatchNorm2d(32 * self.NUM),
            nn.GELU(),
            nn.Conv2d(32 * self.NUM, 32 * self.NUM, kernel_size=3, padding=1),
            nn.BatchNorm2d(32 * self.NUM),
            nn.GELU(),
            nn.MaxPool2d(kernel_size=2)
        )
        self.Flatten = nn.Flatten()
        self.linear1 = nn.Linear(int((self.image_width/4)**2*32),1024)
        self.linear2 = nn.Linear(1024, 512)
        self.relu1 = nn.ReLU()

        self.linear_out = nn.Linear(512, 10)


    def forward(self, X):
        X = self.Encoder_Conv_layers(X)
        X = self.Flatten(X)
        X = self.linear1(X)
        X = self.relu1(X)
        X = self.linear2(X)
        X = self.relu1(X)

        encode_out = torch.sigmoid(self.linear_out(X))
        return encode_out







class Decoder_conv(nn.Module):
    def __init__(self, data_shape, NUM=1):
        super(Decoder_conv, self).__init__()
        self.NUM = NUM
        self.image_width = data_shape[-1]

        self.linear1 = nn.Linear(10,512)
        self.linear2 = nn.Linear(512,1024)
        self.relu2 = nn.ReLU()
        self.linear3 = nn.Linear(1024,int((self.image_width/4)**2*32))
        # Decoder Network
        self.Decoder_Conv_layers = nn.Sequential(
            nn.GELU(),
            nn.Upsample(scale_factor=2),
            nn.Conv2d(32 * self.NUM,32 * self.NUM, kernel_size=3, padding=1),
            nn.BatchNorm2d(32 * self.NUM),
            nn.GELU(),
            nn.Conv2d(32 * self.NUM, 32 * self.NUM, kernel_size=3, padding=1),
            nn.BatchNorm2d(32 * self.NUM),
            nn.GELU(),
            nn.Upsample(scale_factor=2),
            nn.Conv2d(32 * self.NUM, 16 * self.NUM, kernel_size=3, padding=1),
            nn.BatchNorm2d(16 * self.NUM),
            nn.GELU(),
            nn.Conv2d(16 * self.NUM, 16 * self.NUM, kernel_size=3, padding=1),
            nn.BatchNorm2d(16 * self.NUM),
            nn.GELU(),
            nn.Conv2d(16 * self.NUM, 1, kernel_size=3, padding=1),
            nn.BatchNorm2d(1),
            nn.Sigmoid()
        )


    def forward(self, X):
        X = self.linear1(X)
        X = self.relu2(X)
        X = self.linear2(X)
        X = self.relu2(X)
        X = self.linear3(X)
        X = X.reshape([X.shape[0],
                       32*self.NUM,
                       int(self.image_width/4),
                       int(self.image_width/4)])
        X = self.Decoder_Conv_layers(X)
        return X


class Autoencoder(nn.Module):
    def __init__(self, data_shape):
        super(Autoencoder, self).__init__()
        self.encoder = Encoder_conv(data_shape)
        self.decoder = Decoder_conv(data_shape)

    def forward(self, x):
        return self.decoder(self.encoder(x))

    def get_middle_feature(self, X):
        features = []
        X = self.encoder.Encoder_Conv_layers(X)
        X = self.encoder.Flatten(X)
        features.append(X) #
        X = self.encoder.linear1(X)
        features.append(X) #
        X = self.encoder.relu1(X)
        X = self.encoder.linear2(X)
        features.append(X) #
        X = self.encoder.relu1(X)
        encode_out = torch.sigmoid(self.encoder.linear_out(X))
        features.append(encode_out)

        return features

