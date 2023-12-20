import argparse

import numpy as np
import torch
from matplotlib import pyplot as plt
from sklearn.metrics.pairwise import cosine_similarity, pairwise_distances
import os
from torch.utils.tensorboard import SummaryWriter
from myutil import *
from model import *
from post_clustering import *

setup_seed(100)
parser = argparse.ArgumentParser(description="use pretraining net work for feature extract")
config = get_config(parser)


# data special
num_cluster = config['cluster']['num_cluster']
dim_subspace = config['cluster']['dim_subspace']
ro = config['cluster']['ro']
alpha = config['cluster']['alpha']


features_save_dir = "features"

print(f"image_size is {config['features_extract']['image_size']}")
model_name = "ibot_vitb16"
features_path = os.path.join(config['features_extract']['features_save_dir'],
                                 model_name + "_" + config['Dataset'] + "imgSize=" + str(config['features_extract']['image_size']) + ".pt")

saved_features = torch.load(features_path)
features = saved_features['data']
label = saved_features['label']
del saved_features
# 使用features构建graph
sparse_adj = bulid_graph(features, K=3)
datas = bulid_pyg_data(features, sparse_adj)

# model
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model = GCNCluster(features=features, hidden_channels=16, num_sample=features.shape[0])
model.to(device)

# loss and optimizer
optimizer = torch.optim.Adam(model.parameters(), lr=config['train']['lr'])


# 创建SummaryWriter对象
log_dir = 'logs'
writer = SummaryWriter(log_dir)

# 将配置信息记录到TensorBoard
config_str = yaml.dump(config, default_flow_style=False)
writer.add_text('Config', config_str)


for epoch in range(config['train']['epochs']):
    attribute_features, graph_features, gcn_raw_features, gcn_reconstruction, attribute_expression, graph_expression = model(datas)

    # self expression loss
    attribute_express_loss = F.mse_loss(attribute_features, torch.mm(attribute_expression, attribute_features))
    graph_express_loss = F.mse_loss(graph_features, torch.mm(graph_expression, graph_features))

    # 矩阵1范数：绝对值+列和 最大者
    attribute_express_confficient_loss = torch.linalg.matrix_norm(attribute_expression, 1)
    graph_express_confficient_loss = torch.linalg.matrix_norm(graph_expression, 1)

    # 自编码器的重构损失
    con_loss = F.mse_loss(gcn_raw_features, gcn_reconstruction)

    total_loss = config['train']['lambda1'] * attribute_express_confficient_loss + \
                 config['train']['lambda2'] * attribute_express_loss + \
                 config['train']['lambda3'] * graph_express_confficient_loss + \
                 config['train']['lambda4'] * graph_express_loss + \
                 config['train']['lambda5'] * con_loss

    optimizer.zero_grad()
    total_loss.backward()
    optimizer.step()

    # 记录损失变化
    writer.add_scalar("Loss/total_loss", total_loss.item(), epoch)
    writer.add_scalar("Loss/attribute_express_confficient_loss", attribute_express_confficient_loss.item(), epoch)
    writer.add_scalar("Loss/attribute_express_loss", attribute_express_loss.item(), epoch)
    writer.add_scalar("Loss/graph_express_confficient_loss", graph_express_confficient_loss.item(), epoch)
    writer.add_scalar("Loss/graph_express_loss", graph_express_loss.item(), epoch)
    writer.add_scalar("Loss/con_loss", con_loss.item(), epoch)

    if epoch % config['train']['eval_epochs'] == 0:
        print('---------------------------------------------------------')
        loss_tuple = (epoch, total_loss.item(), attribute_express_confficient_loss.item(), attribute_express_loss.item(), graph_express_confficient_loss.item(),
        graph_express_loss.item(), con_loss.item())
        print("epoch :%02d, total_loss :%.4f, attConffLoss : %.4f, attExpLoss : %.4f, graConffLoss : %.4f, graExpLoss:%.4f, con_loss:%.4f" %
              loss_tuple)

        # 融合自表示矩阵
        print("融合结果")
        C = (attribute_expression + graph_expression).detach().to('cpu').numpy()
        y_pred = sklearn_spectral_clustering(C, num_cluster)
        ACC = cluster_accuracy(label, y_pred)
        NMI = nmi(label, y_pred)
        print('Epoch %02d: loss=%.4f, acc=%.4f, nmi=%.4f' %
              (epoch, total_loss.item() / y_pred.shape[0], ACC, NMI))

        writer.add_scalar("Results/ACC", ACC, epoch)
        writer.add_scalar("Results/NMI", NMI, epoch)









