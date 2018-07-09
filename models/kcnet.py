import os, sys

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(BASE_DIR)
sys.path.append(os.path.join(BASE_DIR, "../utils"))

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
from utils.kcnet_utils import LoaclGeometricStructure, batch_knn, graph_max_pooling


class KCNetClassify(nn.Module):

    def __init__(self, class_nums, device_id=0, initial_weights=True):
        super(KCNetClassify, self).__init__()

        self.class_nums = class_nums
        self.knn_points = 16
        self.device_id = device_id

        if initial_weights:
            self.initialize_weights()

        self.kc = LoaclGeometricStructure(32, 16, 0.005)
        self.mlp1 = nn.Sequential(
            nn.Conv1d(32 + 3, 64, 1),
            nn.BatchNorm1d(64),
            nn.ReLU(True),
            nn.Conv1d(64, 64, 1),
            nn.BatchNorm1d(64),
            nn.ReLU(True)
        )
        self.mlp2 = nn.Sequential(
            nn.Conv1d(64, 64, 1),
            nn.BatchNorm1d(64),
            nn.ReLU(True),
            nn.Conv1d(64, 128, 1),
            nn.BatchNorm1d(128),
            nn.ReLU(True)
        )
        self.mlp3 = nn.Sequential(
            nn.Conv1d(192, 1024, 1),
            nn.BatchNorm1d(1024),
            nn.ReLU(True)
        )
        self.classify = nn.Sequential(
            nn.Linear(1024, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(True),
            nn.Linear(512, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(True),
            nn.Dropout(0.5),
            nn.Linear(256, class_nums)
        )

        self.criterion = nn.CrossEntropyLoss()
        self.optimizer = optim.Adam(self.parameters(), weight_decay=1e-5)
        self.schedule = optim.lr_scheduler.StepLR(self.optimizer, 20, 0.5)

        self.cuda(device_id)

    def forward(self, points):
        knn_graph, _ = batch_knn(points, points.clone(), self.knn_points + 1)
        x = self.kc(points, knn_graph[:, :, 1:].contiguous())
        x = torch.cat([points, x], dim=1)
        x = self.mlp1(x)
        y = graph_max_pooling(x, knn_graph)
        x = self.mlp2(x)
        x = torch.cat([x, y], dim=1)
        x = self.mlp3(x)
        x = F.max_pool1d(x, x.size(2), stride=1).squeeze(2)
        x = self.classify(x)
        return x

    def initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv1d):
                m.weight.data.normal_(0, 0.01)
                if m.bias is not None:
                    m.bias.data.zero_()
            elif isinstance(m, nn.Linear):
                m.weight.data.normal_(0, 0.01)
                m.bias.data.zero_()
            elif isinstance(m, nn.BatchNorm1d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

    def loss(self, outputs, targets):
        return self.criterion(outputs, targets)

    def fit(self, dataloader, epoch):
        self.train()
        batch_loss = 0.
        epoch_loss = 0.
        batch_nums = 0
        if self.schedule is not None:
            self.schedule.step()

        print('----------epoch %d start train----------' % epoch)

        for batch_idx, (inputs, targets) in enumerate(dataloader):
            inputs = inputs.cuda(self.device_id)
            targets = targets.cuda(self.device_id)
            self.optimizer.zero_grad()

            outputs = self(inputs)
            losses = self.loss(outputs, targets)
            losses.backward()
            self.optimizer.step()

            batch_loss += losses.item()
            epoch_loss += losses.item()
            batch_nums += 1
            if (batch_idx + 1) % 4 == 0:
                print('[%d, %5d] loss %.3f' % (epoch, batch_idx, batch_loss / 4))
                batch_loss = 0.

        print('-----------epoch %d end train-----------' % epoch)
        print('epoch %d loss %.3f' % (epoch, epoch_loss / batch_nums))

        return epoch_loss / batch_nums

    def score(self, dataloader):
        self.eval()
        correct = 0.
        total = 0

        with torch.no_grad():
            for batch_idx, (inputs, targets) in enumerate(dataloader):
                inputs = inputs.cuda(self.device_id)
                targets = targets.cuda(self.device_id)

                outputs = self(inputs)
                _, predicted = torch.max(outputs.data, 1)
                total += targets.size(0)
                correct += (predicted == targets).sum().item()

        print('Accuracy of the network on the test images: %d %%' % (100 * correct / total))

        return correct / total


class KCNetSegment(nn.Module):

    def __init__(self, class_nums, category_nums, device_id=None, initial_weights=True):
        super(KCNetSegment, self).__init__()

        self.class_nums = class_nums
        self.category_nums = category_nums
        self.knn_points = 18
        self.device_id = device_id

        self.kc = LoaclGeometricStructure(16, 18, 0.005)
        self.mlp1 = nn.Sequential(
            nn.Conv1d(3 + 16, 64, 1),
            nn.BatchNorm1d(64),
            nn.ReLU(True)
        )
        self.mlp2 = nn.Sequential(
            nn.Conv1d(64, 64, 1),
            nn.BatchNorm1d(64),
            nn.ReLU(True)
        )
        self.mlp3 = nn.Sequential(
            nn.Conv1d(64, 128, 1),
            nn.BatchNorm1d(128),
            nn.ReLU(True)
        )
        self.mlp4 = nn.Sequential(
            nn.Conv1d(128, 128, 1),
            nn.BatchNorm1d(128),
            nn.ReLU(True)
        )
        self.mlp5 = nn.Sequential(
            nn.Conv1d(128, 512, 1),
            nn.BatchNorm1d(512),
            nn.ReLU(True)
        )
        self.mlp6 = nn.Sequential(
            nn.Conv1d(512, 1024, 1),
            nn.BatchNorm1d(1024),
            nn.ReLU(True)
        )
        self.mlp7 = nn.Sequential(
            nn.Conv1d(3 + 16 + 64 + 64 + 128 + 128 + 512 + 1024 + category_nums, 512, 1),
            nn.BatchNorm1d(512),
            nn.ReLU(True),
            nn.Conv1d(512, 256, 1),
            nn.BatchNorm1d(256),
            nn.ReLU(True),
            nn.Dropout(0.3),
            nn.Conv1d(256, class_nums, 1)
        )

        if initial_weights:
            self.initialize_weights()

        self.criterion = nn.CrossEntropyLoss()
        self.optimizer = optim.Adam(self.parameters(), weight_decay=1e-5)
        self.schedule = optim.lr_scheduler.StepLR(self.optimizer, 20, 0.5)

        self.cuda(device_id)

    def forward(self, points, labels):
        knn_graph, _ = batch_knn(points, points.clone(), self.knn_points + 1)
        x1 = self.kc(points, knn_graph[:, :, 1:].contiguous())
        x1 = torch.cat([points, x1], dim=1)
        x2 = self.mlp1(x1)
        x3 = self.mlp2(x2)
        x4 = self.mlp3(x3)
        x5 = graph_max_pooling(x4, knn_graph)
        x5 = self.mlp4(x5)
        x6 = self.mlp5(x5)
        x7 = graph_max_pooling(x6, knn_graph)
        x7 = self.mlp6(x7)
        x7 = F.max_pool1d(x7, x7.size(2), stride=1)
        x7 = x7.repeat([1, 1, knn_graph.size(1)])

        index = labels.unsqueeze(1).repeat([1, knn_graph.size(1)]).unsqueeze(1)
        one_hot = torch.zeros([knn_graph.size(0), self.category_nums, knn_graph.size(1)])
        one_hot = one_hot.cuda(self.device_id)
        one_hot = one_hot.scatter_(1, index, 1)

        x = torch.cat([x1, x2, x3, x4, x5, x6, x7, one_hot], dim=1)
        x = self.mlp7(x)

        return x

    def initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv1d):
                m.weight.data.normal_(0, 0.01)
                if m.bias is not None:
                    m.bias.data.zero_()
            elif isinstance(m, nn.Linear):
                m.weight.data.normal_(0, 0.01)
                m.bias.data.zero_()
            elif isinstance(m, nn.BatchNorm1d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

    def loss(self, outputs, targets):
        return self.criterion(outputs, targets)

    def fit(self, dataloader, epoch):
        self.train()
        batch_loss = 0.
        epoch_loss = 0.
        batch_nums = 0
        if self.schedule is not None:
            self.schedule.step()

        print('----------epoch %d start train----------' % epoch)

        for batch_idx, (inputs, targets, labels) in enumerate(dataloader):
            inputs = inputs.cuda(self.device_id)
            targets = targets.cuda(self.device_id)
            labels = labels.cuda(self.device_id)
            self.optimizer.zero_grad()

            outputs = self(inputs, labels)
            losses = self.loss(outputs, targets)
            losses.backward()
            self.optimizer.step()

            batch_loss += losses.item()
            epoch_loss += losses.item()
            batch_nums += 1
            if (batch_idx + 1) % 4 == 0:
                print('[%d, %5d] loss %.3f' % (epoch, batch_idx, batch_loss / 4))
                batch_loss = 0.

        print('-----------epoch %d end train-----------' % epoch)
        print('epoch %d loss %.3f' % (epoch, epoch_loss / batch_nums))

        return epoch_loss / batch_nums

    def score(self, dataloader):
        self.eval()
        correct = 0.
        total = 0

        with torch.no_grad():
            for batch_idx, (inputs, targets, labels) in enumerate(dataloader):
                inputs = inputs.cuda(self.device_id)
                targets = targets.cuda(self.device_id)
                labels = labels.cuda(self.device_id)

                outputs = self(inputs, labels)
                _, predicted = torch.max(outputs.data, 1)
                total += targets.size(0) * targets.size(1)
                correct += (predicted == targets).sum().item()

        print('Accuracy of the network on the test images: %d %%' % (100 * correct / total))

        return correct / total
