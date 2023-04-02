import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

DEVICE = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
numClasses = 4
numPoints = 4


provNum, alphaNum, adNum = 38, 25, 35

class wR2(nn.Module):
    def __init__(self, num_classes=1000):
        super(wR2, self).__init__()
        hidden1 = nn.Sequential(
            nn.Conv2d(in_channels=3, out_channels=48, kernel_size=5, padding=2, stride=2),
            nn.BatchNorm2d(num_features=48),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2, padding=1),
            nn.Dropout(0.2)
        )
        hidden2 = nn.Sequential(
            nn.Conv2d(in_channels=48, out_channels=64, kernel_size=5, padding=2),
            nn.BatchNorm2d(num_features=64),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=1, padding=1),
            nn.Dropout(0.2)
        )
        hidden3 = nn.Sequential(
            nn.Conv2d(in_channels=64, out_channels=128, kernel_size=5, padding=2),
            nn.BatchNorm2d(num_features=128),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2, padding=1),
            nn.Dropout(0.2)
        )
        hidden4 = nn.Sequential(
            nn.Conv2d(in_channels=128, out_channels=160, kernel_size=5, padding=2),
            nn.BatchNorm2d(num_features=160),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=1, padding=1),
            nn.Dropout(0.2)
        )
        hidden5 = nn.Sequential(
            nn.Conv2d(in_channels=160, out_channels=192, kernel_size=5, padding=2),
            nn.BatchNorm2d(num_features=192),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2, padding=1),
            nn.Dropout(0.2)
        )
        hidden6 = nn.Sequential(
            nn.Conv2d(in_channels=192, out_channels=192, kernel_size=5, padding=2),
            nn.BatchNorm2d(num_features=192),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=1, padding=1),
            nn.Dropout(0.2)
        )
        hidden7 = nn.Sequential(
            nn.Conv2d(in_channels=192, out_channels=192, kernel_size=5, padding=2),
            nn.BatchNorm2d(num_features=192),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2, padding=1),
            nn.Dropout(0.2)
        )
        hidden8 = nn.Sequential(
            nn.Conv2d(in_channels=192, out_channels=192, kernel_size=5, padding=2),
            nn.BatchNorm2d(num_features=192),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=1, padding=1),
            nn.Dropout(0.2)
        )
        hidden9 = nn.Sequential(
            nn.Conv2d(in_channels=192, out_channels=192, kernel_size=3, padding=1),
            nn.BatchNorm2d(num_features=192),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2, padding=1),
            nn.Dropout(0.2)
        )
        hidden10 = nn.Sequential(
            nn.Conv2d(in_channels=192, out_channels=192, kernel_size=3, padding=1),
            nn.BatchNorm2d(num_features=192),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=1, padding=1),
            nn.Dropout(0.2)
        )
        self.features = nn.Sequential(
            hidden1,
            hidden2,
            hidden3,
            hidden4,
            hidden5,
            hidden6,
            hidden7,
            hidden8,
            hidden9,
            hidden10
        )
        self.classifier = nn.Sequential(
            nn.Linear(23232, 100),
            # nn.ReLU(inplace=True),
            nn.Linear(100, 100),
            # nn.ReLU(inplace=True),
            nn.Linear(100, num_classes),
        )

    def forward(self, x):
        x1 = self.features(x)  # [1,3,480,480]----->[1,192,11,11]
        x11 = x1.view(x1.size(0), -1)  # [1,192,11,11]----->[1,23232]
        x = self.classifier(x11)  # [1,23232]----->[1,1000]
        return x


class fh02(nn.Module):
    def __init__(self, num_points, num_classes, wrPath=None):
        super(fh02, self).__init__()
        self.load_wR2(wrPath)
        self.classifier1 = nn.Sequential(
            # nn.Dropout(),
            nn.Linear(53248, 128),
            # nn.ReLU(inplace=True),
            # nn.Dropout(),
            nn.Linear(128, provNum),
        )
        self.classifier2 = nn.Sequential(
            # nn.Dropout(),
            nn.Linear(53248, 128),
            # nn.ReLU(inplace=True),
            # nn.Dropout(),
            nn.Linear(128, alphaNum),
        )
        self.classifier3 = nn.Sequential(
            # nn.Dropout(),
            nn.Linear(53248, 128),
            # nn.ReLU(inplace=True),
            # nn.Dropout(),
            nn.Linear(128, adNum),
        )
        self.classifier4 = nn.Sequential(
            # nn.Dropout(),
            nn.Linear(53248, 128),
            # nn.ReLU(inplace=True),
            # nn.Dropout(),
            nn.Linear(128, adNum),
        )
        self.classifier5 = nn.Sequential(
            # nn.Dropout(),
            nn.Linear(53248, 128),
            # nn.ReLU(inplace=True),
            # nn.Dropout(),
            nn.Linear(128, adNum),
        )
        self.classifier6 = nn.Sequential(
            # nn.Dropout(),
            nn.Linear(53248, 128),
            # nn.ReLU(inplace=True),
            # nn.Dropout(),
            nn.Linear(128, adNum),
        )
        self.classifier7 = nn.Sequential(
            # nn.Dropout(),
            nn.Linear(53248, 128),
            # nn.ReLU(inplace=True),
            # nn.Dropout(),
            nn.Linear(128, adNum),
        )

    def load_wR2(self, path):
        self.wR2 = wR2(numPoints)
        self.wR2 = torch.nn.DataParallel(self.wR2, device_ids=range(torch.cuda.device_count()))
        if not path is None:
            self.wR2.load_state_dict(torch.load(path))
            # self.wR2 = self.wR2.cuda()
        # for param in self.wR2.parameters():
        #     param.requires_grad = False

    def forward(self, x):
        x0 = self.wR2.module.features[0](x)  # [1,3,480,480]--->[1,48,121,121]
        _x1 = self.wR2.module.features[1](x0)  # [1,48,121,121]--->[1,64,122,122]
        x2 = self.wR2.module.features[2](_x1)  # [1,64,122,122]--->[1,128,62,62]
        _x3 = self.wR2.module.features[3](x2)  # [1,128,62,62]--->[1,160,63,63]
        x4 = self.wR2.module.features[4](_x3)  # [1,160,63,63]--->[1,192,32,32]
        _x5 = self.wR2.module.features[5](x4)  # [1,192,32,32]--->[1,192,33,33]

        x6 = self.wR2.module.features[6](_x5)  # [1,192,33,33]--->[1,192,17,17]
        x7 = self.wR2.module.features[7](x6)  # [1,192,17,17]--->[1,192,18,18]
        x8 = self.wR2.module.features[8](x7)  # [1,192,18,18]--->[1,192,10,10]
        x9 = self.wR2.module.features[9](x8)  # [1,192,10,10]--->[1,192,11,11]
        x9 = x9.view(x9.size(0), -1)  # [1,192,11,11]--->[1,23232]
        boxLoc = self.wR2.module.classifier(x9)  # [1,23232]--->[1,4]   #预测车牌位置

        # 多尺度卷积：在不同层数的feature map在通道维数上进行拼接，然后预测
        with torch.no_grad():
            h1, w1 = _x1.data.size()[2], _x1.data.size()[3]
            p1 = torch.FloatTensor([[w1, 0, 0, 0], [0, h1, 0, 0], [0, 0, w1, 0], [0, 0, 0, h1]]).cuda()
            h2, w2 = _x3.data.size()[2], _x3.data.size()[3]
            p2 = torch.FloatTensor([[w2, 0, 0, 0], [0, h2, 0, 0], [0, 0, w2, 0], [0, 0, 0, h2]]).cuda()
            h3, w3 = _x5.data.size()[2], _x5.data.size()[3]
            p3 = torch.FloatTensor([[w3, 0, 0, 0], [0, h3, 0, 0], [0, 0, w3, 0], [0, 0, 0, h3]]).cuda()

            # x, y, w, h --> x1, y1, x2, y2
            assert boxLoc.data.size()[1] == 4
            postfix = torch.FloatTensor([[1, 0, 1, 0], [0, 1, 0, 1], [-0.5, 0, 0.5, 0], [0, -0.5, 0, 0.5]]).cuda()
            boxNew = boxLoc.mm(postfix).clamp(min=0, max=1)

            # input = torch.rand(2, 1, 10, 10)
            # rois = torch.LongTensor([[0, 1, 2, 7, 8], [0, 3, 3, 8, 8], [1, 3, 3, 8, 8]])
            roi1 = roi_pooling_ims(_x1, boxNew.mm(p1), size=(16, 8))  # [1,64,16,8]
            roi2 = roi_pooling_ims(_x3, boxNew.mm(p2), size=(16, 8))  # [1,160,16,8]
            roi3 = roi_pooling_ims(_x5, boxNew.mm(p3), size=(16, 8))  # [1,192,16,8]
            rois = torch.cat((roi1, roi2, roi3), 1)  # [1,416,16,8]

            _rois = rois.view(rois.size(0), -1)  # [1,53248]

            y0 = self.classifier1(_rois)  # [1,38]  预测车牌的第一个字符：省份
            y1 = self.classifier2(_rois)  # [1,25]  预测车牌的第二个字符：市
            y2 = self.classifier3(_rois)  # [1,35]  预测车牌的第三个字符
            y3 = self.classifier4(_rois)  # [1,35]  预测车牌的第四个字符
            y4 = self.classifier5(_rois)  # [1,35]  预测车牌的第五个字符
            y5 = self.classifier6(_rois)  # [1,35]  预测车牌的第六个字符
            y6 = self.classifier7(_rois)  # [1,35]  预测车牌的第七个字符
            return boxLoc, [y0, y1, y2, y3, y4, y5, y6]


def roi_pooling_ims(input, rois, size=(7, 7), spatial_scale=1.0):
    # written for one roi one image
    # size: (w, h)
    assert (rois.dim() == 2)
    assert len(input) == len(rois)
    assert (rois.size(1) == 4)
    output = []
    rois = rois.data.float()
    num_rois = rois.size(0)

    rois[:, 1:].mul_(spatial_scale)
    rois = rois.long()
    for i in range(num_rois):
        roi = rois[i]
        im = input.narrow(0, i, 1)[..., roi[1]:(roi[3] + 1), roi[0]:(roi[2] + 1)]
        output.append(F.adaptive_max_pool2d(im, size))

    return torch.cat(output, 0)




def regcognise(path):
    fh02_model_path = "./models/fh02.pth"
    wr2_model_path = "./models/wR2.pth"
    fh02_ = fh02(numPoints,numClasses,wr2_model_path)
    fh02_ = torch.nn.DataParallel(fh02_, device_ids=range(torch.cuda.device_count()))
    fh02_.load_state_dict(torch.load(fh02_model_path))
    fh02_ = fh02_.to(DEVICE)

    fh02_.eval()
    imageSize = (480, 480)

    img = cv2.imread(path)
    resizedImage = cv2.resize(img, imageSize)
    resizedImage = np.transpose(resizedImage, (2, 0, 1))
    resizedImage = resizedImage.astype('float32')
    resizedImage /= 255.0

    provinces = ["皖", "沪", "津", "渝", "冀", "晋", "蒙", "辽", "吉", "黑", "苏", "浙", "京", "闽", "赣", "鲁",
                 "豫", "鄂", "湘", "粤",
                 "桂",
                 "琼", "川", "贵", "云", "藏", "陕", "甘", "青", "宁", "新", "警", "学", "O"]
    alphabets = ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'J', 'K', 'L', 'M', 'N', 'P', 'Q', 'R', 'S', 'T', 'U', 'V',
                 'W',
                 'X', 'Y', 'Z', 'O']
    ads = ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'J', 'K', 'L', 'M', 'N', 'P', 'Q', 'R', 'S', 'T', 'U', 'V', 'W',
           'X',
           'Y', 'Z', '0', '1', '2', '3', '4', '5', '6', '7', '8', '9', 'O']



    XI = resizedImage

    x = torch.tensor(XI).unsqueeze(0).to(DEVICE)

    # Forward pass: Compute predicted y by passing x to the model
    fps_pred, y_pred = fh02_(x)

    outputY = [el.data.cpu().numpy().tolist() for el in y_pred]
    labelPred = [t[0].index(max(t[0])) for t in outputY]

    [cx, cy, w, h] = fps_pred.data.cpu().numpy()[0].tolist()

    cv2Img = img
    left_up = [(cx - w / 2) * cv2Img.shape[1], (cy - h / 2) * cv2Img.shape[0]]
    right_down = [(cx + w / 2) * cv2Img.shape[1], (cy + h / 2) * cv2Img.shape[0]]
    cv2.rectangle(cv2Img, (int(left_up[0]), int(left_up[1])), (int(right_down[0]), int(right_down[1])),
                  (0, 0, 255),
                  2)
    #   The first character is Chinese character, can not be printed normally, thus is omitted.
    lpn = provinces[labelPred[0]] + alphabets[labelPred[1]] + ads[labelPred[2]] + ads[labelPred[3]] + ads[
        labelPred[4]] + ads[labelPred[5]] + ads[labelPred[6]]

    return lpn











