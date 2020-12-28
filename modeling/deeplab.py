import torch
import torch.nn as nn
import torch.nn.functional as F
from modeling.sync_batchnorm.batchnorm import SynchronizedBatchNorm2d
from modeling.aspp import build_aspp
from modeling.decoder import build_decoder
from modeling.backbone import build_backbone

class DeepLab(nn.Module):
    def __init__(self, backbone='resnet', num_classes=21,
                 sync_bn=True, freeze_bn=False, v2=False, contrastive_dimension = 256):
        super(DeepLab, self).__init__()
        self.v2=v2
        if v2:
            output_stride = 8
        else:
            output_stride = 16

        if sync_bn == True:
            BatchNorm = SynchronizedBatchNorm2d
        else:
            BatchNorm = nn.BatchNorm2d

        self.backbone = build_backbone(backbone, output_stride, BatchNorm)
        self.aspp = build_aspp(backbone, output_stride, BatchNorm, v2=v2, num_classes=num_classes)
        if not v2:
            self.decoder = build_decoder(num_classes, backbone, BatchNorm)

        if self.v2:
            dim_in = 2048
            feat_dim = contrastive_dimension
        else:
            dim_in = 256
            feat_dim = contrastive_dimension

        self.projection_head = nn.Sequential(
            nn.Linear(dim_in, feat_dim),
            nn.BatchNorm1d(feat_dim),
            nn.ReLU(inplace=True),
            nn.Linear(feat_dim, feat_dim)
        )
        self.prediction_head = nn.Sequential(
            nn.Linear(feat_dim, feat_dim),
            nn.BatchNorm1d(feat_dim),
            nn.ReLU(inplace=True),
            nn.Linear(feat_dim, feat_dim)
        )

        self.freeze_bn = freeze_bn

    def forward(self, input, return_features=False):
        x_enc, low_level_feat = self.backbone(input)
        x = self.aspp(x_enc)
        if not self.v2:
            x = self.decoder(x, low_level_feat, return_features=return_features)

        if return_features and not self.v2:# unpack features from deepalb v3
            x, features = x
        elif return_features and self.v2: #  unpack features from deepalb v2
            features = x_enc

        x = F.interpolate(x, size=input.size()[2:], mode='bilinear', align_corners=True)

        if return_features:
            return x, features
        else:
            return x

    def forward_projection_head(self, features):
        return self.projection_head(features)

    def forward_prediction_head(self, features):
        return self.prediction_head(features)

    def freeze_bn_now(self):
        for m in self.modules():
            if isinstance(m, SynchronizedBatchNorm2d):
                m.eval()
            elif isinstance(m, nn.BatchNorm2d):
                m.eval()
            elif isinstance(m, nn.BatchNorm1d):
                m.eval()

    def unfreeze_bn_now(self):
        for m in self.modules():
            if isinstance(m, SynchronizedBatchNorm2d):
                m.train()
            elif isinstance(m, nn.BatchNorm2d):
                m.train()
            elif isinstance(m, nn.BatchNorm1d):
                m.train()


    def get_1x_lr_params(self):
        modules = [self.backbone]
        for i in range(len(modules)):
            for m in modules[i].named_modules():

                if self.freeze_bn:
                    if isinstance(m[1], nn.Conv2d  )  or isinstance(m[1], nn.BatchNorm1d)  or isinstance(m[1], nn.Linear):
                        for p in m[1].parameters():
                            if p.requires_grad:
                                yield p
                else:
                    if isinstance(m[1], nn.Conv2d) or isinstance(m[1], SynchronizedBatchNorm2d) \
                            or isinstance(m[1], nn.BatchNorm2d)  or isinstance(m[1], nn.BatchNorm1d)  or isinstance(m[1], nn.Linear):
                        for p in m[1].parameters():
                            if p.requires_grad:
                                yield p

    def get_10x_lr_params(self):
        if self.v2:
            modules = [self.aspp, self.prediction_head, self.projection_head]
        else:
            modules = [self.aspp, self.decoder, self.prediction_head, self.projection_head]

        for i in range(len(modules)):
            for m in modules[i].named_modules():

                if self.freeze_bn:
                    if isinstance(m[1], nn.Conv2d)  or isinstance(m[1], nn.BatchNorm1d)  or isinstance(m[1], nn.Linear):
                        for p in m[1].parameters():
                            if p.requires_grad:
                                yield p
                else:
                    if isinstance(m[1], nn.Conv2d) or isinstance(m[1], SynchronizedBatchNorm2d) \
                            or isinstance(m[1], nn.BatchNorm2d)  or isinstance(m[1], nn.BatchNorm1d)  or isinstance(m[1], nn.Linear):
                        for p in m[1].parameters():
                            if p.requires_grad:
                                yield p




    def freeze_network_but_dropout(self):
        is_next_learned_dropout = False
        for m in self.modules():

            if isinstance(m, nn.Conv2d) or isinstance(m, SynchronizedBatchNorm2d)  or isinstance(m, nn.BatchNorm2d) \
                or isinstance(m, nn.BatchNorm1d)  or isinstance(m, nn.Linear):
                if not is_next_learned_dropout: # freeze all but learned dropout
                    m.eval()
                else:
                    m.train()


    def freeze_just_dropout(self):
        is_next_learned_dropout = False
        for m in self.modules():

            if isinstance(m, nn.Conv2d) or isinstance(m, SynchronizedBatchNorm2d)  or isinstance(m, nn.BatchNorm2d) \
                or isinstance(m, nn.BatchNorm1d)  or isinstance(m, nn.Linear):
                if is_next_learned_dropout: # freeze all but learned dropout
                    m.eval()
                else:
                    m.train()


if __name__ == "__main__":
    model = DeepLab(backbone='mobilenet', output_stride=16)
    model.eval()
    input = torch.rand(1, 3, 513, 513)
    output = model(input)
    print(output.size())


