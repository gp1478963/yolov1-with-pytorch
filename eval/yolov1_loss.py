from torch import nn
import numpy
import torch
import calcate_iou
class YoloV1Loss(nn.Module):
    def __init__(self, lambda_coord=5., lambda_noobj=.5):
        super(YoloV1Loss, self).__init__()
        self.lambda_coord = lambda_coord
        self.lambda_noobj = lambda_noobj

    def forward(self, predictions, targets):
        coord_mask = targets[:, :, 4] == 1
        noobj_mask = targets[:, :, 4] == 0

        coord_data_mask = torch.unsqueeze(coord_mask, dim=-1).expand_as(targets)
        noobj_data_mask = torch.unsqueeze(noobj_mask, dim=-1).expand_as(targets)

        targets_coord = targets[coord_data_mask].reshape(-1, 30)
        targets_noobj = targets[noobj_data_mask].reshape(-1, 30)
        preds_coord = predictions[coord_data_mask].reshape(-1, 30)
        preds_noobj = predictions[noobj_data_mask].reshape(-1, 30)

        for pred_coord, targets_coord in (preds_coord, targets_coord):
            pred_box = torch.hstack((pred_coord[:4], pred_coord[5:9])).reshape(-1, 4)
            target_box = torch.hstack((targets_coord[:4], targets_coord[5:9])).reshape(-1, 4)
            iou = calcate_iou.calc_iou(pred_box, target_box)
            per_prebox_max_iou = torch.max(iou, dim=0)
            per_prebox_max_iou_index = torch.argmax(per_prebox_max_iou, dim=0)







loss_function = YoloV1Loss(lambda_coord=5., lambda_noobj=.5)
pre_mat = torch.rand((7, 7, 30), dtype=torch.float)
pre_mat[1, 1, [4, 9]] = 1
pre_mat[2, 2, [4, 9]] = 1

target_mat = torch.zeros((7, 7, 30), dtype=torch.float)
target_mat[1, 1, [4, 9]] = 1
target_mat[1, 2, [4, 9]] = 1

loss_function(pre_mat, target_mat)




