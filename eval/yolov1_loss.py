from torch import nn
import numpy
import torch
import calcate_iou
import torchvision

class YoloV1Loss(nn.Module):
    def __init__(self, lambda_coord=5., lambda_noobj=.5, CELL_SILE=1/7):
        super(YoloV1Loss, self).__init__()
        self.lambda_coord = lambda_coord
        self.lambda_noobj = lambda_noobj
        self.CELL_SILE = CELL_SILE

    def forward(self, predictions, targets):
        coord_mask = targets[:, :, 4] == 1
        noobj_mask = targets[:, :, 4] == 0

        confidence_loss_noobj = nn.functional.mse_loss(predictions[noobj_mask].reshape(-1, 30)[:, [4, 9]],
                                                       targets[noobj_mask].reshape(-1, 30)[:, [4, 9]],
                                                       size_average=False, reduction='sum') * self.lambda_noobj

        have_obj_mask = torch.unsqueeze(coord_mask, dim=-1).expand_as(targets)
        targets_coord = targets[have_obj_mask].reshape(-1, 30)
        preds_coord = predictions[have_obj_mask].reshape(-1, 30)

        # c1, c2, w, h c, p
        coord_coord = torch.zeros((preds_coord.size()[0] // 2, 6))
        coord_coord_t = torch.zeros((preds_coord.size()[0] // 2, 6))

        for (pred_coord, target_coord, iou_index, iou_target_index) in zip(preds_coord, targets_coord,
                                                                        coord_coord, coord_coord_t):
            pred_box = torch.hstack((pred_coord[:4], pred_coord[5:9])).reshape(-1, 4)
            target_box = torch.square(target_coord[:4].reshape(-1, 4))
            pred_box[:, :2] = pred_box[:, :2] * self.CELL_SILE - pred_box[:, 2:4]
            pred_box[:, 2:] = pred_box[:, :2] * self.CELL_SILE + pred_box[:, 2:4]
            target_box[:, :2] = target_box[:, :2] * self.CELL_SILE - target_box[:, 2:4]
            target_box[:, 2:] = target_box[:, :2] * self.CELL_SILE + target_box[:, 2:4]
            iou = torchvision.ops.box_iou(pred_box, target_box)
            # iou = calcate_iou.calc_iou(pred_box, target_box)
            per_prebox_max_iou, per_prebox_max_iou_index = torch.max(iou, dim=0)
            obj_index = torch.argmax(target_coord[10:], dim=0)

            iou_target_index[:4] = target_coord[:4]
            iou_target_index[4] = per_prebox_max_iou
            iou_target_index[5] = 1

            begin_index = per_prebox_max_iou_index*5
            iou_index[:4] = pred_coord[begin_index: begin_index + 4]
            iou_index[4] = pred_coord[begin_index + 4]
            iou_index[5] = pred_coord[10 + obj_index]

        coordence_center_loss = nn.functional.mse_loss(coord_coord[:, :2], coord_coord_t[:, :2],
                                                       size_average=False, reduction='sum') * self.lambda_coord
        ccoord_xy_loss = nn.functional.mse_loss(coord_coord[:, 2:4], coord_coord_t[:, 2:4],
                                                size_average=False) * self.lambda_coord
        confidence_loss = nn.functional.mse_loss(coord_coord[:, 5], coord_coord_t[:, 5],
                                                 size_average=False, reduction='sum') * self.lambda_coord
        classier_loss = nn.functional.mse_loss(coord_coord[:, -1], coord_coord_t[:, -1],
                                               size_average=False, reduction='sum') * self.lambda_coord

        return coordence_center_loss + ccoord_xy_loss + confidence_loss + classier_loss + confidence_loss_noobj


if __name__ == '__main__':
    loss_function = YoloV1Loss(lambda_coord=5., lambda_noobj=.5, CELL_SILE=1/7)
    pre_mat = torch.rand((7, 7, 30), dtype=torch.float)
    pre_mat[0, 0, [4, 9, 19]] = 1
    pre_mat[2, 2, [4, 9, 23]] = 1
    target_mat = torch.zeros((7, 7, 30), dtype=torch.float)
    target_mat[0, 0, [4, 9, 20]] = 1
    target_mat[1, 2, [4, 9, 23]] = 1
    print(loss_function(pre_mat, target_mat))

# mat1 = torch.rand((1, 3))
#
# cer = nn.MSELoss()
# mat2 = torch.ones((1, 3))
# mat3 = mat2 - mat1
# print(mat3)
# print(cer((mat1, mat1)))
