import torch
import torchvision
from torch import nn


class YoloV1Loss(nn.Module):
    def __init__(self, lambda_coord=5., lambda_noobj=.5, CELL_SILE=1 / 7, device='cpu'):
        super(YoloV1Loss, self).__init__()
        self.lambda_coord = lambda_coord
        self.lambda_noobj = lambda_noobj
        self.CELL_SILE = CELL_SILE
        self.device = device

    def forward(self, predictions, targets):
        coord_mask = targets[:, :, :, 4] == 1
        noobj_mask = targets[:, :, :, 4] == 0

        confidence_loss_noobj = nn.functional.mse_loss(predictions[noobj_mask].reshape(-1, 30)[:, [4, 9]],
                                                       targets[noobj_mask].reshape(-1, 30)[:, [4, 9]],
                                                       size_average=False, reduction='sum') * self.lambda_noobj

        have_obj_mask = torch.unsqueeze(coord_mask, dim=-1).expand_as(targets)
        targets_coord = targets[have_obj_mask].reshape(-1, 30)
        preds_coord = predictions[have_obj_mask].reshape(-1, 30)

        coord_mask = torch.zeros(preds_coord.size(), dtype=torch.bool, device=self.device)
        coord_confidence = torch.zeros(preds_coord.size(), dtype=torch.bool, device=self.device)
        classic_mask = torch.zeros(preds_coord.size(), dtype=torch.bool, device=self.device)

        for (pred_coord, target_coord, mask, confidence, class_mask) in zip(preds_coord, targets_coord,
                                                                            coord_mask, coord_confidence, classic_mask
                                                                            ):
            pred_box = torch.hstack((pred_coord[:4], pred_coord[5:9])).reshape(-1, 4).clone()
            pred_box[:, 2:] = torch.square(pred_box[:, 2:])
            target_box = target_coord[:4].reshape(-1, 4).clone()
            target_box[:, 2:] = torch.square(target_box[:, 2:])
            pred_box[:, :2] = pred_box[:, :2] * self.CELL_SILE - pred_box[:, 2:4]/2
            pred_box[:, 2:] = pred_box[:, :2] * self.CELL_SILE + pred_box[:, 2:4]/2
            target_box[:, :2] = target_box[:, :2] * self.CELL_SILE - target_box[:, 2:4]/2
            target_box[:, 2:] = target_box[:, :2] * self.CELL_SILE + target_box[:, 2:4]/2
            iou = torchvision.ops.box_iou(pred_box, target_box)

            per_prebox_max_iou, per_prebox_max_iou_index = torch.max(iou, dim=0)
            obj_index = torch.argmax(target_coord[10:], dim=0)
            class_mask[10 + obj_index] = True
            mask[per_prebox_max_iou_index * 5: per_prebox_max_iou_index * 5 + 4] = True
            confidence[per_prebox_max_iou_index * 5 + 4] = True
            # target_coord[per_prebox_max_iou_index * 5] = per_prebox_max_iou.data

        calculate_boxes = preds_coord[coord_mask].reshape(-1, 4)
        calculate_boxes_target = targets_coord[coord_mask].reshape(-1, 4)

        coordence_center_loss = nn.functional.mse_loss(calculate_boxes[:, :2], calculate_boxes_target[:, :2],
                                                       size_average=False, reduction='sum') * self.lambda_coord

        coordence_wh_loss = nn.functional.mse_loss(calculate_boxes[:, 2:4], calculate_boxes_target[:, 2:4],
                                                   size_average=False, reduction='sum') * self.lambda_coord

        confidence_loss = nn.functional.mse_loss(preds_coord[coord_confidence], targets_coord[coord_confidence],
                                                 size_average=False, reduction='sum') * self.lambda_coord
        classier_loss = nn.functional.mse_loss(preds_coord[classic_mask], targets_coord[classic_mask],
                                               size_average=False, reduction='sum') * self.lambda_coord

        total_loss = confidence_loss_noobj + coordence_center_loss + coordence_wh_loss + confidence_loss + classier_loss
        return classier_loss, confidence_loss, coordence_center_loss + coordence_wh_loss, total_loss


if __name__ == '__main__':
    import torch_directml

    loss_function = YoloV1Loss(lambda_coord=5., lambda_noobj=.5, CELL_SILE=1 / 7)
    if torch_directml.is_available():
        device = torch_directml.device(0)
        loss_function.device = device
    else:
        device = 'cpu'

    loss_function = loss_function.to(device)
    pre_mat = torch.rand((7, 7, 30), dtype=torch.float, device=device)
    pre_mat[0, 0, [4, 9, 19]] = 1
    pre_mat[2, 2, [4, 9, 23]] = 1
    target_mat = torch.zeros((7, 7, 30), dtype=torch.float, device=device)
    target_mat[0, 0, [4, 9, 20]] = 1
    target_mat[1, 2, [4, 9, 23]] = 1
    print(loss_function(pre_mat.unsqueeze(0), target_mat.unsqueeze(0)))

# mat1 = torch.rand((1, 3))
#
# cer = nn.MSELoss()
# mat2 = torch.ones((1, 3))
# mat3 = mat2 - mat1
# print(mat3)
# print(cer((mat1, mat1)))
