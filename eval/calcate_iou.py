import torch


def calc_iou(boxes1, boxes2):
    N = boxes1.shape[0]
    M = boxes2.shape[0]

    boxes1_bigger_e = boxes1[:, :2].unsqueeze(1).expand(N, M, 2)
    boxes2_bigger_e = boxes2[:, :2].unsqueeze(0).expand(N, M, 2)
    left_coords = torch.max(boxes1_bigger_e, boxes2_bigger_e)

    boxes1_bigger_e = boxes1[:, 2:].unsqueeze(1).expand(N, M, 2)
    boxes2_bigger_e = boxes2[:, 2:].unsqueeze(0).expand(N, M, 2)
    right_coords = torch.min(boxes1_bigger_e, boxes2_bigger_e)

    collections_area_wh = (right_coords - left_coords)
    collections_area_wh[collections_area_wh < 0] = 0
    collections_area = collections_area_wh[0] * collections_area_wh[1]

    boxes1_area = (boxes1[:, 2] - boxes1[:, 0]) * (boxes1[:, 3] - boxes1[:, 1]).unsqueeze(1).expand(N, M)
    boxes2_area = (boxes2[:, 2] - boxes2[:, 0]) * (boxes2[:, 3] - boxes2[:, 1]).unsqueeze(0).expand(N, M)

    return collections_area / (boxes1_area + boxes2_area - collections_area)
