import torch.nn as nn
from mmdet.apis import init_detector, inference_detector
import torch

class MMDetCustomModel(nn.Module):
    def __init__(self, model_cfg):
        super().__init__()
        image_detector_config_file = model_cfg.IMAGE_DETECTOR_CONFIG_FILE
        image_detector_weight_file = model_cfg.IMAGE_DETECTOR_WEIGHTS_FILE
        self.image_model = init_detector(image_detector_config_file, image_detector_weight_file, device='cuda:0')
    def forward(self, data_dict):
        #* 图像
        images = data_dict['mmcv_images']
        images_results = []
        for image in images:
            image_result = inference_detector(self.image_model, image)
            image_result_bboxes = image_result.pred_instances.bboxes
            image_result_scores = image_result.pred_instances.scores.unsqueeze(-1)
            image_result_labels = image_result.pred_instances.labels
            cur_image_result = torch.cat([image_result_bboxes, image_result_scores], dim=-1).unsqueeze(0)
            images_results.append(cur_image_result)
        boxes2d_by_detector = torch.cat(images_results,dim=0)
        data_dict["results_2d"] = boxes2d_by_detector
        return data_dict