import torch.nn as nn
import torch
import numpy as np


class MMDetCustomModel(nn.Module):
    def __init__(self, model_cfg):
        super().__init__()
        self.model_cfg = model_cfg
        image_detector_config_file = model_cfg.IMAGE_DETECTOR_CONFIG_FILE
        image_detector_weight_file = model_cfg.IMAGE_DETECTOR_WEIGHTS_FILE
        self.use_gt = model_cfg.get("USE_GT", False)
        if self.use_gt:
            self.val_activated = model_cfg.VAL_ACTIVATED
            self.lower_score = model_cfg.LOWER_SCORE
            self.higher_score = model_cfg.HIGHER_SCORE
            self.propotion = model_cfg.PROPOTION

        deploy_cfg = model_cfg.get("DEPLOY_CFG", None)
        if deploy_cfg is not None:
            from mmdeploy.utils import get_input_shape, load_config
            from mmdeploy.apis.utils import build_task_processor
            self.speed_up = True
            detector_2d_cfg = model_cfg.IMAGE_DETECTOR_CONFIG_FILE
            device = 'cuda'
            backend_model = model_cfg.BACKEND_MODEL
            deploy_cfg, detector_2d_cfg = load_config(deploy_cfg, detector_2d_cfg)
            self.task_processor = build_task_processor(detector_2d_cfg, deploy_cfg, device)
            self.model = self.task_processor.build_backend_model(backend_model)
            self.input_shape = get_input_shape(deploy_cfg)
        else:
            from mmdet.apis import init_detector
            self.image_model = init_detector(image_detector_config_file, image_detector_weight_file, device='cuda:0')
            self.speed_up = False
            self._setup_mmdet_3x_preprocessor()

    def _setup_mmdet_3x_preprocessor(self):
        """Set up proper DetDataPreprocessor for mmdet 3.x models."""
        import mmdet
        if int(mmdet.__version__.split('.')[0]) >= 3:
            from mmdet.models.data_preprocessors import DetDataPreprocessor
            if not isinstance(self.image_model.data_preprocessor, DetDataPreprocessor):
                self.image_model.data_preprocessor = DetDataPreprocessor(
                    mean=[123.675, 116.28, 103.53],
                    std=[58.395, 57.12, 57.375],
                    bgr_to_rgb=True,
                ).cuda()

    def forward(self, data_dict):
        images = data_dict['images']

        if self.use_gt:
            if self.training or self.val_activated:
                gt_boxes2d_num = data_dict["gt_boxes2d"].shape[1]
                gt_boxes2d_scores = ((self.higher_score - self.lower_score) * torch.rand(1, gt_boxes2d_num, 1) + self.lower_score).cuda()
                random_indices = torch.randperm(gt_boxes2d_num)
                selected_indices = random_indices[:int(gt_boxes2d_num * self.propotion)]
                gt_boxes2d = data_dict["gt_boxes2d"][:, selected_indices, :]
                gt_boxes2d_scores = gt_boxes2d_scores[:, selected_indices, :]
                gt_boxes2d = torch.cat([gt_boxes2d, gt_boxes2d_scores], dim=-1)

        import mmdet
        _mmdet_major = int(mmdet.__version__.split('.')[0])

        if _mmdet_major >= 3:
            images_results = self._forward_mmdet3x(images)
        else:
            images_results = self._forward_mmdet2x(images)

        boxes2d_by_detector = torch.cat(images_results, dim=0)

        if self.use_gt:
            if self.training or self.val_activated:
                if boxes2d_by_detector.shape[1] + gt_boxes2d_num > 100:
                    drop_num = boxes2d_by_detector.shape[1] + gt_boxes2d_num - 100
                    boxes2d_by_detector = boxes2d_by_detector[:, :-drop_num, :]
                    boxes2d_by_detector = torch.cat([boxes2d_by_detector, gt_boxes2d], dim=1)
                else:
                    boxes2d_by_detector = torch.cat([boxes2d_by_detector, gt_boxes2d], dim=1)
        data_dict["results_2d"] = boxes2d_by_detector.cuda()
        return data_dict

    def _forward_mmdet3x(self, images):
        """Handle mmdet 3.x batched inference."""
        from mmdet.utils import get_test_pipeline_cfg
        from mmcv.transforms import Compose

        cfg = self.image_model.cfg.copy()
        test_pipeline_cfg = get_test_pipeline_cfg(cfg)
        if isinstance(images[0], np.ndarray):
            test_pipeline_cfg[0].type = 'mmdet.LoadImageFromNDArray'
        test_pipeline = Compose(test_pipeline_cfg)

        all_inputs = []
        all_data_samples = []
        for image in images:
            if isinstance(image, np.ndarray):
                data_ = dict(img=image, img_id=0)
            else:
                data_ = dict(img_path=image, img_id=0)
            data_ = test_pipeline(data_)
            all_inputs.append(data_['inputs'])
            all_data_samples.append(data_['data_samples'])

        # Stack inputs and run through data_preprocessor (which normalizes)
        batch_data = dict(inputs=[t.cuda() for t in all_inputs], data_samples=all_data_samples)
        batch_data = self.image_model.data_preprocessor(batch_data, False)
        with torch.no_grad():
            results = self.image_model.predict(batch_data['inputs'], batch_data['data_samples'])

        images_results = []
        for result in results:
            if hasattr(result, 'pred_instances') and len(result.pred_instances.bboxes) > 0:
                bboxes = result.pred_instances.bboxes
                scores = result.pred_instances.scores.unsqueeze(-1)
                labels = result.pred_instances.labels
                valid_mask = (labels == 0)
                bboxes = bboxes[valid_mask]
                scores = scores[valid_mask]
                if bboxes.shape[0] > 0:
                    cur_result = torch.cat([bboxes, scores], dim=-1).unsqueeze(0)
                else:
                    cur_result = torch.zeros(1, 0, 5, device='cuda')
            else:
                cur_result = torch.zeros(1, 0, 5, device='cuda')

            # Pad/truncate to exactly 100 boxes
            cur_num = cur_result.shape[1]
            if cur_num < 100:
                added = torch.zeros([1, 100 - cur_num, 5], device=cur_result.device, dtype=cur_result.dtype)
                cur_result = torch.cat([cur_result, added], dim=1)
            elif cur_num > 100:
                cur_result = cur_result[:, :100, :]
            images_results.append(cur_result)

        return images_results

    def _forward_mmdet2x(self, images):
        """Handle mmdet 2.x inference."""
        from mmdet.apis import inference_detector

        images_results = []
        for image in images:
            if self.speed_up:
                model_inputs, _ = self.task_processor.create_input(image, self.input_shape)
                with torch.no_grad():
                    image_result = self.model.test_step(model_inputs)[0]
                bboxes = image_result.pred_instances.bboxes
                scores = image_result.pred_instances.scores.unsqueeze(-1)
                labels = image_result.pred_instances.labels
            else:
                image_result = inference_detector(self.image_model, image)
                if isinstance(image_result, list):
                    # mmdet 2.x format: list of per-class arrays
                    result_array = image_result[0][0]
                    if result_array.ndim == 1:
                        result_array = result_array.reshape(-1, 5)
                    cur_result = torch.from_numpy(result_array).cuda().float().unsqueeze(0)
                    # Pad/truncate
                    cur_num = cur_result.shape[1]
                    if cur_num < 100:
                        added = torch.zeros([1, 100 - cur_num, 5], device=cur_result.device, dtype=cur_result.dtype)
                        cur_result = torch.cat([cur_result, added], dim=1)
                    elif cur_num > 100:
                        cur_result = cur_result[:, :100, :]
                    images_results.append(cur_result)
                    continue
                else:
                    bboxes = image_result.pred_instances.bboxes
                    scores = image_result.pred_instances.scores.unsqueeze(-1)
                    labels = image_result.pred_instances.labels

            valid_mask = (labels == 0)
            bboxes = bboxes[valid_mask]
            scores = scores[valid_mask]
            if bboxes.shape[0] > 0:
                cur_result = torch.cat([bboxes, scores], dim=-1).unsqueeze(0)
            else:
                cur_result = torch.zeros(1, 0, 5, device='cuda')

            cur_num = cur_result.shape[1]
            if cur_num < 100:
                added = torch.zeros([1, 100 - cur_num, 5], device=cur_result.device, dtype=cur_result.dtype)
                cur_result = torch.cat([cur_result, added], dim=1)
            elif cur_num > 100:
                cur_result = cur_result[:, :100, :]
            images_results.append(cur_result)

        return images_results
