# ------------------------------------------------------------------------------
# Copyright (c) Microsoft
# Licensed under the MIT License.
# Written by Bin Xiao (Bin.Xiao@microsoft.com)
# ------------------------------------------------------------------------------

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from collections import defaultdict
from collections import OrderedDict
import logging
import os

from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval
import json_tricks as json
import numpy as np

from dataset.HandKeypointsDataset import HandKeypointsDataset
# from dataset.JointsDataset import JointsDataset

logger = logging.getLogger(__name__)


class MyDataset(HandKeypointsDataset):
    '''
    "keypoints": {
        0: "nose",
        1: "left_eye",
        2: "right_eye",
        3: "left_ear",
        4: "right_ear",
        5: "left_shoulder",
    },
	"skeleton": [
        [16,14],[14,12],[17,15],[15,13],[12,13],[6,12],[7,13], [6,7],[6,8],
        [7,9],[8,10],[9,11],[2,3],[1,2],[1,3],[2,4],[3,5],[4,6],[5,7]]
    '''
    def __init__(self, cfg, root, image_set, is_train, transform=None):
        super().__init__(cfg, root, image_set, is_train, transform)
        self.nms_thre = cfg.TEST.NMS_THRE
        self.image_thre = cfg.TEST.IMAGE_THRE
        self.soft_nms = cfg.TEST.SOFT_NMS
        self.oks_thre = cfg.TEST.OKS_THRE
        self.in_vis_thre = cfg.TEST.IN_VIS_THRE
        self.bbox_file = cfg.TEST.COCO_BBOX_FILE
        self.use_gt_bbox = cfg.TEST.USE_GT_BBOX
        self.image_width = cfg.MODEL.IMAGE_SIZE[0]
        self.image_height = cfg.MODEL.IMAGE_SIZE[1]
        self.aspect_ratio = self.image_width * 1.0 / self.image_height
        self.pixel_std = 200
        # 添加上採樣相關配置
        self.upsample_factor = cfg.DATASET.get('UPSAMPLE_FACTOR', 1)  # 默認為1，不進行上採樣
        
        # 讀取左右手的json file
        anno_path = os.path.join(self.root, 'annotations', cfg.DATASET.HAND_ORIENTATION+'_hand_'+self.image_set+'.json')
        self.coco = COCO(anno_path)
        # self.coco = COCO(self._get_ann_file_keypoint())
        

        # deal with class names
        cats = [cat['name']
                for cat in self.coco.loadCats(self.coco.getCatIds())]
        self.classes = ['__background__'] + cats
        logger.info('=> classes: {}'.format(self.classes))
        self.num_classes = len(self.classes)
        self._class_to_ind = dict(zip(self.classes, range(self.num_classes)))
        self._class_to_coco_ind = dict(zip(cats, self.coco.getCatIds()))
        self._coco_ind_to_class_ind = dict(
            [
                (self._class_to_coco_ind[cls], self._class_to_ind[cls])
                for cls in self.classes[1:]
            ]
        )

        # load image file names
        self.image_set_index = self._load_image_set_index()
        self.num_images = len(self.image_set_index)
        logger.info('=> num_images: {}'.format(self.num_images))

        self.num_joints = 6
        #self.flip_pairs = [[1, 2], [3, 4], [5, 6], [7, 8], [9, 10], [11, 12], [13, 14], [15, 16]]
        self.parent_ids = None
        #self.upper_body_ids = (0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10)
        #self.lower_body_ids = (11, 12, 13, 14, 15, 16)

        self.joints_weight = np.array(
            [
                1., 1., 1., 1., 1., 1.
            ],
            dtype=np.float32
        ).reshape((self.num_joints, 1))

        self.db = self._get_db()

         # 進行數據上採樣
        if is_train and self.upsample_factor > 1:
            self.db = self._upsample_data(self.db)

        if is_train and cfg.DATASET.SELECT_DATA:
            self.db = self.select_data(self.db)


        logger.info('=> load {} samples'.format(len(self.db)))

    def _get_ann_file_keypoint(self):
        """ self.root / annotations / person_keypoints_train2017.json """
        prefix = 'person_keypoints' \
            if 'test' not in self.image_set else 'image_info'
        return os.path.join(
            self.root,
            'annotations',
            prefix + '_' + self.image_set + '.json'
        )

    def _load_image_set_index(self):
        """ image id: int """
        image_ids = self.coco.getImgIds()
        return image_ids

    def _get_db(self):
        if self.is_train or self.use_gt_bbox:
            # use ground truth bbox
            gt_db = self._load_coco_keypoint_annotations()
        else:
            # use bbox from detection
            gt_db = self._load_coco_person_detection_results()
        return gt_db

    def _load_coco_keypoint_annotations(self):
        """ ground truth bbox and keypoints """
        gt_db = []
        for index in self.image_set_index:
            gt_db.extend(self._load_coco_keypoint_annotation_kernal(index))
        return gt_db
    
    def _upsample_data(self, db):
        """
        對數據進行上採樣
        """
        logger.info(f'=> Upsampling data by factor {self.upsample_factor}')
        original_len = len(db)
        upsampled_db = []
        
        for _ in range(self.upsample_factor):
            for item in db:
                # 創建新的數據項
                new_item = item.copy()                
                upsampled_db.append(new_item)
        
        logger.info(f'=> Upsampled from {original_len} to {len(upsampled_db)} samples')
        return upsampled_db
    

    def _load_coco_keypoint_annotation_kernal(self, index):
        """
        coco ann: [u'segmentation', u'area', u'iscrowd', u'image_id', u'bbox', u'category_id', u'id']
        iscrowd:
            crowd instances are handled by marking their overlaps with all categories to -1
            and later excluded in training
        bbox:
            [x1, y1, w, h]
        :param index: coco image id
        :return: db entry
        """
        im_ann = self.coco.loadImgs(index)[0]
        width = im_ann['width']
        height = im_ann['height']

        annIds = self.coco.getAnnIds(imgIds=index, iscrowd=False)
        objs = self.coco.loadAnns(annIds)

        # sanitize bboxes
        valid_objs = []
        for obj in objs:
            x, y, w, h = obj['bbox']
            x1 = np.max((0, x))
            y1 = np.max((0, y))
            x2 = np.min((width - 1, x1 + np.max((0, w - 1))))
            y2 = np.min((height - 1, y1 + np.max((0, h - 1))))
            # if obj['area'] > 0 and x2 >= x1 and y2 >= y1:
            if x2 >= x1 and y2 >= y1:
                obj['clean_bbox'] = [x1, y1, x2-x1, y2-y1]
                valid_objs.append(obj)
        objs = valid_objs

        rec = []
        for obj in objs:
            cls = self._coco_ind_to_class_ind[obj['category_id']]
            if cls != 1:
                continue

            # ignore objs without keypoints annotation
            if max(obj['keypoints']) == 0:
                continue

            joints_3d = np.zeros((self.num_joints, 3), dtype=np.float64)
            joints_3d_vis = np.zeros((self.num_joints, 3), dtype=np.float64)
            for ipt in range(self.num_joints):
                joints_3d[ipt, 0] = obj['keypoints'][ipt * 3 + 0]
                joints_3d[ipt, 1] = obj['keypoints'][ipt * 3 + 1]
                joints_3d[ipt, 2] = 0
                t_vis = obj['keypoints'][ipt * 3 + 2]
                if t_vis > 1:
                    t_vis = 1
                joints_3d_vis[ipt, 0] = t_vis
                joints_3d_vis[ipt, 1] = t_vis
                joints_3d_vis[ipt, 2] = 0

            center, scale = self._box2cs(obj['clean_bbox'][:4])

            image_path = os.path.join(
                self.root, 'images', self.image_set, im_ann['file_name'])
            rec.append({
                'image': image_path,
                'center': center,
                'scale': scale,
                'joints_3d': joints_3d,
                'joints_3d_vis': joints_3d_vis,
                'filename': '',
                'imgnum': 0,
                'x': x,
                'y': y,
                'w': w,
                'h': h,
            })

        return rec

    def _box2cs(self, box):
        x, y, w, h = box[:4]
        return self._xywh2cs(x, y, w, h)

    def _xywh2cs(self, x, y, w, h):
        center = np.zeros((2), dtype=np.float32)
        center[0] = x + w * 0.5
        center[1] = y + h * 0.5
        
        if w > self.aspect_ratio * h:
            h = w * 1.0 / self.aspect_ratio
        elif w < self.aspect_ratio * h:
            w = h * self.aspect_ratio
        scale = np.array(
            [w * 1.0 / self.pixel_std, h * 1.0 / self.pixel_std],
            dtype=np.float32)
        if center[0] != -1:
            scale = scale * 1.25 # 1.25 is slightly larger than heatmap gaussian sigma

        return center, scale

    def image_path_from_index(self, index):
        """ example: images / train2017 / 000000119993.jpg """
        file_name = '%012d.jpg' % index
        if '2014' in self.image_set:
            file_name = 'COCO_%s_' % self.image_set + file_name

        prefix = 'test2017' if 'test' in self.image_set else self.image_set

        data_name = prefix + '.zip@' if self.data_format == 'zip' else prefix

        image_path = os.path.join(
            self.root, 'images', data_name, file_name)

        return image_path

    def _load_coco_person_detection_results(self):
        all_boxes = None
        with open(self.bbox_file, 'r') as f:
            all_boxes = json.load(f)

        if not all_boxes:
            logger.error('=> Load %s fail!' % self.bbox_file)
            return None

        logger.info('=> Total boxes: {}'.format(len(all_boxes)))

        kpt_db = []
        num_boxes = 0
        for n_img in range(0, len(all_boxes)):
            det_res = all_boxes[n_img]
            if det_res['category_id'] != 1:
                continue
            img_name = self.image_path_from_index(det_res['image_id'])
            box = det_res['bbox']
            score = det_res['score']

            if score < self.image_thre:
                continue

            num_boxes = num_boxes + 1

            center, scale = self._box2cs(box)
            joints_3d = np.zeros((self.num_joints, 3), dtype=np.float64)
            joints_3d_vis = np.ones(
                (self.num_joints, 3), dtype=np.float64)
            kpt_db.append({
                'image': img_name,
                'center': center,
                'scale': scale,
                'score': score,
                'joints_3d': joints_3d,
                'joints_3d_vis': joints_3d_vis,
            })

        logger.info('=> Total boxes after fliter low score@{}: {}'.format(
            self.image_thre, num_boxes))
        return kpt_db

    def evaluate(self, cfg, preds, output_dir, all_boxes, img_path,
             *args, **kwargs):
        """
        Evaluate the keypoint detection results using pixel distance as metric
        
        Args:
            cfg: config object
            preds: list of predictions, each prediction is a list of keypoints
            output_dir: output directory for saving results
            all_boxes: bounding box information
            img_path: list of image paths
        
        Returns:
            An OrderedDict containing evaluation metrics
        """
        logger.info('=> Evaluating using pixel distance metric')
        
        # Create output directory if it doesn't exist
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
        
        # Convert predictions to standard format
        num_samples = len(all_boxes)
        all_preds = np.zeros((num_samples, self.num_joints, 3), dtype=np.float32)
        all_boxes = np.zeros((num_samples, 6), dtype=np.float32)
        image_ids = []
        idx = 0
        
        for i in range(len(preds)):
            # Get image id
            image_id = self.image_set_index[i]
            image_ids.append(image_id)
            
            # Get keypoints
            kpt_pred = preds[i]
            box_pred = all_boxes[i]
            
            # Record predictions
            all_preds[idx] = kpt_pred
            all_boxes[idx] = box_pred
            idx += 1
        
        # Get ground truth keypoints
        gt_db = self._get_db()
        gt_keypoints = np.zeros((num_samples, self.num_joints, 3), dtype=np.float32)
        
        # Match predictions with ground truth based on image_id
        for i, image_id in enumerate(image_ids):
            gt_keypoints_for_image = None
            for item in gt_db:
                if self.coco.loadImgs(image_id)[0]['file_name'] in item['image']:
                    gt_keypoints_for_image = item['joints_3d']
                    break
            
            if gt_keypoints_for_image is not None:
                gt_keypoints[i] = gt_keypoints_for_image
        
        # Calculate pixel distance for each keypoint
        distances = []
        keypoint_distances = [[] for _ in range(self.num_joints)]  # 為每個關鍵點創建一個距離列表
        valid_keypoints = 0
        total_keypoints = 0
        keypoint_valid_counts = np.zeros(self.num_joints, dtype=np.int32)
        
        # Store per-image results
        per_image_results = []
        
        for i in range(num_samples):
            image_id = image_ids[i]
            image_name = self.coco.loadImgs(image_id)[0]['file_name']
            
            # Store results for this image
            image_result = {
                'image_id': int(image_id),
                'image_name': image_name,
                'keypoints': []
            }
            
            image_valid_keypoints = 0
            image_total_keypoints = 0
            image_distances = []
            image_keypoint_distances = [[] for _ in range(self.num_joints)]
            
            for j in range(self.num_joints):
                # Check if both prediction and ground truth are valid
                pred_vis = all_preds[i, j, 2] > 0
                gt_vis = gt_keypoints[i, j, 0] > 0 and gt_keypoints[i, j, 1] > 0
                
                total_keypoints += 1
                image_total_keypoints += 1
                
                # Store keypoint result
                keypoint_result = {
                    'keypoint_id': j,
                    'keypoint_name': self._get_keypoint_name(j),
                    'predicted': [float(all_preds[i, j, 0]), float(all_preds[i, j, 1])],
                    'ground_truth': [float(gt_keypoints[i, j, 0]), float(gt_keypoints[i, j, 1])],
                    'visibility': {
                        'prediction': bool(pred_vis),
                        'ground_truth': bool(gt_vis)
                    }
                }
                
                if pred_vis and gt_vis:
                    valid_keypoints += 1
                    image_valid_keypoints += 1
                    keypoint_valid_counts[j] += 1
                    
                    # Calculate Euclidean distance
                    pred_x, pred_y = all_preds[i, j, 0], all_preds[i, j, 1]
                    gt_x, gt_y = gt_keypoints[i, j, 0], gt_keypoints[i, j, 1]
                    
                    dist = np.sqrt((pred_x - gt_x)**2 + (pred_y - gt_y)**2)
                    distances.append(dist)
                    keypoint_distances[j].append(dist)  # 將距離添加到對應關鍵點的列表中
                    image_distances.append(dist)
                    image_keypoint_distances[j].append(dist)
                    
                    # Add distance to keypoint result
                    keypoint_result['distance'] = float(dist)
                else:
                    keypoint_result['distance'] = None
                
                image_result['keypoints'].append(keypoint_result)
            
            # Add image-level statistics
            if image_distances:
                image_result['statistics'] = {
                    'mean_distance': float(np.mean(image_distances)),
                    'median_distance': float(np.median(image_distances)),
                    'max_distance': float(np.max(image_distances)),
                    'min_distance': float(np.min(image_distances)),
                    'valid_keypoints': int(image_valid_keypoints),
                    'total_keypoints': int(image_total_keypoints),
                    'valid_percentage': float(image_valid_keypoints / image_total_keypoints * 100)
                }
                
                # 添加每個關鍵點的距離統計
                image_result['keypoint_statistics'] = {}
                for j in range(self.num_joints):
                    if image_keypoint_distances[j]:
                        image_result['keypoint_statistics'][self._get_keypoint_name(j)] = {
                            'mean_distance': float(np.mean(image_keypoint_distances[j])),
                            'median_distance': float(np.median(image_keypoint_distances[j])),
                            'max_distance': float(np.max(image_keypoint_distances[j])),
                            'min_distance': float(np.min(image_keypoint_distances[j])),
                            'count': len(image_keypoint_distances[j])
                        }
            else:
                image_result['statistics'] = {
                    'mean_distance': None,
                    'median_distance': None,
                    'max_distance': None,
                    'min_distance': None,
                    'valid_keypoints': int(image_valid_keypoints),
                    'total_keypoints': int(image_total_keypoints),
                    'valid_percentage': 0.0
                }
                image_result['keypoint_statistics'] = {}
            
            per_image_results.append(image_result)
        
        # Calculate overall metrics
        mean_dist = np.mean(distances) if distances else float('inf')
        median_dist = np.median(distances) if distances else float('inf')
        max_dist = np.max(distances) if distances else float('inf')
        min_dist = np.min(distances) if distances else float('inf')
        std_dist = np.std(distances) if distances else float('inf')
        
        # 計算每個關鍵點的距離統計
        keypoint_stats = {}
        for j in range(self.num_joints):
            if keypoint_distances[j]:
                keypoint_stats[self._get_keypoint_name(j)] = {
                    'mean_distance': float(np.mean(keypoint_distances[j])),
                    'median_distance': float(np.median(keypoint_distances[j])),
                    'max_distance': float(np.max(keypoint_distances[j])),
                    'min_distance': float(np.min(keypoint_distances[j])),
                    'std_distance': float(np.std(keypoint_distances[j])),
                    'count': int(keypoint_valid_counts[j]),
                    'valid_percentage': float(keypoint_valid_counts[j] / num_samples * 100)
                }
            else:
                keypoint_stats[self._get_keypoint_name(j)] = {
                    'mean_distance': float('inf'),
                    'median_distance': float('inf'),
                    'max_distance': float('inf'),
                    'min_distance': float('inf'),
                    'std_distance': float('inf'),
                    'count': 0,
                    'valid_percentage': 0.0
                }
        
        # Print results
        logger.info('=> Mean pixel distance: {:.2f}'.format(mean_dist))
        logger.info('=> Median pixel distance: {:.2f}'.format(median_dist))
        logger.info('=> Max pixel distance: {:.2f}'.format(max_dist))
        logger.info('=> Min pixel distance: {:.2f}'.format(min_dist))
        logger.info('=> Std pixel distance: {:.2f}'.format(std_dist))
        logger.info('=> Valid keypoints: {}/{} ({:.2f}%)'.format(
            valid_keypoints, total_keypoints, 
            valid_keypoints / total_keypoints * 100 if total_keypoints > 0 else 0
        ))
        
        # Print distance statistics for each keypoint
        for j in range(self.num_joints):
            kpt_name = self._get_keypoint_name(j)
            if keypoint_distances[j]:
                logger.info('=> {} mean distance: {:.2f}'.format(
                    kpt_name, keypoint_stats[kpt_name]['mean_distance']))
                logger.info('=> {} median distance: {:.2f}'.format(
                    kpt_name, keypoint_stats[kpt_name]['median_distance']))
        
        # Save overall results to file
        result_file = os.path.join(output_dir, 'pixel_distance_results.json')
        with open(result_file, 'w') as f:
            json.dump({
                'mean_distance': float(mean_dist),
                'median_distance': float(median_dist),
                'max_distance': float(max_dist),
                'min_distance': float(min_dist),
                'std_distance': float(std_dist),
                'valid_keypoints': int(valid_keypoints),
                'total_keypoints': int(total_keypoints),
                'valid_percentage': float(valid_keypoints / total_keypoints * 100 if total_keypoints > 0 else 0),
                'keypoint_statistics': keypoint_stats
            }, f, indent=4)
        
        # Save per-image results to file
        per_image_file = os.path.join(output_dir, 'per_image_results.json')
        with open(per_image_file, 'w') as f:
            json.dump(per_image_results, f, indent=4)
        
        logger.info('=> Evaluation results saved to {}'.format(result_file))
        logger.info('=> Per-image results saved to {}'.format(per_image_file))
        
        # Return metrics in the specified format
        name_value = [
            ('E0_mean_dist', float(keypoint_stats['E0']['mean_distance'])),
            ('E1_mean_dist', float(keypoint_stats['E1']['mean_distance'])),
            ('E2_mean_dist', float(keypoint_stats['E2']['mean_distance'])),
            ('W0_mean_dist', float(keypoint_stats['W0']['mean_distance'])),
            ('W1_mean_dist', float(keypoint_stats['W1']['mean_distance'])),
            ('W2_mean_dist', float(keypoint_stats['W2']['mean_distance'])),
            ('Mean_distance', float(mean_dist)),
            ('Median_distance', float(median_dist))
        ]
        name_value = OrderedDict(name_value)
        
        return name_value, name_value["Mean_distance"]

    def _get_keypoint_name(self, keypoint_id):
        """
        Get the name of a keypoint based on its ID
        """
        keypoint_names = {
            0: "E0",
            1: "E1",
            2: "E2",
            3: "W0",
            4: "W1",
            5: "W2"
        }
        return keypoint_names.get(keypoint_id, f"keypoint_{keypoint_id}")

    def _write_coco_keypoint_results(self, keypoints, res_file):
        data_pack = [
            {
                'cat_id': self._class_to_coco_ind[cls],
                'cls_ind': cls_ind,
                'cls': cls,
                'ann_type': 'keypoints',
                'keypoints': keypoints
            }
            for cls_ind, cls in enumerate(self.classes) if not cls == '__background__'
        ]

        results = self._coco_keypoint_results_one_category_kernel(data_pack[0])
        logger.info('=> writing results json to %s' % res_file)
        with open(res_file, 'w') as f:
            json.dump(results, f, sort_keys=True, indent=4)
        try:
            json.load(open(res_file))
        except Exception:
            content = []
            with open(res_file, 'r') as f:
                for line in f:
                    content.append(line)
            content[-1] = ']'
            with open(res_file, 'w') as f:
                for c in content:
                    f.write(c)

    def _coco_keypoint_results_one_category_kernel(self, data_pack):
        cat_id = data_pack['cat_id']
        keypoints = data_pack['keypoints']
        cat_results = []

        for img_kpts in keypoints:
            if len(img_kpts) == 0:
                continue

            _key_points = np.array([img_kpts[k]['keypoints']
                                    for k in range(len(img_kpts))])
            key_points = np.zeros(
                (_key_points.shape[0], self.num_joints * 3), dtype=np.float64
            )

            for ipt in range(self.num_joints):
                key_points[:, ipt * 3 + 0] = _key_points[:, ipt, 0]
                key_points[:, ipt * 3 + 1] = _key_points[:, ipt, 1]
                key_points[:, ipt * 3 + 2] = _key_points[:, ipt, 2]  # keypoints score.

            result = [
                {
                    'image_id': img_kpts[k]['image'],
                    'category_id': cat_id,
                    'keypoints': list(key_points[k]),
                    'score': img_kpts[k]['score'],
                    'center': list(img_kpts[k]['center']),
                    'scale': list(img_kpts[k]['scale'])
                }
                for k in range(len(img_kpts))
            ]
            cat_results.extend(result)

        return cat_results

    def _do_python_keypoint_eval(self, res_file, res_folder):
        coco_dt = self.coco.loadRes(res_file)
        coco_eval = COCOeval(self.coco, coco_dt, 'keypoints')
        coco_eval.params.useSegm = None
        coco_eval.evaluate()
        coco_eval.accumulate()
        coco_eval.summarize()

        stats_names = ['AP', 'Ap .5', 'AP .75', 'AP (M)', 'AP (L)', 'AR', 'AR .5', 'AR .75', 'AR (M)', 'AR (L)']

        info_str = []
        for ind, name in enumerate(stats_names):
            info_str.append((name, coco_eval.stats[ind]))

        return info_str
