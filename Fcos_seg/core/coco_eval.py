import json
import tempfile
from collections import OrderedDict

import numpy as np
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval


def coco_evaluate(dataset, predictions):
    coco_results = {}
    coco_results['bbox'] = make_coco_detection(predictions, dataset)

    results = COCOResult('bbox')

    with tempfile.NamedTemporaryFile() as f:
        path = f.name
        res = evaluate_predictions_on_coco(
            dataset.coco, coco_results['bbox'], path, 'bbox'
        )
        results.update(res)
    

    return res


def evaluate_predictions_on_coco(coco_gt, results, result_file, iou_type):
    with open(result_file, 'w') as f:
        json.dump(results, f)

    coco_dt = coco_gt.loadRes(str(result_file)) if results else COCO()

    coco_eval = COCOeval(coco_gt, coco_dt, iou_type)
    coco_eval.evaluate()
    coco_eval.accumulate()
    coco_eval.summarize()

    
    return coco_eval



def make_coco_detection(predictions, dataset):
    coco_results = []

    for image_id, pred in enumerate(predictions):
        orig_id = dataset.id_to_img_map[image_id]

        if len(pred) == 0:
            continue

        img_meta = dataset.get_image_meta(image_id)
        width = img_meta['width']
        height = img_meta['height']
        pred = pred.resize((width, height))
        pred = pred.convert('xywh')

        boxes = pred.bbox.tolist()
        scores = pred.get_field('scores').tolist()
        labels = pred.get_field('labels').tolist()

        labels = [dataset.contiguous_category_id_to_json_id[i] for i in labels]

        coco_results.extend(
            [
                {
                    'image_id': orig_id,
                    'category_id': labels[k],
                    'bbox': box,
                    'score': scores[k],
                }
                for k, box in enumerate(boxes)
            ]
        )

    return coco_results


class COCOResult:
    METRICS = {
        'bbox': ['AP', 'AP50', 'AP75', 'APs', 'APm', 'APl'],
        'segm': ['AP', 'AP50', 'AP75', 'APs', 'APm', 'APl'],
        'box_proposal': [
            'AR@100',
            'ARs@100',
            'ARm@100',
            'ARl@100',
            'AR@1000',
            'ARs@1000',
            'ARm@1000',
            'ARl@1000',
        ],
        'keypoints': ['AP', 'AP50', 'AP75', 'APm', 'APl'],
    }

    def __init__(self, *iou_types):
        allowed_types = ("box_proposal", "bbox", "segm", "keypoints")
        assert all(iou_type in allowed_types for iou_type in iou_types)
        results = OrderedDict()
        for iou_type in iou_types:
            results[iou_type] = OrderedDict(
                [(metric, -1) for metric in COCOResult.METRICS[iou_type]]
            )
        self.results = results

    def update(self, coco_eval):
        if coco_eval is None:
            return

        assert isinstance(coco_eval, COCOeval)
        s = coco_eval.stats
        iou_type = coco_eval.params.iouType
        res = self.results[iou_type]
        metrics = COCOResult.METRICS[iou_type]
        for idx, metric in enumerate(metrics):
            res[metric] = s[idx]

    def __repr__(self):
        return repr(self.results)
