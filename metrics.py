import numpy as np
from PIL import Image
import os
from scipy.misc import imresize


def get_all_metrics(tp, fp, fn):
	precision = tp / (tp + fp + np.finfo(float).eps)
	recall = tp / (tp + fn + np.finfo(float).eps)
	f1 = 2*tp/(2*tp + fn + fp + np.finfo(float).eps)
	return precision, recall, f1


def calculate_metrics(gt_dir, pred_dir):
	vid_stats = {}
	for vid in os.listdir(gt_dir):
		print(vid)
		tp = 0
		fp = 0
		fn = 0
		for gt_fname in os.listdir(os.path.join(gt_dir, vid)):
			gt = Image.open(os.path.join(gt_dir, vid, gt_fname))
			gt_arr = np.asarray(gt)
			pred_fname = gt_fname.split(".")[0] + "_instance_00.png"
			pred = Image.open(os.path.join(pred_dir, vid, pred_fname))
			pred_resized = imresize(pred, np.shape(gt_arr), interp='nearest')	# predictions are incorrect size, this is how they resized for rvos submission
			pred_arr = np.asarray(pred_resized)/255
			if np.sum(gt_arr) > 0:	# fmo present in gt
				if np.sum(pred_arr) == 0:
					fn += 1
				else:
					gt_arr_bool = np.array(gt_arr, dtype=bool)
					pred_arr_bool = np.array(pred_arr, dtype=bool)
					intersection = np.logical_and(gt_arr_bool, pred_arr_bool)
					union = np.logical_or(gt_arr_bool, pred_arr_bool)
					iou = np.sum(intersection) / np.sum(union)
					if iou > 0.5:
						tp += 1
					else:
						fp += 1
						fn += 1
			else:	# no fmo in gt
				if np.sum(pred_arr) > 0:
					fp += 1

		vid_stats[vid] = {}	


		p, r, f1 = get_all_metrics(tp, fp, fn)
		vid_stats[vid]['precision'] = p
		vid_stats[vid]['recall'] = r
		vid_stats[vid]['f1'] = f1
	overall_p = sum([v['precision'] for k,v in vid_stats.items()])/len(vid_stats)
	overall_r = sum([v['recall'] for k,v in vid_stats.items()])/len(vid_stats)
	overall_f1 = sum([v['f1'] for k,v in vid_stats.items()])/len(vid_stats)
	overall_stats = {'precision': overall_p,
					 'recall': overall_r,
					 'f1': overall_f1}
	return overall_stats, vid_stats