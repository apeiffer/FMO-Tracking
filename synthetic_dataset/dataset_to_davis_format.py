import numpy as np
import cv2
import os
import shutil
import random
import yaml
import PIL
from PIL import Image
import random


# New directory = empty directory with subfolders Annotations, ImageSets, JPEGImages
# Old directory = output from dataset generator
def create_file_structure_for_train(old_directory, new_directory):
	im = Image.open("/ihome/alabrinidis/kmb256/cs2770/databases/DAVIS2017/Annotations/480p/tennis/00000.png")
	palette = im.getpalette()

	vid_nums = set()
	for i, filename in enumerate(os.listdir(old_directory)):
		if i%100 == 0:
			print(i)
		
		vid_num, seq_num, filetype = filename.split("_")
		seq_num = seq_num[3:]	# change from 8 digit to 5 digit number (removing leading zeros)
		vid_nums.add(vid_num)
		
		vid_dir = os.path.join(new_directory, "JPEGImages/", vid_num)
		if not os.path.exists(vid_dir):
			os.makedirs(vid_dir)
			vid_dir2 = os.path.join(new_directory, "Annotations/", vid_num)
			os.makedirs(vid_dir2)
			vid_dir3 = os.path.join(new_directory, "Annotations_traj/", vid_num)
			os.makedirs(vid_dir3)
		
		if filetype.endswith("im.png"):
			old_path = os.path.join(old_directory, filename)
			im = Image.open(old_path)
			new_path = os.path.join(new_directory, "JPEGImages/", vid_num, seq_num+'.jpg')
			im.save(new_path)	# convert to jpg
		elif filetype.endswith("segmask.png"):
			old_path = os.path.join(old_directory, filename)
			im = Image.open(old_path)
			a = np.array(im)  
			a[np.where(a!=0)] = 1
			im = Image.fromarray(a)
			im.putpalette(palette)
			new_path = os.path.join(new_directory, "Annotations/", vid_num, seq_num+'.png')
			im.save(new_path)
		elif filetype.endswith("psf.png"):
			old_path = os.path.join(old_directory, filename)
			im = Image.open(old_path)
			a = np.array(im)  
			a[np.where(a!=0)] = 1
			im = Image.fromarray(a)
			im.putpalette(palette)
			new_path = os.path.join(new_directory, "Annotations_traj/", vid_num, seq_num+'.png')
			im.save(new_path)
		else:
			continue

	vid_nums = list(vid_nums)
	random.shuffle(vid_nums)
	num_train = int(0.9*len(vid_nums))
	im_set_path = os.path.join(new_directory, "ImageSets", "train.txt")
	with open(im_set_path, 'w') as file_handler:
		for item in vid_nums[:num_train]:
			file_handler.write("{}\n".format(item))
	im_set_path = os.path.join(new_directory, "ImageSets", "val.txt")
	with open(im_set_path, 'w') as file_handler:
		for item in vid_nums[num_train:]:
			file_handler.write("{}\n".format(item))


def create_file_structure_for_fmo():
	im = Image.open("/ihome/alabrinidis/kmb256/cs2770/databases/DAVIS2017/Annotations/480p/tennis/00000.png")
	palette = im.getpalette()

	vids = []
	im_dir = "/ihome/alabrinidis/kmb256/cs2770/FMOv2"
	gt_dir = "/ihome/alabrinidis/kmb256/cs2770/FMOv2_gt"
	new_dir = "/ihome/alabrinidis/kmb256/cs2770/databases/FMOv2"
	for folder in os.listdir(im_dir):
		if os.path.isdir(os.path.join(im_dir, folder)):
			vids.append(folder)

			vid_dir = os.path.join(new_dir, "JPEGImages/", folder)
			os.makedirs(vid_dir)
			vid_dir2 = os.path.join(new_dir, "Annotations/", folder)
			os.makedirs(vid_dir2)

			for filename in os.listdir(os.path.join(im_dir, folder)):
				fname = filename[:-4]	# remove .png
				fname = str(int(fname)-1).zfill(5)	# change from 1 indexed to 0 indexed
				old_path = os.path.join(im_dir, folder, filename)
				im = Image.open(old_path)		
				new_path = os.path.join(new_dir, "JPEGImages/", folder, fname + '.jpg')
				im.save(new_path)	# convert to jpg
			for filename in os.listdir(os.path.join(gt_dir, folder)):
				old_path = os.path.join(gt_dir, folder, filename)
				im = Image.open(old_path)
				a = np.array(im)
				a[np.where(a!=0)] = 1
				im = Image.fromarray(a)
				im.putpalette(palette)
				fname = filename[:-4]	# remove .png
				fname = str(int(fname)-1).zfill(5)	# change from 1 indexed to 0 indexed
				new_path = os.path.join(new_dir, "Annotations/", folder, fname + '.png')
				im.save(new_path)

	im_set_path = os.path.join(new_dir, "ImageSets", "val.txt")
	with open(im_set_path, 'w') as file_handler:
		for item in vids:
			file_handler.write("{}\n".format(item))


def create_yaml(dataset_dir, filename):
	imagesetpath = os.path.join(dataset_dir, 'ImageSets', 'train.txt')
	with open(imagesetpath) as f:
		content = f.readlines()
	train_set = [x.strip() for x in content] 

	imagesetpath = os.path.join(dataset_dir, 'ImageSets', 'val.txt')
	with open(imagesetpath) as f:
		content = f.readlines()
	val_set = [x.strip() for x in content] 

	data = {}
	data['sets'] = ['train', 'val']
	data['years'] = [2017]
	seqs = []
	for folder in os.listdir(os.path.join(dataset_dir, 'Annotations')):
		path = os.path.join(dataset_dir, 'Annotations', folder)
		num_frames = len([f for f in os.listdir(path) if os.path.isfile(os.path.join(path, f))])
		if folder in train_set:
			chosen_set = 'train'
		elif folder in val_set:
			chosen_set = 'val'
		else:
			print("ERROR")
		seq = {'name': folder, 'num_frames': num_frames, 'set': chosen_set, 'year': 2017}
		seqs.append(seq)
	data['sequences'] = seqs
	with open(filename, 'w') as outfile:
		yaml.dump(data, outfile)


def resize_dataset(db_directory, dataset):
  new_dataset = dataset + '_resized'
  os.mkdir(os.path.join(db_directory, new_dataset))

  for folder in ['Annotations', 'JPEGImages', 'Annotations_traj']:
    os.mkdir(os.path.join(db_directory, new_dataset, folder))
    for seq in os.listdir(os.path.join(db_directory, dataset, folder)):
      os.mkdir(os.path.join(db_directory, new_dataset, folder, seq))
      for file in os.listdir(os.path.join(db_directory, dataset, folder, seq)):
        im = Image.open(os.path.join(db_directory, dataset, folder, seq, file))
        if im.size[0] < 400:
          target_dims = (int(im.size[0]), int(im.size[1]))
        elif im.size[0] < 800:
          target_dims = (int(im.size[0]/2), int(im.size[1]/2))
        else:
          target_dims = (int(im.size[0]/4), int(im.size[1]/4))
        im_resized = im.resize(target_dims, PIL.Image.NEAREST)
        im_resized.save(os.path.join(db_directory, new_dataset, folder, seq, file))

  shutil.copytree(os.path.join(db_directory, dataset, 'ImageSets'), os.path.join(db_directory, new_dataset, 'ImageSets'))


def reorder_for_trajectory(input_dir, output_dir):
	os.mkdir(output_dir)
	for folder in ['Annotations', 'JPEGImages']:
		os.mkdir(os.path.join(output_dir, folder))
		for seq in os.listdir(os.path.join(input_dir, folder)):
			os.mkdir(os.path.join(output_dir, folder, seq))
			seq_len = len(os.listdir(os.path.join(input_dir, folder, seq)))
			for i, file in enumerate(sorted(os.listdir(os.path.join(input_dir, folder, seq)))):
				if folder == 'Annotations':
					if i == 0:
						continue
					filenum = int(file[:5])
					new_filename = str(filenum-1).zfill(5) + '.png'
					shutil.copyfile(os.path.join(input_dir, folder, seq, file), os.path.join(output_dir, folder, seq, new_filename))
				elif folder == 'JPEGImages':
					if i == seq_len-1:
						continue
					shutil.copyfile(os.path.join(input_dir, folder, seq, file), os.path.join(output_dir, folder, seq, file))
	shutil.copytree(os.path.join(input_dir, 'ImageSets'), os.path.join(output_dir, 'ImageSets'))



