import os
from os import listdir
from os.path import isfile, join, isdir
import glob
import shutil
import argparse
from geopatterns import GeoPattern
import cairosvg
import random
import cv2
import scipy
from scipy import signal
from skimage.draw import line_aa
from helpers import *
import math  
import imageio
import skimage.transform
import pdb
import h5py


def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument("--bg_path", required=True)
    parser.add_argument("--fg_path", required=True)
    parser.add_argument("--dataset_path", required=True)
    parser.add_argument("--dataset_size", type=int, required=True)
    parser.add_argument("--sequence_len", type=int, required=True)


    return parser.parse_args()


def main():
    # random.seed(3524)
    max_shape = [256, 512]
    generate_inputs = True

    args = parse_args()

    fgs = [f for f in listdir(args.fg_path) if isfile(join(args.fg_path, f))]
    videos = [f for f in listdir(args.bg_path) if isdir(join(args.bg_path, f))]
    bgs = [sorted(glob.glob(join(args.bg_path,f,'color','*.jpg'))) for f in videos]

    ti = 0

    compression_type = None

    while ti < (args.dataset_size):
        print(ti)
        si = random.randint(0,len(videos)-1)
        vid = videos[si]
        frms = bgs[si]
        start_frame = random.randint(3,len(frms)-1-args.sequence_len)   # first frame, start at 3 to fix frmi issue
        fg = fgs[random.randint(0,len(fgs)-1)]

        F = cv2.imread(join(args.fg_path,fg))/255

        shp = np.mod(F.shape,100)
        if np.min(shp[:2]) < 4: # changing this from 20 to 4 to allow smaller FMOs
            continue
        F = skimage.transform.resize(F, shp, order=3)

        diam = round(min(F.shape[0:2]))

        F = F[:diam,:diam,:]
        rad = diam/2
        M = diskMask(rad)
        M3 = np.repeat(M[:, :, np.newaxis], 3, axis=2)
        FM = F*M3

        seq = 0
        ori = random.uniform(0,2*math.pi)
        while seq < args.sequence_len:
            print("%d_%d" % (ti, seq))

            # create background
            bg = frms[start_frame + seq]    # use next frame in video as background
            frmi = int(bg[-12:-4])
            if frmi < 3:
                continue
            B = cv2.imread(bg)/255

            bg0 = bg[:-12]+str(frmi-1).zfill(8) +bg[-4:]
            bg00 = bg[:-12]+str(frmi-2).zfill(8) +bg[-4:]
            B0 = cv2.imread(bg0)/255
            B00 = cv2.imread(bg00)/255

            BC = np.zeros([B.shape[0],B.shape[1],3,3])
            BC[:,:,:,0] = B
            BC[:,:,:,1] = B0
            BC[:,:,:,2] = B00
            BMED = np.median(BC,3)


            ## Generate random trajectories
            H = np.zeros(B.shape[0:2])
            rind = random.randint(0,9)
            tlen = random.uniform(1.5,9.0)*rad
            if seq == 0:
                start = [random.randint(0, H.shape[0]-1), random.randint(0, H.shape[1]-1)]
            else:
                start = prev_end  # start from endpoint of previous frame, should probably add some noise eventually
            towrite = np.zeros([2,4])
            towrite[:,0] = start
            if rind == 0:
                ## generate broken line
                prc = random.uniform(0.15,0.85)
                ori0 = []
                for pr1 in [prc, 1-prc]:
                    if ori0 != []:
                        while True:
                            ori = random.uniform(0,2*math.pi)
                            delta = np.mod(ori - ori0 + 3*math.pi, 2*math.pi) - math.pi
                            if np.abs(delta) < math.pi/6 or np.abs(delta) > 5*math.pi/6: ## if < 30 or > 150 degrees
                                continue
                            break

                    end = [round(start[0] + math.sin(ori)*tlen*pr1), round(start[1] + math.cos(ori)*tlen*pr1)]
                    rr, cc, val = line_aa(start[0], start[1], end[0], end[1])
                    valid = np.logical_and(np.logical_and(rr < H.shape[0], cc < H.shape[1]), np.logical_and(rr > 0, cc > 0))
                    rr = rr[valid]
                    cc = cc[valid]
                    val = val[valid]
                    H[rr, cc] = val   
                    if ori0 == []:
                        towrite[:,1] = np.array(end) - np.array(start) 
                    else:
                        towrite[:,3] = np.array(end) - np.array(start) 

                    start = end
                    ori0 = ori
            elif rind == 1:
                ## generate parabola
                end = [round(start[0] + math.sin(ori)*tlen), round(start[1] + math.cos(ori)*tlen)]
                towrite[:,1] = np.array(end) - np.array(start) 
                towrite[:,2] = [random.uniform(10.0,20.0), random.uniform(10.0,20.0)] 
                H = renderTraj(towrite, H) 
            else:
                end = [round(start[0] + math.sin(ori)*tlen), round(start[1] + math.cos(ori)*tlen)]
                rr, cc, val = line_aa(start[0], start[1], end[0], end[1])
                valid = np.logical_and(np.logical_and(rr < H.shape[0], cc < H.shape[1]), np.logical_and(rr > 0, cc > 0))
                rr = rr[valid]
                cc = cc[valid]
                val = val[valid]
                H[rr, cc] = val    
                towrite[:,1] = np.array(end) - np.array(start) 

            # if np.sum(H) < tlen:
            #     continue
            prev_end = end
            SEGMASK = psf_to_full_mask(H, M)
            if np.sum(H) > 0:
                H = H/np.sum(H)
            ########


            HM = signal.fftconvolve(H, M, mode='same')
            HM3 = np.repeat(HM[:, :, np.newaxis], 3, axis=2)
            HF = np.zeros(B.shape)
            for kk in range(3):
                HF[:,:,kk] = signal.fftconvolve(H, FM[:,:,kk], mode='same')

            im = B*(1-HM3) + HF

            TH = np.zeros(B.shape[0:2])
            TH[round(TH.shape[0]/2),round(TH.shape[1]/2)] = 1
            M = signal.fftconvolve(TH, M, mode='same')
            Fsave = np.zeros(B.shape)
            for kk in range(3):
                Fsave[:,:,kk] = signal.fftconvolve(TH, FM[:,:,kk], mode='same')

            # imshow(im)
            # pdb.set_trace()
            fname = "%08d_%08d_" % (ti, seq)
            # if seq == 0:
            #     imageio.imwrite(join(args.dataset_path,fname+"F.png"), (255*Fsave).astype(np.uint8))
            #     imageio.imwrite(join(args.dataset_path,fname+"M.png"), (255*M).astype(np.uint8))
            # imageio.imwrite(join(args.dataset_path,fname+"bgr.png"), (255*BMED).astype(np.uint8))
            imageio.imwrite(join(args.dataset_path,fname+"im.png"), (255*im).astype(np.uint8))
            # imageio.imwrite(join(args.dataset_path,fname+"psf.png"), (255*(H/np.max(H))).astype(np.uint8))
            imageio.imwrite(join(args.dataset_path,fname+"segmask.png"), (255*(SEGMASK/max(np.max(SEGMASK),1))).astype(np.uint8))
            
            # with open(join(args.dataset_path,fname+"traj.txt"), 'w') as fff:
            #     for k1 in range(towrite.shape[0]):
            #         for k2 in range(towrite.shape[1]):
            #             fff.write('%.1f ' % towrite[k1,k2])
            #         fff.write('\n')
            seq += 1
        ti = ti + 1

def get_data_processed(im, bgr, M, psf, max_shape):
    X = np.zeros([1,max_shape[0],max_shape[1],6])
    Y = np.zeros([1,max_shape[0],max_shape[1],1])
    ki = 0
    I = skimage.transform.resize(im, max_shape, order=3)
    I = I - np.mean(I)
    I = I / np.sqrt(np.var(I))

    H = skimage.transform.resize(psf, max_shape, order=1)

    M = skimage.transform.resize(M, max_shape, order=1)

    rad = np.sqrt(np.sum(M))
    DT = scipy.ndimage.morphology.distance_transform_edt(H == 0)
    DT = DT / (2*rad)
    DT[DT > 1] = 1

    X[ki,:,:,:3] = I

    if True:
        B = skimage.transform.resize(bgr, max_shape, order=3)
        B = B - np.mean(B)
        B = B / np.sqrt(np.var(B))
        X[ki,:,:,3:] = B

    Y[ki,:,:,0] = 1 - DT

    return X,Y


def psf_to_full_mask(psf, disk):
    ''' Creates a full segmentation mask from the psf and the object/disk shape '''
    imshape = psf.shape
    tiled_mask_shape = [dim*3 for dim in imshape]
    tiled_mask = np.zeros(tiled_mask_shape) # padding so that we don't deal with out of bounds errors
    diskshape = disk.shape
    disk_center = [dim/2 for dim in diskshape]
    line_idx = np.nonzero(psf)  # get pixels on psf line
    layers = []
    for x,y in zip(line_idx[0], line_idx[1]):
        new_layer = np.zeros_like(tiled_mask)
        startx = int(imshape[0] + x - disk_center[0])
        endx = int(startx + diskshape[0])
        starty = int(imshape[1] + y - disk_center[1])
        endy = int(starty + diskshape[1])
        new_layer[startx:endx, starty:endy] = disk    # place a disk centered at point on psf line 
        layers.append(new_layer)
    if layers == []:
        return psf
    full_mask_tiled = np.maximum.reduce(layers)     # combine all disks
    full_mask = full_mask_tiled[imshape[0]:2*imshape[0], imshape[1]:2*imshape[1]]   # remove "out of bounds" padding
    return full_mask


if __name__ == "__main__":
    main()