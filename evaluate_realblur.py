## Restormer: Efficient Transformer for High-Resolution Image Restoration
## Syed Waqas Zamir, Aditya Arora, Salman Khan, Munawar Hayat, Fahad Shahbaz Khan, and Ming-Hsuan Yang
## https://arxiv.org/abs/2111.09881

import os
import numpy as np
from glob import glob
from natsort import natsorted
from skimage import io
import cv2
from skimage.metrics import structural_similarity
from tqdm import tqdm
import concurrent.futures
from argparse import ArgumentParser
from concurrent.futures import ProcessPoolExecutor

# def image_align(deblurred, gt):
#   # this function is based on kohler evaluation code
#   z = deblurred
#   c = np.ones_like(z)
#   x = gt

#   zs = (np.sum(x * z) / np.sum(z * z)) * z # simple intensity matching

#   warp_mode = cv2.MOTION_HOMOGRAPHY
#   warp_matrix = np.eye(3, 3, dtype=np.float32)

#   # Specify the number of iterations.
#   number_of_iterations = 100

#   termination_eps = 0

#   criteria = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT,
#               number_of_iterations, termination_eps)

#   # Run the ECC algorithm. The results are stored in warp_matrix.
#   (cc, warp_matrix) = cv2.findTransformECC(cv2.cvtColor(x, cv2.COLOR_RGB2GRAY), cv2.cvtColor(zs, cv2.COLOR_RGB2GRAY), warp_matrix, warp_mode, criteria, inputMask=None, gaussFiltSize=5)

#   target_shape = x.shape
#   shift = warp_matrix

#   zr = cv2.warpPerspective(
#     zs,
#     warp_matrix,
#     (target_shape[1], target_shape[0]),
#     flags=cv2.INTER_CUBIC+ cv2.WARP_INVERSE_MAP,
#     borderMode=cv2.BORDER_REFLECT)

#   cr = cv2.warpPerspective(
#     np.ones_like(zs, dtype='float32'),
#     warp_matrix,
#     (target_shape[1], target_shape[0]),
#     flags=cv2.INTER_NEAREST+ cv2.WARP_INVERSE_MAP,
#     borderMode=cv2.BORDER_CONSTANT,
#     borderValue=0)

#   zr = zr * cr
#   xr = x * cr

#   return zr, xr, cr, shift

def image_align(deblurred, gt):
    # Ensure both images are of same shape
    target_shape = gt.shape[:2]
    deblurred_resized = cv2.resize(deblurred, (target_shape[1], target_shape[0]), interpolation=cv2.INTER_LINEAR)
    
    z = deblurred_resized
    c = np.ones_like(z)
    x = gt

    zs = (np.sum(x * z) / np.sum(z * z)) * z # simple intensity matching

    warp_mode = cv2.MOTION_HOMOGRAPHY
    warp_matrix = np.eye(3, 3, dtype=np.float32)

    number_of_iterations = 100
    termination_eps = 0

    criteria = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, number_of_iterations, termination_eps)

    (cc, warp_matrix) = cv2.findTransformECC(cv2.cvtColor(x, cv2.COLOR_RGB2GRAY), cv2.cvtColor(zs, cv2.COLOR_RGB2GRAY), warp_matrix, warp_mode, criteria, inputMask=None, gaussFiltSize=5)

    target_shape = x.shape
    shift = warp_matrix

    zr = cv2.warpPerspective(
        zs,
        warp_matrix,
        (target_shape[1], target_shape[0]),
        flags=cv2.INTER_CUBIC + cv2.WARP_INVERSE_MAP,
        borderMode=cv2.BORDER_REFLECT)

    cr = cv2.warpPerspective(
        np.ones_like(zs, dtype='float32'),
        warp_matrix,
        (target_shape[1], target_shape[0]),
        flags=cv2.INTER_NEAREST + cv2.WARP_INVERSE_MAP,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=0)

    zr = zr * cr
    xr = x * cr

    return zr, xr, cr, shift

# def compute_psnr(image_true, image_test, image_mask, data_range=None):
#   # this function is based on skimage.metrics.peak_signal_noise_ratio
#   err = np.sum((image_true - image_test) ** 2, dtype=np.float64) / np.sum(image_mask)
#   return 10 * np.log10((data_range ** 2) / err)

def compute_psnr(image_true, image_test, image_mask, data_range=None):
    err = np.sum((image_true - image_test) ** 2, dtype=np.float64) / np.sum(image_mask)
    return 10 * np.log10((data_range ** 2) / err)


# def compute_ssim(tar_img, prd_img, cr1):
#     ssim_pre, ssim_map = structural_similarity(tar_img, prd_img, channel_axis=2, gaussian_weights=True, use_sample_covariance=False, data_range = 1.0, full=True)
#     ssim_map = ssim_map * cr1
#     r = int(3.5 * 1.5 + 0.5)  # radius as in ndimage
#     win_size = 2 * r + 1
#     pad = (win_size - 1) // 2
#     ssim = ssim_map[pad:-pad,pad:-pad,:]
#     crop_cr1 = cr1[pad:-pad,pad:-pad,:]
#     ssim = ssim.sum(axis=0).sum(axis=0)/crop_cr1.sum(axis=0).sum(axis=0)
#     ssim = np.mean(ssim)
#     return ssim

def compute_ssim(tar_img, prd_img, cr1):
    ssim_pre, ssim_map = structural_similarity(tar_img, prd_img, channel_axis=2, gaussian_weights=True, use_sample_covariance=False, data_range=1.0, full=True)
    ssim_map = ssim_map * cr1
    r = int(3.5 * 1.5 + 0.5)
    win_size = 2 * r + 1
    pad = (win_size - 1) // 2
    ssim = ssim_map[pad:-pad, pad:-pad, :]
    crop_cr1 = cr1[pad:-pad, pad:-pad, :]
    ssim = ssim.sum(axis=0).sum(axis=0) / crop_cr1.sum(axis=0).sum(axis=0)
    ssim = np.mean(ssim)
    return ssim

# def proc(filename):
#     tar,prd = filename
#     tar_img = io.imread(tar)
#     prd_img = io.imread(prd)
    
#     tar_img = tar_img.astype(np.float32)/255.0
#     prd_img = prd_img.astype(np.float32)/255.0
    
#     prd_img, tar_img, cr1, shift = image_align(prd_img, tar_img)

#     PSNR = compute_psnr(tar_img, prd_img, cr1, data_range=1)
#     SSIM = compute_ssim(tar_img, prd_img, cr1)
#     return (PSNR,SSIM)
def proc(filename):
    tar, prd = filename
    tar_img = io.imread(tar)
    prd_img = io.imread(prd)
    tar_img = tar_img.astype(np.float32) / 255.0
    prd_img = prd_img.astype(np.float32) / 255.0
    prd_img, tar_img, cr1, shift = image_align(prd_img, tar_img)
    PSNR = compute_psnr(tar_img, prd_img, cr1, data_range=1)
    SSIM = compute_ssim(tar_img, prd_img, cr1)
    return (PSNR, SSIM)

# datasets = ['RealBlur_J', 'RealBlur_R', 'SAM_datasets_blur_png_3x3', 'SAM_datasets_blur_png_5x5', 'SAM_datasets_blur_png_7x7', 'SAM_datasets_blur_png_9x9']

# parser = ArgumentParser()
# parser.add_argument('--dataset', choices=['RealBlur_J', 'RealBlur_R', 'SAM_datasets_blur_png_3x3', 'SAM_datasets_blur_png_5x5', 'SAM_datasets_blur_png_7x7', 'SAM_datasets_blur_png_9x9'],  required=True)
# parser.add_argument("--dir", type=str, required=True)
# args = parser.parse_args()

# file_path = os.path.join(args.dir, 'visualization', args.dataset)
# print(file_path)
# gt_path = os.path.join('datasets', 'test', args.dataset, 'target')
# path_list = natsorted(glob(os.path.join(file_path, '*.png')) + glob(os.path.join(file_path, '*.jpg')))
# gt_list = natsorted(glob(os.path.join(gt_path, '*.png')) + glob(os.path.join(gt_path, '*.jpg')))
# assert len(path_list) != 0, "Predicted files not found"
# assert len(gt_list) != 0, "Target files not found"

# psnr, ssim = [], []
# img_files =[(i, j) for i,j in zip(gt_list,path_list)]
# with concurrent.futures.ProcessPoolExecutor(max_workers=10) as executor:
#     for filename, PSNR_SSIM in zip(img_files, executor.map(proc, img_files)):
#         psnr.append(PSNR_SSIM[0])
#         ssim.append(PSNR_SSIM[1])

# avg_psnr = sum(psnr)/len(psnr)
# avg_ssim = sum(ssim)/len(ssim)

# print('For {:s} dataset PSNR: {:f} SSIM: {:f}\n'.format(args.dataset, avg_psnr, avg_ssim))

# if __name__ == '__main__':
#     datasets = ['RealBlur_J', 'RealBlur_R']
#     parser = ArgumentParser()
#     parser.add_argument('--dataset', choices=['RealBlur_J', 'RealBlur_R'], required=True)
#     parser.add_argument("--dir", type=str, required=True)
#     args = parser.parse_args()
#     file_path = os.path.join(args.dir, 'visualization', args.dataset)
#     gt_path = os.path.join('datasets', 'test', args.dataset, 'target')
#     path_list = natsorted(glob(os.path.join(file_path, '*.png')) + glob(os.path.join(file_path, '*.jpg')))
#     gt_list = natsorted(glob(os.path.join(gt_path, '*.png')) + glob(os.path.join(gt_path, '*.jpg')))
#     assert len(path_list) != 0, "Predicted files not found"
#     assert len(gt_list) != 0, "Target files not found"
#     img_files = [(i, j) for i, j in zip(gt_list, path_list)]
#     with ProcessPoolExecutor() as executor:
#         for filename, PSNR_SSIM in zip(img_files, executor.map(proc, img_files)):
#             print(f"{filename}: PSNR={PSNR_SSIM[0]}, SSIM={PSNR_SSIM[1]}")

if __name__ == '__main__':
    datasets = ['RealBlur_J', 'RealBlur_R', 'SAM_datasets_blur_png_3x3', 'SAM_datasets_blur_png_5x5', 'SAM_datasets_blur_png_7x7', 'SAM_datasets_blur_png_9x9', 'SAM_datasets_blur']
    parser = ArgumentParser()
    parser.add_argument('--dataset', choices=['RealBlur_J', 'RealBlur_R', 'SAM_datasets_blur_png_3x3', 'SAM_datasets_blur_png_5x5', 'SAM_datasets_blur_png_7x7', 'SAM_datasets_blur_png_9x9', 'SAM_datasets_blur'], required=True)
    parser.add_argument("--dir", type=str, required=True)
    args = parser.parse_args()
    file_path = os.path.join(args.dir, 'visualization', args.dataset)
    gt_path = os.path.join('datasets', 'test', args.dataset, 'target')
    path_list = natsorted(glob(os.path.join(file_path, '*.png')) + glob(os.path.join(file_path, '*.jpg')))
    gt_list = natsorted(glob(os.path.join(gt_path, '*.png')) + glob(os.path.join(gt_path, '*.jpg')))
    assert len(path_list) != 0, "Predicted files not found"
    assert len(gt_list) != 0, "Target files not found"

    img_files = [(i, j) for i, j in zip(gt_list, path_list)]
    psnr_values, ssim_value = [], []
    with ProcessPoolExecutor() as executor:
        for filename, PSNR_SSIM in zip(img_files, executor.map(proc, img_files)):
            # print(f"{filename}: PSNR={PSNR_SSIM[0]}, SSIM={PSNR_SSIM[1]}")
            ssim_value.append(PSNR_SSIM[1])
            psnr_values.append(PSNR_SSIM[0])
    avg_psnr = sum(psnr_values)/len(psnr_values)
    avg_ssim = sum(ssim_value)/len(ssim_value)
    print(f'avg_psnr:{avg_psnr}, avg_ssim:{avg_ssim}')