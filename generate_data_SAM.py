import cv2
import os
import numpy as np

import shutil
# Define folder paths
# input_folder = "E:\Project\HI-Diff\SAM_datasets"
# output_folder_01 = "E:\Project\HI-Diff\SAM_datasets_blur_png_3x3"
# output_folder_02 = "E:\Project\HI-Diff\SAM_datasets_blur_png_5x5"
# output_folder_03 = "E:\Project\HI-Diff\SAM_datasets_blur_png_7x7"
# output_folder_04 = "E:\Project\HI-Diff\SAM_datasets_blur_png_9x9"

# # Ensure output folder exists
# os.makedirs(output_folder_01, exist_ok=True)
# os.makedirs(output_folder_02, exist_ok=True)
# os.makedirs(output_folder_03, exist_ok=True)
# os.makedirs(output_folder_04, exist_ok=True)

# # Get list of .bmp images and sort them
# image_files = sorted([f for f in os.listdir(input_folder) if f.lower().endswith(('.bmp', '.png', '.jpg'))])


# # Process each image
# for image_file in image_files:  # Remove zip() since we just want to iterate through the list
#     image_path = os.path.join(input_folder, image_file)
#     new_name = os.path.splitext(image_file)[0] + ".png"  # Ensure extension is .png

#     # Read the image
#     image = cv2.imread(image_path)

#     if image is None:
#         print(f"Skipping {image_file}: Unable to read.")
#         continue

#     # Apply Gaussian blur
#     blurred_image_3 = cv2.GaussianBlur(image, (3, 3), 0)
#     blurred_image_5 = cv2.GaussianBlur(image, (5, 5), 0)
#     blurred_image_7 = cv2.GaussianBlur(image, (7, 7), 0)
#     blurred_image_9 = cv2.GaussianBlur(image, (9, 9), 0)

#     # Save as PNG with new name - fix path joining
#     cv2.imwrite(os.path.join(output_folder_01, new_name), blurred_image_3)
#     cv2.imwrite(os.path.join(output_folder_02, new_name), blurred_image_5) 
#     cv2.imwrite(os.path.join(output_folder_03, new_name), blurred_image_7)
#     cv2.imwrite(os.path.join(output_folder_04, new_name), blurred_image_9)
#     print(f"Processed {image_file} -> {os.path.join(output_folder_01, new_name)}, {os.path.join(output_folder_02, new_name)}, {os.path.join(output_folder_03, new_name)}, {os.path.join(output_folder_04, new_name)}")
# print("Blurring and renaming completed successfully!")


# Create output folders for input and target

# # Define the folder paths correctly
# folder_01 = "E:\\Project\\HI-Diff\\SAM_datasets_blur_png_3x3"
# folder_02 = "E:\\Project\\HI-Diff\\SAM_datasets_blur_png_5x5"
# folder_03 = "E:\\Project\\HI-Diff\\SAM_datasets_blur_png_7x7"
# folder_04 = "E:\\Project\\HI-Diff\\SAM_datasets_blur_png_9x9"

# blurred_SAM_dataset = "E:\\Project\\HI-Diff\\SAM_datasets_blur"

# # Create output directory
# os.makedirs(blurred_SAM_dataset, exist_ok=True)

# # Check if directories exist
# for folder in [folder_01, folder_02, folder_03, folder_04]:
#     if not os.path.exists(folder):
#         print(f"Warning: Directory {folder} does not exist!")
#         continue

# # Get list of files from each directory
# image_file_01 = sorted([f for f in os.listdir(folder_01) if f.lower().endswith('.png')]) if os.path.exists(folder_01) else []
# image_file_02 = sorted([f for f in os.listdir(folder_02) if f.lower().endswith('.png')]) if os.path.exists(folder_02) else []
# image_file_03 = sorted([f for f in os.listdir(folder_03) if f.lower().endswith('.png')]) if os.path.exists(folder_03) else []
# image_file_04 = sorted([f for f in os.listdir(folder_04) if f.lower().endswith('.png')]) if os.path.exists(folder_04) else []
# # Process images from each folder and copy to common folder with kernel size in filename
# for image_file in image_file_01:
#     src_path = os.path.join(folder_01, image_file)
#     base_name = os.path.splitext(image_file)[0]
#     new_name = f"{base_name}_3x3.png"
#     dst_path = os.path.join(blurred_SAM_dataset, new_name)
#     shutil.copy2(src_path, dst_path)
#     print(f"Copied {image_file} -> {new_name}")

# for image_file in image_file_02:
#     src_path = os.path.join(folder_02, image_file) 
#     base_name = os.path.splitext(image_file)[0]
#     new_name = f"{base_name}_5x5.png"
#     dst_path = os.path.join(blurred_SAM_dataset, new_name)
#     shutil.copy2(src_path, dst_path)
#     print(f"Copied {image_file} -> {new_name}")

# for image_file in image_file_03:
#     src_path = os.path.join(folder_03, image_file)
#     base_name = os.path.splitext(image_file)[0]
#     new_name = f"{base_name}_7x7.png"
#     dst_path = os.path.join(blurred_SAM_dataset, new_name)
#     shutil.copy2(src_path, dst_path)
#     print(f"Copied {image_file} -> {new_name}")

# for image_file in image_file_04:
#     src_path = os.path.join(folder_04, image_file)
#     base_name = os.path.splitext(image_file)[0]
#     new_name = f"{base_name}_9x9.png"
#     dst_path = os.path.join(blurred_SAM_dataset, new_name)
#     shutil.copy2(src_path, dst_path)
#     print(f"Copied {image_file} -> {new_name}")

print("All files copied and renamed successfully!")

folder = "E:\\Project\\HI-Diff\\SAM_datasets_256x256"
# Create output directory for each kernel size
output_base = "E:\\Project\\HI-Diff\\SAM_datasets_blur"
kernel_sizes = ["3x3", "5x5", "7x7", "9x9"]

for kernel in kernel_sizes:
    output_dir = os.path.join(output_base, f"SAM_datasets_blur_png_{kernel}")
    os.makedirs(output_dir, exist_ok=True)

# Get list of files from input folder
image_files = sorted([f for f in os.listdir(folder) if f.lower().endswith('.png')])

# Process each image file
for image_file in image_files:
    src_path = os.path.join(folder, image_file)
    base_name = os.path.splitext(image_file)[0]
    
    # Create copies with different kernel size names
    for kernel in kernel_sizes:
        new_name = f"{base_name}_{kernel}.png"
        dst_path = os.path.join(output_base, f"SAM_datasets_blur_png_{kernel}", new_name)
        shutil.copy2(src_path, dst_path)
        print(f"Created {new_name} in {kernel} folder")

print("All files copied and renamed with different kernel sizes successfully!")







