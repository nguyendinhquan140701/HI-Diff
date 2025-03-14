import cv2
import os
import numpy as np

# import os
# import cv2

# # Define folder paths
# input_folder = "E:\Project\HI-Diff\SAM_datasets"
# output_folder = "E:\Project\HI-Diff\SAM_datasets_blur_png_5"
# name_reference_folder = "E:\Project\HI-Diff\\test-45-input-realBlur-raw"

# # Ensure output folder exists
# os.makedirs(output_folder, exist_ok=True)

# # Get list of .bmp images from input folder (in original order)
# image_files = [f for f in os.listdir(input_folder) if f.lower().endswith('.bmp')]

# # Get list of reference filenames (in original order)
# reference_files = [f for f in os.listdir(name_reference_folder)]

# # Check if the number of images matches the reference filenames
# if len(image_files) != len(reference_files):
#     print(f"Error: Mismatch in file count! {len(image_files)} images, {len(reference_files)} reference names.")
# else:
#     # Process each image
#     for image_file, new_name in zip(image_files, reference_files):
#         image_path = os.path.join(input_folder, image_file)
#         new_name = os.path.splitext(new_name)[0] + ".png"  # Ensure extension is .png
#         output_path = os.path.join(output_folder, new_name)

#         # Read the image
#         image = cv2.imread(image_path)

#         if image is None:
#             print(f"Skipping {image_file}: Unable to read.")
#             continue

#         # Apply Gaussian blur
#         blurred_image = cv2.GaussianBlur(image, (5, 5), 0)

#         # Save as PNG with new name
#         cv2.imwrite(output_path, blurred_image)

#         print(f"Processed {image_file} -> {output_path}")

#     print("Blurring and renaming completed successfully!")




# Define folder paths
input_folder = "E:\Project\HI-Diff\SAM_datasets"
output_folder = "E:\Project\HI-Diff\SAM_datasets_blur_png_5_2"
name_reference_folder = "E:\Project\HI-Diff\\test-45-input-realBlur-raw"

# Ensure output folder exists
os.makedirs(output_folder, exist_ok=True)

# Get list of .bmp images and sort them
image_files = sorted([f for f in os.listdir(input_folder) if f.lower().endswith('.bmp')])

# Get list of reference filenames and sort them
reference_files = sorted([f for f in os.listdir(name_reference_folder)])

# Check if the number of images matches the reference filenames
if len(image_files) != len(reference_files):
    print(f"Error: Mismatch in file count! {len(image_files)} images, {len(reference_files)} reference names.")
else:
    # Process each image
    for image_file, new_name in zip(image_files, reference_files):
        image_path = os.path.join(input_folder, image_file)
        new_name = os.path.splitext(new_name)[0] + ".png"  # Ensure extension is .png
        output_path = os.path.join(output_folder, new_name)

        # Read the image
        image = cv2.imread(image_path)

        if image is None:
            print(f"Skipping {image_file}: Unable to read.")
            continue

        # Apply Gaussian blur
        blurred_image = cv2.GaussianBlur(image, (5, 5), 0)

        # Save as PNG with new name
        cv2.imwrite(output_path, blurred_image)

        print(f"Processed {image_file} -> {output_path}")

    print("Blurring and renaming completed successfully!")




# # Define input and output folder paths
# input_folder = "E:\Project\HI-Diff\SAM_datasets_blur_9"
# output_folder = "E:\Project\HI-Diff\SAM_datasets_blur_png"

# # Create output folder if it doesn't exist
# os.makedirs(output_folder, exist_ok=True)

# # Get list of all BMP files in the input folder
# bmp_files = sorted([f for f in os.listdir(input_folder) if f.lower().endswith('.bmp')])

# # Process each BMP file with a new naming format
# for index, bmp_file in enumerate(bmp_files, start=1):
#     input_path = os.path.join(input_folder, bmp_file)
#     scene_number = f"001-{index:02d}" 
#     output_path = os.path.join(output_folder, f"scene{scene_number}.png")

#     # Read the BMP image
#     image = cv2.imread(input_path)

#     if image is None:
#         print(f"Skipping {bmp_file}: Unable to read.")
#         continue

#     # Save as PNG with new name
#     cv2.imwrite(output_path, image)

#     print(f"Converted and renamed {bmp_file} -> {output_path}")

# print("Conversion and renaming completed!")

# import os

# # Define folder paths
# source_folder = "E:\Project\HI-Diff\SAM_datasets_blur_png"
# name_reference_folder = "E:\Project\HI-Diff\\test-45-input-realBlur-raw"

# # Get sorted lists of files
# source_files = sorted(os.listdir(source_folder))  # Files to rename
# name_reference_files = sorted(os.listdir(name_reference_folder))  # New names

# # Ensure we have enough filenames
# if len(source_files) != len(name_reference_files):
#     print(f"Error: Mismatch in file count! {len(source_files)} in source, {len(name_reference_files)} in reference.")
# else:
#     for old_name, new_name in zip(source_files, name_reference_files):
#         old_path = os.path.join(source_folder, old_name)
#         new_path = os.path.join(source_folder, new_name)  # Keep the new name

#         # Rename the file
#         os.rename(old_path, new_path)
#         print(f"Renamed: {old_name} -> {new_name}")

#     print("Renaming completed successfully!")
