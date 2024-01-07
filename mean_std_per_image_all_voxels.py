import SimpleITK as sitk
import numpy as np
import os

def resample_itk_image_LA(itk_image):
    # Get original spacing and size
    original_spacing = itk_image.GetSpacing()
    original_size = itk_image.GetSize()
    
    out_spacing = (1,1,1)

    # Calculate new size
    out_size = [
        int(np.round(original_size[0] * (original_spacing[0] / out_spacing[0]))),
        int(np.round(original_size[1] * (original_spacing[1] / out_spacing[1]))),
        int(np.round(original_size[2] * (original_spacing[2] / out_spacing[2])))
    ]

    # Instantiate resample filter with properties
    resample = sitk.ResampleImageFilter()
    resample.SetOutputSpacing(out_spacing)
    resample.SetSize(out_size)
    resample.SetOutputDirection(itk_image.GetDirection())
    resample.SetOutputOrigin(itk_image.GetOrigin())
    resample.SetTransform(sitk.Transform())
    resample.SetDefaultPixelValue(itk_image.GetPixelIDValue())
    resample.SetInterpolator(sitk.sitkNearestNeighbor)

    # Execute resampling
    resampled_image = resample.Execute(itk_image)
    return resampled_image


# Path to the directory containing .nii.gz files
data_directory = r'C:\My_Data\M2M Data\data\data_2\three_vendor\GE MEDICAL SYSTEMS\LA'   # Replace with the path to your data directory

# Initialize lists to store per-image mean and standard deviation
means_per_image = []
stds_per_image = []

# Approach 1: Calculate mean and standard deviation for each image separately
for filename in os.listdir(data_directory):
    if filename.endswith('.nii.gz'):
        file_path = os.path.join(data_directory, filename)
        img = sitk.ReadImage(file_path)
        img = resample_itk_image_LA(img)
        data = sitk.GetArrayFromImage(img)
        
        # Calculate mean and standard deviation for each image
        mean_value = np.mean(data)
        std_deviation = np.std(data)
        
        means_per_image.append(mean_value)
        stds_per_image.append(std_deviation)

# Approach 2: Calculate mean and standard deviation across all voxel values for the entire dataset
voxel_values = []

for filename in os.listdir(data_directory):
    if filename.endswith('.nii.gz'):
        file_path = os.path.join(data_directory, filename)
        img = sitk.ReadImage(file_path)
        img = resample_itk_image_LA(img)
        data = sitk.GetArrayFromImage(img)
        
        # Flatten and append voxel values to the list
        voxel_values.extend(data.ravel())

# Calculate overall mean and standard deviation across all voxel values
mean_voxel_based = np.mean(voxel_values)
std_voxel_based = np.std(voxel_values)

# Calculate mean and standard deviation across all images for Approach 1
mean_image_based = np.mean(means_per_image)
std_image_based = np.mean(stds_per_image)

# Output the calculated values
print(f"Mean (Approach 1 - per image): {mean_image_based}")
print(f"Standard Deviation (Approach 1 - per image): {std_image_based}")
print(f"Mean (Approach 2 - voxel-based): {mean_voxel_based}")
print(f"Standard Deviation (Approach 2 - voxel-based): {std_voxel_based}")

# Compare the per-image statistics with voxel-based statistics
print("\nComparison between per-image statistics and voxel-based statistics:")
print(f"Diff in mean: {mean_voxel_based - mean_image_based}")
print(f"Diff in std: {std_voxel_based - std_image_based}")
