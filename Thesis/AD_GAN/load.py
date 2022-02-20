import os
import shutil
import nibabel as nib
import numpy as np
import pandas as pd
from tqdm import tqdm
import csv

# Imports
train_name = '/Users/jeffroszell/Documents/School/Thesis/GH_ML_Learning/LearningML/Thesis/' \
             'AD_GAN/csv/train_path.csv'
test_name = '/Users/jeffroszell/Documents/School/Thesis/GH_ML_Learning/LearningML/Thesis/' \
            'AD_GAN/csv/test_path.csv'
counter = 0

file_path = train_name
new_image_path = '/Users/jeffroszell/Documents/School/Thesis/GH_ML_Learning/LearningML/Thesis/' \
                 'AD_GAN/image_files/training_dataset_images'

nc_filepath = '/Users/jeffroszell/Documents/School/Thesis/GH_ML_Learning/LearningML/Thesis/' \
              'AD_GAN/image_files/AD'
ad_filepath = '/Users/jeffroszell/Documents/School/Thesis/GH_ML_Learning/LearningML/Thesis/' \
              'AD_GAN/image_files/NC'

# Collecting the NifTi filepaths
paths = ['AD_Update', 'CN_Update']

with open(train_name, 'w', newline='') as new_train_file:
    train_w = csv.writer(new_train_file)
    with open(test_name, 'w', newline='') as new_test_file:
        test_w = csv.writer(new_test_file)
        for path in paths:
            img_path = os.path.join('/Users/jeffroszell/Documents/School/Thesis/GH_ML_Learning/LearningML/Thesis/'
                                    'AD_GAN/orig_data', path)
            for path, dirs, files, in os.walk(img_path):
                for file in files:
                    if file.endswith('.nii') and not file.startswith('._'):
                        img_path = os.path.join(path, file)
                        if counter % 5 == 0 or 'T2' in path:
                            test_w.writerow([img_path])
                        else:
                            train_w.writerow([img_path])
                        counter += 1

# Pulling in NifTi image data and splitting them into slices
img_paths = pd.read_csv(file_path, header=None)


def read_save_nifti_file(filepath, name):
    scan = nib.load(filepath)
    image = scan.get_fdata()
    image = np.squeeze(image)

    height, width, depth = image.shape

    image_1 = image[round(height / 2) - 10:round(height / 2) + 10, :, :]
    image_2 = image[:, round(width / 2) - 10:round(width / 2) + 10, :]
    image_3 = image[:, :, round(depth / 2) - 10:round(depth / 2) + 10]

    # Save 20 center slices of 3 different views for each subject
    for i in range(20):
        im_1 = image_1[i, :, :]
        im_2 = image_2[:, i, :]
        im_3 = image_3[:, :, i]

        filename_1 = name + '_' + filepath.split('/')[-1].split('.')[0] + str(i) + '1' + '.nii'
        filename_2 = name + '_' + filepath.split('/')[-1].split('.')[0] + str(i) + '2' + '.nii'
        filename_3 = name + '_' + filepath.split('/')[-1].split('.')[0] + str(i) + '3' + '.nii'

        im_1 = nib.Nifti1Image(im_1, scan.affine, scan.header)
        im_2 = nib.Nifti1Image(im_2, scan.affine, scan.header)
        im_3 = nib.Nifti1Image(im_3, scan.affine, scan.header)

        nib.save(im_1, os.path.join(new_image_path, filename_1))
        nib.save(im_2, os.path.join(new_image_path, filename_2))
        nib.save(im_3, os.path.join(new_image_path, filename_3))


for ind in tqdm(range(len(img_paths))):
    path = img_paths.iloc[ind, 0]
    if '/AD_Update' in path:
        if 'siemens_3' in path.lower():
            read_save_nifti_file(path, 'AD_siemens_3')
        if 'siemens_15' in path.lower():
            read_save_nifti_file(path, 'AD_siemens_15')
        if 'philips_3' in path.lower():
            read_save_nifti_file(path, 'AD_philips_3')
        if 'philips_15' in path.lower():
            read_save_nifti_file(path, 'AD_philips_15')
        if 'ge_3' in path.lower():
            read_save_nifti_file(path, 'AD_GE_3')
        if 'ge_15' in path.lower():
            read_save_nifti_file(path, 'AD_GE_15')

    if '/CN_Update' in path:
        if 'siemens_3' in path.lower():
            read_save_nifti_file(path, 'NC_siemens_3')
        if 'siemens_15' in path.lower():
            read_save_nifti_file(path, 'NC_siemens_15')
        if 'philips_3' in path.lower():
            read_save_nifti_file(path, 'NC_philips_3')
        if 'philips_15' in path.lower():
            read_save_nifti_file(path, 'NC_philips_15')
        if 'ge_3' in path.lower():
            read_save_nifti_file(path, 'NC_GE_3')
        if 'ge_15' in path.lower():
            read_save_nifti_file(path, 'NC_GE_15')

# Sorting into the AD and NC folders for processing
files = os.listdir(new_image_path)

for f in files:
    # move if NC
    if 'NC_' in f:
        shutil.move(os.path.join(new_image_path, f), os.path.join(nc_filepath, f))
    # move if AD
    elif 'AD_' in f:
        shutil.move(os.path.join(new_image_path, f), os.path.join(ad_filepath, f))
    # notify if something else
    else:
        print('Could not categorize file with name %s' % f)
