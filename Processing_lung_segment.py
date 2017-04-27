import numpy as np  # linear algebra
import dicom
import os
import scipy.ndimage
import matplotlib.pyplot as plt
import pandas as pd
from skimage import measure, morphology
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
import datetime
import math

# Some constants
INPUT_FOLDER = '/home/wisdom/deeplearningdata/kaggle_2017_cancer/stage1/'
patients = os.listdir(INPUT_FOLDER)
patients.sort()
patients_1 = patients[:1*math.ceil(len(patients)/10)]
patients_2 = patients[1*math.ceil(len(patients)/10):2*math.ceil(len(patients)/10)]
patients_3 = patients[2*math.ceil(len(patients)/10):3*math.ceil(len(patients)/10)]
patients_4 = patients[3*math.ceil(len(patients)/10):4*math.ceil(len(patients)/10)]
patients_5 = patients[4*math.ceil(len(patients)/10):5*math.ceil(len(patients)/10)]
patients_6 = patients[5*math.ceil(len(patients)/10):6*math.ceil(len(patients)/10)]
patients_7 = patients[6*math.ceil(len(patients)/10):7*math.ceil(len(patients)/10)]
patients_8 = patients[7*math.ceil(len(patients)/10):8*math.ceil(len(patients)/10)]
patients_9 = patients[8*math.ceil(len(patients)/10):9*math.ceil(len(patients)/10)]
patients_10 = patients[9*math.ceil(len(patients)/10):]

patients_list = [patients_1,patients_2,patients_3,patients_4,patients_5,patients_6,patients_7,patients_8,patients_9,patients_10]

labels = pd.read_csv('/home/wisdom/PycharmProjects/kaggle_2017_cancer/input/stage1_labels.csv', index_col=0)
much_data = []
# Load the scans in given folder path
def load_scan(path):
    slices = [dicom.read_file(path + '/' + s) for s in os.listdir(path)]
    slices.sort(key=lambda x: float(x.ImagePositionPatient[2]))
    try:
        slice_thickness = np.abs(slices[0].ImagePositionPatient[2] - slices[1].ImagePositionPatient[2])
    except:
        slice_thickness = np.abs(slices[0].SliceLocation - slices[1].SliceLocation)

    for s in slices:
        s.SliceThickness = slice_thickness

    return slices

def get_pixels_hu(slices):
    image = np.stack([s.pixel_array for s in slices])
    # Convert to int16 (from sometimes int16),
    # should be possible as values should always be low enough (<32k)
    image = image.astype(np.int16)

    # Set outside-of-scan pixels to 0
    # The intercept is usually -1024, so air is approximately 0
    image[image == -2000] = 0

    # Convert to Hounsfield units (HU)
    for slice_number in range(len(slices)):

        intercept = slices[slice_number].RescaleIntercept
        slope = slices[slice_number].RescaleSlope

        if slope != 1:
            image[slice_number] = slope * image[slice_number].astype(np.float64)
            image[slice_number] = image[slice_number].astype(np.int16)

        image[slice_number] += np.int16(intercept)

    return np.array(image, dtype=np.int16)

def resample(image, scan, new_spacing=[1,1,1]):
    # Determine current pixel spacing
    spacing = np.array([scan[0].SliceThickness] + scan[0].PixelSpacing, dtype=np.float32)

    resize_factor = spacing / new_spacing
    new_real_shape = image.shape * resize_factor
    new_shape = np.round(new_real_shape)
    real_resize_factor = new_shape / image.shape
    new_spacing = spacing / real_resize_factor

    image = scipy.ndimage.interpolation.zoom(image, real_resize_factor, mode='nearest')

    return image, new_spacing

# Lung segmentation
def largest_label_volume(im, bg=-1):
    vals, counts = np.unique(im, return_counts=True)

    counts = counts[vals != bg]
    vals = vals[vals != bg]

    if len(counts) > 0:
        return vals[np.argmax(counts)]
    else:
        return None

def segment_lung_mask(image, fill_lung_structures=True):
    # not actually binary, but 1 and 2.
    # 0 is treated as background, which we do not want
    binary_image = np.array(image > -320.00, dtype=np.int8) + 1
    labels = measure.label(binary_image)
    # Pick the pixel in the very corner to determine which label is air.
    #   Improvement: Pick multiple background labels from around the patient
    #   More resistant to "trays" on which the patient lays cutting the air
    #   around the person in half
    background_label = labels[0, 0, 0]
    # Fill the air around the person
    binary_image[background_label == labels] = 2

    # Method of filling the lung structures (that is superior to something like
    # morphological closing)
    if fill_lung_structures:
        # For every slice we determine the largest solid structure
        for i, axial_slice in enumerate(binary_image):
            # axial_slice = axial_slice - 1  #it will be minus 1 in next section so disable here, Jin
            # labeling = measure.label(axial_slice) # specifically set background=0 as below, but not necessary, Jin
            labeling = measure.label(axial_slice, background=0)
            # l_max = largest_label_volume(labeling, bg=0) # have to change to bg=-1 as shown below, Jin
            l_max = largest_label_volume(labeling, bg=-1)
            if l_max is not None:  # This slice contains some lung
                binary_image[i][labeling != l_max] = 1

    binary_image -= 1  # Make the image actual binary
    binary_image = 1 - binary_image  # Invert it, lungs are now 1

    # Remove other air pockets insided body
    labels = measure.label(binary_image, background=0)
    # l_max = largest_label_volume(labels, bg=0) # have to change to bg=-1 as shown below, Jin
    l_max = largest_label_volume(labels, bg=-1)
    if l_max is not None:  # There are air pockets
        binary_image[labels != l_max] = 0

    return binary_image

#for i, patients in enumerate(patients_list):
#    if i >3:
#       for num, patient in enumerate(patients):
for num, patient in enumerate(patients_8):
    t1 = datetime.datetime.now()
    try:
        label = labels.get_value(patient, 'cancer')
        if label == 1:
            label = np.array([0, 1])
        elif label == 0:
            label = np.array([1, 0])
        patient = load_scan(INPUT_FOLDER + patient)
        patient_pixels = get_pixels_hu(patient)
        pix_resampled, _ = resample(patient_pixels, patient, [1, 1, 1])
        segmented_lungs = segment_lung_mask(pix_resampled, False)
        segmented_lungs_fill = segment_lung_mask(pix_resampled, True)
        lungs_structures = segmented_lungs_fill - segmented_lungs
        #lungs_structures_3D_image = lungs_structures.transpose(2, 1, 0)
        #patient_lungs_structures_3D_image=zero_center(normalize(lungs_structures_3D_image))

        # print(img_data.shape,label)
        much_data.append([lungs_structures, label])
        print('The translation patient number is:' + str(num))
        t2 = datetime.datetime.now()
        print('The used time is:' + str(t2-t1))
    except KeyError as e:
        print('KeyError data is:' + str(num))
        print('KeyError patient is:' + str(patient))

#np.save('./preprocessing_model/muchdata_{}.npy'.format(str(i+1)), much_data)
np.save('./preprocessing_model/muchdata_{}.npy'.format(str(8)), much_data)