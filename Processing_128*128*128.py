import numpy as np  # linear algebra
import dicom
import os
import scipy.ndimage
import matplotlib.pyplot as plt
import pandas as pd
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
import datetime
import cv2
import math
from skimage import measure

# Some constants
INPUT_FOLDER = '/home/wisdom/deeplearningdata/kaggle_2017_cancer/stage1/'
#INPUT_FOLDER = '/home/wisdom/deeplearningdata/kaggle_2017_cancer/stage2/'
patients = os.listdir(INPUT_FOLDER)
patients.sort()
labels = pd.read_csv('./input/stage1_labels.csv', index_col=0)
#labels = pd.read_csv('./input/stage1_solution.csv', index_col=0)
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

def resample(image, spacing, new_spacing=[1,1,1]):
    # Determine current pixel spacing

    resize_factor = spacing / new_spacing
    new_real_shape = image.shape * resize_factor
    new_shape = np.round(new_real_shape)
    real_resize_factor = new_shape / image.shape

    image = scipy.ndimage.interpolation.zoom(image, real_resize_factor, mode='nearest')

    return image

def plot_3d(image, threshold=-300):
    # Position the scan upright,
    # so the head of the patient would be at the top facing the camera
    p = image.transpose(2, 1, 0)

    verts, faces = measure.marching_cubes(p, threshold)

    fig = plt.figure(figsize=(10, 10))
    ax = fig.add_subplot(111, projection='3d')

    # Fancy indexing: `verts[faces]` to generate a collection of triangles
    mesh = Poly3DCollection(verts[faces], alpha=0.70)
    face_color = [0.45, 0.45, 0.75]
    mesh.set_facecolor(face_color)
    ax.add_collection3d(mesh)

    ax.set_xlim(0, p.shape[0])
    ax.set_ylim(0, p.shape[1])
    ax.set_zlim(0, p.shape[2])

    plt.show()

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
            l_max = largest_label_volume(labeling, bg=0) # have to change to bg=-1 as shown below, Jin
            # l_max = largest_label_volume(labeling, bg=-1)
            if l_max is not None:  # This slice contains some lung
                binary_image[i][labeling != l_max] = 1

    binary_image -= 1  # Make the image actual binary
    binary_image = 1 - binary_image  # Invert it, lungs are now 1

    # Remove other air pockets insided body
    labels = measure.label(binary_image, background=0)
    l_max = largest_label_volume(labels, bg=0) # have to change to bg=-1 as shown below, Jin
    # l_max = largest_label_volume(labels, bg=-1)
    if l_max is not None:  # There are air pockets
        binary_image[labels != l_max] = 0

    return binary_image

# Normalization
MIN_BOUND = -1000.0
MAX_BOUND = 400.0

def normalize(image):
    image = (image - MIN_BOUND) / (MAX_BOUND - MIN_BOUND)
    image[image > 1] = 1.
    image[image < 0] = 0.
    return image

# Zero centering
PIXEL_MEAN = 0.25

def zero_center(image):
    image = image - PIXEL_MEAN
    return image

HM_SLICES = 128
IMG_PX_SIZE=128
def chunks(l, n):
    count=0
    for i in range(0, len(l), n):
        if(count < HM_SLICES):
            yield l[i:i + n]
            count=count+1

def mean(a):
    return sum(a) / len(a)

def resize_image(slices, img_px_size=IMG_PX_SIZE, hm_slices=HM_SLICES):

    new_slices = []
    slices = [cv2.resize(each_slice, (img_px_size, img_px_size)) for each_slice in slices]

    chunk_sizes = math.ceil(len(slices) / hm_slices)
    for slice_chunk in chunks(slices, chunk_sizes):
        slice_chunk = list(map(mean, zip(*slice_chunk)))
        new_slices.append(slice_chunk)

    while len(new_slices) < hm_slices:
        new_slices.append(new_slices[-1])

    while len(new_slices) > hm_slices:
        new_val = list(map(mean, zip(*[new_slices[hm_slices - 1], new_slices[hm_slices], ])))
        del new_slices[hm_slices]
        new_slices[hm_slices - 1] = new_val

    return np.array(new_slices)

t3 = datetime.datetime.now()
for num, patient in enumerate(patients):
    t1 = datetime.datetime.now()
    try:
        label_num = labels.get_value(patient, 'cancer')
        if label_num == 1:
            label = np.array([0, 1])
        elif label_num == 0:
            label = np.array([1, 0])
        patient = load_scan(INPUT_FOLDER + patient)
        patient_pixels = get_pixels_hu(patient)

        spacing = np.array([patient[0].SliceThickness] + patient[0].PixelSpacing, dtype=np.float32)
        new_spacing = []
        new_spacing.append(patient_pixels.shape[0] * spacing[0] / HM_SLICES)
        new_spacing.append(patient_pixels.shape[1] * spacing[1] / HM_SLICES)
        new_spacing.append(patient_pixels.shape[2] * spacing[2] / HM_SLICES)
        pix_resampled= resample(patient_pixels, spacing, new_spacing)
        segmented_lungs = segment_lung_mask(pix_resampled, False)
        segmented_lungs_fill = segment_lung_mask(pix_resampled, True)
        #lungs_structures = segmented_lungs_fill - segmented_lungs
        #plot_3d(lungs_structures, 0)

        #much_data.append([lungs_structures, label])
        much_data.append([segmented_lungs_fill, label])
        if label_num == 1:
            much_data.append([segmented_lungs_fill, label])
        print('The translation patient number is:' + str(num))
        t2 = datetime.datetime.now()
        print('The used time is:' + str(t2-t1))
    except KeyError as e:
        print('KeyError data is:' + str(num))
        print('KeyError patient is:' + str(patient))

np.save('./preprocessing_model/muchdata_lungs_fill_add_cancer_128*128*128.npy', much_data)
#np.save('./preprocessing_model/muchdata_lungs_fill_128*128*128.npy', much_data)
#np.save('./preprocessing_model/muchdata_lungs_structures_128*128*128_test.npy', much_data)

t4 = datetime.datetime.now()
print('The start time is:' + str(t3))
print('The end time is:' + str(t4))
print('The used time is:' + str(t4-t3))
