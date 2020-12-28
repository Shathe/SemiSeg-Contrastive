import cv2
import numpy as np

'''
Functions for the frequency domain (fourier)
Functionality allows to insert the specify low frequency of one sample into another sample's frequency
'''


def get_low_frequency_mask(img, beta = 0.05):
    """
    :param img: image from which take the shape
    :param beta: percentage of the low frequency to take in (central crop in the frequency domain)
    :return: returns a mask which has zero values on high frequencies and ones on low frequencies
    (low and high frequencies depends on the bet parameter)
    """
    rows, cols = img.shape
    crow, ccol = int(rows / 2), int(cols / 2)
    radius_row, radius_col = int(rows * beta  / 2), int(cols * beta  / 2)
    low_freq_mask = np.zeros((rows, cols, 2), np.uint8)
    low_freq_mask[crow - radius_row:crow + radius_row, ccol - radius_col:ccol + radius_col] = 1

    return low_freq_mask

def get_high_frequency_mask(img, beta = 0.05):
    """

    :param img: image from which take the shape
    :param beta: percentage of the low frequency to take out (central crop in the frequency domain)
    :return: returns a mask which has ones values on high frequencies and zeros on low frequencies
    (low and high frequencies depends on the bet parameter)
    """
    rows, cols = img.shape
    crow, ccol = int(rows / 2), int(cols / 2)
    radius_row, radius_col = int(rows * beta  / 2), int(cols * beta  / 2)
    high_freq_mask = np.ones((rows, cols, 2), np.uint8)
    high_freq_mask[crow - radius_row:crow + radius_row, ccol - radius_col:ccol + radius_col] = 0

    return high_freq_mask


def remove_low_freq(img, beta = 0.05):
    """

    :param img: (1-channel) image from which to remove the low frequencies
    :param beta: threshold on the frequencies
    :return: the same input image but without  low frequencies
    """
    source_fourier_center = np.fft.fftshift(cv2.dft(np.float32(img), flags=cv2.DFT_COMPLEX_OUTPUT))
    highPass_mask = get_high_frequency_mask(img, beta)

    high_frequencies = highPass_mask * source_fourier_center # get high frequencies from the image only

    resulting_image =  np.fft.ifftshift(high_frequencies) # undo fourier
    resulting_image = cv2.idft(resulting_image)
    resulting_image = cv2.magnitude(resulting_image[:, :, 0], resulting_image[:, :, 1])

    # normalize it
    resulting_image = np.abs(resulting_image)
    resulting_image -= resulting_image.min()
    resulting_image = resulting_image * 255 / resulting_image.max()
    resulting_image = resulting_image.astype(np.uint8)

    return resulting_image

def change_low_freq(source_LF, target_LF, beta = 0.05):
    """

    :param source_LF: (1-channel) image from which to take the low frequencies
    :param target_LF: (1-channel) image from which to take the high frequencies
    :param beta: threshold on the frequencies
    :return: an image which have low frequencies of source_LF and high frequencies of target_LF
    """
    source_fourier_center = np.fft.fftshift(cv2.dft(np.float32(source_LF), flags=cv2.DFT_COMPLEX_OUTPUT))
    target_fourier_center = np.fft.fftshift(cv2.dft(np.float32(target_LF), flags=cv2.DFT_COMPLEX_OUTPUT))
    lowPass_mask = get_low_frequency_mask(source_LF, beta) # low frequencies
    highPass_mask = get_high_frequency_mask(target_LF, beta)

    low_frequencies = lowPass_mask * source_fourier_center # get low frequencies from source
    high_frequencies = highPass_mask * target_fourier_center # get high frequencies from target
    combined_frequencies = low_frequencies + high_frequencies # combined them


    resulting_image =  np.fft.ifftshift(combined_frequencies) # undo fourier
    resulting_image = cv2.idft(resulting_image)
    resulting_image = cv2.magnitude(resulting_image[:, :, 0], resulting_image[:, :, 1])

    # normalize it
    resulting_image = np.abs(resulting_image)
    resulting_image -= resulting_image.min()
    resulting_image = resulting_image * 255 / resulting_image.max()
    resulting_image = resulting_image.astype(np.uint8)

    return resulting_image


def remove_low_freq_rgb(img, beta = 0.05):
    """

    :param img: (1-channel) image from which to remove the low frequencies
    :param beta: threshold on the frequencies
    :return: an image which have low frequencies of source_LF and high frequencies of target_LF
    """
    b = remove_low_freq(img[...,0], beta)
    g = remove_low_freq(img[...,1], beta)
    r = remove_low_freq(img[...,2], beta)
    result = np.zeros_like(img)
    result[..., 0] = b
    result[..., 1] = g
    result[..., 2] = r

    return result


def change_low_freq_rgb(source_LF, target_LF, beta = 0.05):
    """

    :param source_LF: (3-channel) image from which to take the low frequencies
    :param target_LF: (3-channel) image from which to take the high frequencies
    :param beta: threshold on the frequencies
    :return: the same input image but without  low frequencies
    """
    b = change_low_freq(source_LF[...,0], target_LF[...,0], beta)
    g = change_low_freq(source_LF[...,1], target_LF[...,1], beta)
    r = change_low_freq(source_LF[...,2], target_LF[...,2], beta)
    result = np.zeros_like(source_LF)
    result[..., 0] = b
    result[..., 1] = g
    result[..., 2] = r

    return result

