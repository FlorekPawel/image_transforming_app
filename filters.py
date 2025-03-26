import numpy as np
from PIL import Image

def grayscale(image, *args):
    image_array = np.array(image).astype(float)

    if image_array.ndim == 2:
        return image
    
    grayscale = np.dot(image_array[..., :3], [0.2989, 0.5870, 0.1140])
    grayscale_uint8 = np.clip(grayscale, 0, 255).astype(np.uint8)
    return Image.fromarray(grayscale_uint8)

def binary(image, threshold):
    image_array = np.array(grayscale(image))    
    binary_image = np.where(image_array > threshold, 255, 0)
    return Image.fromarray(binary_image.astype(np.uint8))
        
def brightness_correction(image, value):
    image_array = np.array(image).astype(float)
    brightness_increased = np.clip(image_array + value, 0, 255).astype(np.uint8)
    return Image.fromarray(brightness_increased)

def contrast_correction(image, value):
    image_array = np.array(image).astype(float)
    contrast_corrected = np.clip(value * (image_array - 128) + 128, 0, 255).astype(np.uint8)
    return Image.fromarray(contrast_corrected)

def negative_filter(image, *args):
    image_array = np.array(image).astype(float)
    negative_image = np.clip(255 - image_array, 0, 255).astype(np.uint8)
    return Image.fromarray(negative_image)

def binarisation(image, threshold):
    image_array = np.array(image).astype(float)
    binary_image = np.where(image_array > threshold, 255, 0).astype(np.uint8)
    return Image.fromarray(binary_image)

def averaging(image, kernel_size):
    image_array = np.array(image).astype(float)
    
    kernel_size = int(kernel_size)
    kernel = np.ones((kernel_size, kernel_size)) / (kernel_size * kernel_size)
    
    if image_array.ndim == 3:
        padded_image = np.pad(image_array, ((kernel_size // 2, kernel_size // 2), 
                                            (kernel_size // 2, kernel_size // 2), 
                                            (0, 0)), 
                              mode='constant', constant_values=0)
        averaged_image = np.zeros_like(image_array)
        
        for i in range(image_array.shape[0]):
            for j in range(image_array.shape[1]):
                for k in range(image_array.shape[2]):
                    averaged_image[i, j, k] = np.sum(padded_image[i:i+kernel_size, j:j+kernel_size, k] * kernel)
    else:
        padded_image = np.pad(image_array, ((kernel_size // 2, kernel_size // 2), 
                                            (kernel_size // 2, kernel_size // 2)), 
                              mode='constant', constant_values=0)
        averaged_image = np.zeros_like(image_array)
        
        for i in range(image_array.shape[0]):
            for j in range(image_array.shape[1]):
                averaged_image[i, j] = np.sum(padded_image[i:i+kernel_size, j:j+kernel_size] * kernel)
    
    return Image.fromarray(np.clip(averaged_image, 0, 255).astype(np.uint8))

def median(image, kernel_size):
    image_array = np.array(image).astype(float)
    kernel_size = int(kernel_size)

    if image_array.ndim == 3:
        padded_image = np.pad(image_array, ((kernel_size // 2, kernel_size // 2), 
                                            (kernel_size // 2, kernel_size // 2), 
                                            (0, 0)), 
                              mode='constant', constant_values=0)
        median_filtered_image = np.zeros_like(image_array)
        
        for i in range(image_array.shape[0]):
            for j in range(image_array.shape[1]):
                for k in range(image_array.shape[2]):
                    median_filtered_image[i, j, k] = np.median(padded_image[i:i+kernel_size, j:j+kernel_size, k])
    else: 
        padded_image = np.pad(image_array, ((kernel_size // 2, kernel_size // 2), 
                                            (kernel_size // 2, kernel_size // 2)), 
                              mode='constant', constant_values=0)
        median_filtered_image = np.zeros_like(image_array)
        
        for i in range(image_array.shape[0]):
            for j in range(image_array.shape[1]):
                region = padded_image[i:i+kernel_size, j:j+kernel_size]
                median_filtered_image[i, j] = np.median(region)
    
    return Image.fromarray(np.clip(median_filtered_image, 0, 255).astype(np.uint8))

def gaussian(image, sigma):
    def gaussian_kernel(size, sigma):
        ax = np.linspace(-(size // 2), size // 2, size)
        kernel = np.exp(-0.5 * (ax / sigma) ** 2)
        kernel = np.outer(kernel, kernel)
        return kernel / np.sum(kernel)

    image_array = np.array(image).astype(float)
    
    size = int(6 * sigma) | 1
    kernel = gaussian_kernel(size, sigma)
    
    if image_array.ndim == 3:
        padded_image = np.pad(image_array, ((size // 2, size // 2), 
                                            (size // 2, size // 2), 
                                            (0, 0)), 
                              mode='reflect')
        blurred_image = np.zeros_like(image_array)

        for i in range(image_array.shape[0]):
            for j in range(image_array.shape[1]):
                for k in range(image_array.shape[2]):
                    blurred_image[i, j, k] = np.sum(padded_image[i:i+size, j:j+size, k] * kernel)
    else:
        padded_image = np.pad(image_array, ((size // 2, size // 2), 
                                            (size // 2, size // 2)), 
                              mode='reflect')
        blurred_image = np.zeros_like(image_array)

        for i in range(image_array.shape[0]):
            for j in range(image_array.shape[1]):
                blurred_image[i, j] = np.sum(padded_image[i:i+size, j:j+size] * kernel)
    
    return Image.fromarray(np.clip(blurred_image, 0, 255).astype(np.uint8))

def sharpening(image, value):
    image_array = np.array(image).astype(float)
    
    kernel = np.array([[0, -1, 0], [-1, value, -1], [0, -1, 0]])

    if image_array.ndim == 3:
        padded_image = np.pad(image_array, ((1, 1), (1, 1), (0, 0)), mode='constant', constant_values=0)
        sharpened_image = np.zeros_like(image_array)

        for i in range(image_array.shape[0]):
            for j in range(image_array.shape[1]):
                for k in range(image_array.shape[2]):
                    sharpened_image[i, j, k] = np.sum(padded_image[i:i+3, j:j+3, k] * kernel)

    else:
        padded_image = np.pad(image_array, ((1, 1), (1, 1)), mode='constant', constant_values=0)
        sharpened_image = np.zeros_like(image_array)

        for i in range(image_array.shape[0]):
            for j in range(image_array.shape[1]):
                sharpened_image[i, j] = np.sum(padded_image[i:i+3, j:j+3] * kernel)

    sharpened_image = np.clip(sharpened_image, 0, 255)

    return Image.fromarray(sharpened_image.astype(np.uint8))

def roberts(image, *args):
    image_array = np.array(image).astype(float)

    if image_array.ndim == 3:
        image_array = np.array(grayscale(image)).astype(float)
    
    kernel_x = np.array([[1, 0], [0, -1]])
    kernel_y = np.array([[0, 1], [-1, 0]])
    
    padded_image = np.pad(image_array, ((1, 1), (1, 1)), mode='constant', constant_values=0)
    edges = np.zeros_like(image_array)

    for i in range(image_array.shape[0]):
        for j in range(image_array.shape[1]):
            gx = np.sum(padded_image[i:i+2, j:j+2] * kernel_x)
            gy = np.sum(padded_image[i:i+2, j:j+2] * kernel_y)
            edges[i, j] = np.sqrt(gx**2 + gy**2)

    return Image.fromarray(np.clip(edges, 0, 255).astype(np.uint8))

def sobel(image, *args):
    image_array = np.array(image).astype(float)

    if image_array.ndim == 3:
        image_array = np.array(grayscale(image)).astype(float)    
    kernel_x = np.array([[1, 0, -1], [2, 0, -2], [1, 0, -1]])
    kernel_y = np.array([[1, 2, 1], [0, 0, 0], [-1, -2, -1]])
    
    padded_image = np.pad(image_array, ((1, 1), (1, 1)), mode='constant', constant_values=0)
    edges = np.zeros_like(image_array)

    for i in range(image_array.shape[0]):
        for j in range(image_array.shape[1]):
            gx = np.sum(padded_image[i:i+3, j:j+3] * kernel_x)
            gy = np.sum(padded_image[i:i+3, j:j+3] * kernel_y)
            edges[i, j] = np.sqrt(gx**2 + gy**2)

    return Image.fromarray(np.clip(edges, 0, 255).astype(np.uint8))

def high_pass(image, value):
    image_array = np.array(image).astype(float)
    if image_array.ndim == 3:
        image_array = np.array(grayscale(image)).astype(float) 

    kernel = np.array([
        [-1, -1, -1],
        [-1,  value, -1],
        [-1, -1, -1]
    ])

    padded_image = np.pad(image_array, ((1, 1), (1, 1)), mode='reflect')
    filtered_image = np.zeros_like(image_array)

    for i in range(image_array.shape[0]):
        for j in range(image_array.shape[1]):
            region = padded_image[i:i+3, j:j+3]
            filtered_image[i, j] = np.sum(region * kernel)

    return Image.fromarray(np.clip(filtered_image, 0, 255).astype(np.uint8))

def laplace(image, *args):
    image_array = np.array(image).astype(float)

    if image_array.ndim == 3:
        image_array = np.array(grayscale(image)).astype(float)  
    kernel = np.array([
        [0,  1,  0],
        [1, -4,  1],
        [0,  1,  0]
    ])

    padded_image = np.pad(image_array, ((1, 1), (1, 1)), mode='reflect')
    filtered_image = np.zeros_like(image_array)

    for i in range(image_array.shape[0]):
        for j in range(image_array.shape[1]):
            region = padded_image[i:i+3, j:j+3]
            filtered_image[i, j] = np.sum(region * kernel)

    return Image.fromarray(np.clip(filtered_image, 0, 255).astype(np.uint8))

def prewitt(image, *args):
    image_array = np.array(image).astype(float)

    if image_array.ndim == 3:
        image_array = np.array(grayscale(image)).astype(float) 
        
    kernel_x = np.array([
        [-1, 0, 1],
        [-1, 0, 1],
        [-1, 0, 1]
    ])

    kernel_y = np.array([
        [-1, -1, -1],
        [ 0,  0,  0],
        [ 1,  1,  1]
    ])

    padded_image = np.pad(image_array, ((1, 1), (1, 1)), mode='reflect')
    filtered_image_x = np.zeros_like(image_array)
    filtered_image_y = np.zeros_like(image_array)

    for i in range(image_array.shape[0]):
        for j in range(image_array.shape[1]):
            region = padded_image[i:i+3, j:j+3]
            filtered_image_x[i, j] = np.sum(region * kernel_x)
            filtered_image_y[i, j] = np.sum(region * kernel_y)

    filtered_image = np.sqrt(filtered_image_x**2 + filtered_image_y**2)
    return Image.fromarray(np.clip(filtered_image, 0, 255).astype(np.uint8))


def kuwahara(image, kernel_size):
    image_array = np.array(image).astype(float)
    kernel_size = int(kernel_size)
    offset = kernel_size // 2

    if image_array.ndim == 3:
        padded_image = np.pad(image_array, ((offset, offset), (offset, offset), (0, 0)), mode='reflect')
        filtered_image = np.zeros_like(image_array)

        for i in range(image_array.shape[0]):
            for j in range(image_array.shape[1]):
                for k in range(image_array.shape[2]):
                    regions = [
                        padded_image[i:i+offset+1, j:j+offset+1, k],
                        padded_image[i:i+offset+1, j+offset:j+kernel_size, k],
                        padded_image[i+offset:i+kernel_size, j:j+offset+1, k],
                        padded_image[i+offset:i+kernel_size, j+offset:j+kernel_size, k]
                    ]
                    means = [np.mean(region) for region in regions]
                    variances = [np.var(region) for region in regions]
                    min_variance_index = np.argmin(variances)
                    filtered_image[i, j, k] = means[min_variance_index]
    else:
        padded_image = np.pad(image_array, ((offset, offset), (offset, offset)), mode='reflect')
        filtered_image = np.zeros_like(image_array)

        for i in range(image_array.shape[0]):
            for j in range(image_array.shape[1]):
                regions = [
                    padded_image[i:i+offset+1, j:j+offset+1],
                    padded_image[i:i+offset+1, j+offset:j+kernel_size],
                    padded_image[i+offset:i+kernel_size, j:j+offset+1],
                    padded_image[i+offset:i+kernel_size, j+offset:j+kernel_size]
                ]
                means = [np.mean(region) for region in regions]
                variances = [np.var(region) for region in regions]
                min_variance_index = np.argmin(variances)
                filtered_image[i, j] = means[min_variance_index]

    return Image.fromarray(np.clip(filtered_image, 0, 255).astype(np.uint8))

def ridge(image, *args):
    image_array = np.array(grayscale(image)).astype(float)
    kernel_x = np.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]])
    kernel_y = np.array([[-1, -2, -1], [0, 0, 0], [1, 2, 1]])
    
    padded_image = np.pad(image_array, ((1, 1), (1, 1)), mode='reflect')
    filtered_image_x = np.zeros_like(image_array)
    filtered_image_y = np.zeros_like(image_array)

    for i in range(image_array.shape[0]):
        for j in range(image_array.shape[1]):
            filtered_image_x[i, j] = np.sum(padded_image[i:i+3, j:j+3] * kernel_x)
            filtered_image_y[i, j] = np.sum(padded_image[i:i+3, j:j+3] * kernel_y)

    filtered_image = np.sqrt(filtered_image_x**2 + filtered_image_y**2)
    return Image.fromarray(np.clip(filtered_image, 0, 255).astype(np.uint8))

def scharr(image, *args):
    image_array = np.array(grayscale(image)).astype(float)
    kernel_x = np.array([
        [3, 0, -3],
        [10, 0, -10],
        [3, 0, -3]
    ])
    kernel_y = np.array([
        [3, 10, 3],
        [0, 0, 0],
        [-3, -10, -3]
    ])
    
    padded_image = np.pad(image_array, ((1, 1), (1, 1)), mode='reflect')
    filtered_image_x = np.zeros_like(image_array)
    filtered_image_y = np.zeros_like(image_array)

    for i in range(image_array.shape[0]):
        for j in range(image_array.shape[1]):
            filtered_image_x[i, j] = np.sum(padded_image[i:i+3, j:j+3] * kernel_x)
            filtered_image_y[i, j] = np.sum(padded_image[i:i+3, j:j+3] * kernel_y)

    filtered_image = np.sqrt(filtered_image_x**2 + filtered_image_y**2)
    return Image.fromarray(np.clip(filtered_image, 0, 255).astype(np.uint8))
