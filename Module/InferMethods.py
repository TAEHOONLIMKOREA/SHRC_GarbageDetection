from Module.TrainMethods import read_image
import cv2
import numpy as np

import tensorflow as tf
import matplotlib.pyplot as plt

NUM_CLASSES = 6

# label:color_rgb:parts:actions
# [0] background:0,0,0::
# [1] garbage:255,53,94::
# [2] ground:245,147,49::
# [3] robot:51,221,255::
# [4] sea:61,61,245::
# [5] ship:250,250,55::



def infer(model, image_tensor):
    predictions = model.predict(np.expand_dims((image_tensor), axis=0))
    predictions = np.squeeze(predictions)
    predictions = np.argmax(predictions, axis=2)
    print("Predictions shape:", predictions.shape)  # 출력 크기 확인
    print("Unique values in predictions:", np.unique(predictions))  # 예측 값 확인
    return predictions


def decode_segmentation_masks(mask, colormap, n_classes):
    r = np.zeros_like(mask).astype(np.uint8)
    g = np.zeros_like(mask).astype(np.uint8)
    b = np.zeros_like(mask).astype(np.uint8)
    for i in range(0, n_classes):
        idx = mask == i
        r[idx] = colormap[i, 0]
        g[idx] = colormap[i, 1]
        b[idx] = colormap[i, 2]
    rgb = np.stack([r,g,b], axis=2)
    return rgb

def get_overlay(image, colored_mask):
    image = tf.keras.preprocessing.image.array_to_img(image)
    image = np.array(image).astype(np.uint8)
    overlay = cv2.addWeighted(image, 0.35, colored_mask, 0.65, 0)
    return overlay


def plot_samples_matplotlib(display_list, count, figsize=(5,3)):
    _, axes = plt.subplots(nrows=1, ncols=len(display_list), figsize=figsize)
    for i in range(len(display_list)):
        if display_list[i].shape[-1] == 3:
            axes[i].imshow(tf.keras.preprocessing.image.array_to_img(display_list[i]))
        else:
            axes[i].imshow(display_list[i])
    plt.savefig("result" + str(count))
    

def plot_predictions(images_list, colormap, model):
    count = 0
    for image_file in images_list:
        image_tensor = read_image(image_file)
        prediction_mask = infer(image_tensor=image_tensor, model=model)
        prediction_colormap = decode_segmentation_masks(prediction_mask, colormap, NUM_CLASSES)
        plt.imsave("prediction_mask_colormap.png", prediction_colormap)
        overlay = get_overlay(image_tensor, prediction_colormap)
        plot_samples_matplotlib([image_tensor, overlay, prediction_colormap], count, figsize=(18, 14))
        count += 1

def create_colormap(labelmap_path):
    colormap = []
    with open(labelmap_path, 'r') as file:
        for line in file:
            line = line.strip()
            if line.startswith('#') or not line:  # 주석 또는 빈 줄 무시
                continue
            parts = line.split(':')
            rgb = tuple(map(int, parts[1].split(',')))  # RGB 값 추출
            colormap.append(rgb)
    
    # NumPy 배열로 변환
    result = np.array(colormap, dtype=np.uint8)
    print("Colormap shape:", result.shape)
    print("Colormap values:", result[:NUM_CLASSES])

    return result 
        

