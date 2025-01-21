from Module.TrainMethods import data_generator, read_image, train
from Module.InferMethods import plot_predictions, create_colormap

from scipy.io import loadmat
import numpy as np
import os
from glob import glob
import tensorflow as tf
from tensorflow import keras
import matplotlib.pyplot as plt
import json
import re

NUM_CLASSES = 6
DATA_DIR = "./Data/TrainDataSet"
NUM_TRAIN_IMAGES = 650
NUM_VAL_IMAGES = 50


def main():    
    # [1] 훈련용 데이터셋 준비
    train_images = sorted(glob(os.path.join(DATA_DIR, 'Images/*')))[:NUM_TRAIN_IMAGES]
    train_masks = sorted(glob(os.path.join(DATA_DIR, 'SegmentationClass/*')))[:NUM_TRAIN_IMAGES]
    val_images = sorted(glob(os.path.join(DATA_DIR, 'Images/*')))[NUM_TRAIN_IMAGES:NUM_VAL_IMAGES+NUM_TRAIN_IMAGES]
    val_masks = sorted(glob(os.path.join(DATA_DIR, 'SegmentationClass/*')))[NUM_TRAIN_IMAGES:NUM_VAL_IMAGES+NUM_TRAIN_IMAGES]

    # numbers = []
    # for mask_path in train_masks:
    #     filename = os.path.basename(mask_path)
    #     # 정규식으로 끝에 있는 숫자 추출
    #     match = re.search(r'(\d+)(?=\.[^\.]+$)', filename)
    #     number = match.group(1)
    #     number = int(number)
    #     numbers.append(number)
    # numbers.sort()

    # count = 1
    # result = []
    # for value in numbers:
    #     if value == count:
    #         count += 1
    #     else:
    #         while count != value:
    #             result.append(count)
    #             print(count)
    #             count += 1
    #         count += 1

    label_path = "./Data/TrainDataSet/labelmap.txt"
    train_dataset = data_generator(train_images, train_masks, label_path)
    val_dataset = data_generator(val_images, val_masks, label_path)


    
    # 출력된 클래스 ID가 0 ~ NUM_CLASSES - 1 범위인지 확인
    # for img, mask in val_dataset.take(1):
    #     print("Image shape:", img.shape)
    #     print("Mask shape:", mask.shape)
    #     print("Unique labels in mask:", tf.unique(tf.reshape(mask, [-1]))[0].numpy())

    # # 데이터 확인 (1개의 배치만 확인)    
    # for images, masks in train_dataset.take(1):  # 첫 번째 배치만 가져옴
    #     # 배치 크기만큼 반복하여 이미지와 마스크 출력
    #     for i in range(16):
    #         # 이미지 복원 (정규화된 값을 다시 0~255로 변환)
    #         image = (images[i].numpy() + 1.0) * 127.5
    #         image = image.astype('uint8')  # 정수형 변환
            
    #         # 마스크는 정수형 그대로 사용
    #         mask = masks[i].numpy()

    #         # 시각화
    #         plt.figure(figsize=(10, 5))

    #         # 원본 이미지
    #         plt.subplot(1, 2, 1)
    #         plt.imshow(image)
    #         plt.title("Image")
    #         plt.axis("off")

    #         # 마스크
    #         plt.subplot(1, 2, 2)
    #         plt.imshow(mask, cmap="jet")  # 마스크는 색상으로 표시
    #         plt.title("Mask")
    #         plt.axis("off")

    #         plt.savefig("test")

    #     break  # 첫 번째 배치까지만 확인
    
    # [2-1] 학습  
    # model = train(train_dataset, val_dataset)
    
    # [2-2] 모델 불러오기
    model = keras.models.load_model('model_2025-01-21 16:53:49.h5')

    # [3] 추론 
    colormap = create_colormap(label_path)
    # colormap = colormap * 100
    # colormap = colormap.astype(np.uint8)
    
    print("Colormap shape:", colormap.shape)
    print("Colormap values:", colormap[:NUM_CLASSES])

    plot_predictions(train_images[:4], colormap, model=model)
    
    
    

if __name__ == '__main__':
    main()