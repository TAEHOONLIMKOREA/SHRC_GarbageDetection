import matplotlib.pyplot as plt
import numpy as np
import random
import tensorflow as tf
import os
import re

# 출력된 클래스 ID가 0 ~ NUM_CLASSES - 1 범위인지 확인하는 함수
def validate_mask_classes(val_dataset):
    for img, mask in val_dataset.take(1):
        print("Image shape:", img.shape)
        print("Mask shape:", mask.shape)
        print("Unique labels in mask:", tf.unique(tf.reshape(mask, [-1]))[0].numpy())

# 이미지와 마스크 개수 비교 후 부족한 넘버를 출력해주는 함수
def validate_count_of_images_and_masks(train_masks):
    numbers = []
    for mask_path in train_masks:
        filename = os.path.basename(mask_path)
        # 정규식으로 끝에 있는 숫자 추출
        match = re.search(r'(\d+)(?=\.[^\.]+$)', filename)
        number = match.group(1)
        number = int(number)
        numbers.append(number)
    numbers.sort()

    count = 1
    result = []
    for value in numbers:
        if value == count:
            count += 1
        else:
            while count != value:
                result.append(count)
                print(count)
                count += 1
            count += 1

def display_images_and_masks(dataset):
    # train_dataset의 각 배치에서 무작위로 하나의 이미지와 마스크 선택하여 시각화
    # 16개의 배치에서 무작위로 하나씩 뽑아 출력 및 저장
    selected_images = []
    selected_masks = []

    for batch_index, (images, masks) in enumerate(dataset.take(16)):  # 최대 16개의 배치 처리
        batch_size = images.shape[0]  # 배치 크기 확인

        # 배치에서 무작위로 하나 선택
        random_index = random.randint(0, batch_size - 1)

        # 선택된 이미지와 마스크 저장
        image = (images[random_index].numpy() + 1.0) * 127.5
        image = image.astype('uint8')  # 정수형 변환

        mask = masks[random_index].numpy()

        selected_images.append(image)
        selected_masks.append(mask)

    # 16개의 이미지와 마스크를 하나의 플롯에 출력
    fig, axes = plt.subplots(4, 8, figsize=(20, 10))  # 4x8 그리드 (이미지 4x2, 마스크 4x2)
    for i in range(16):
        row = i // 4
        col = (i % 4) * 2

        # 이미지 출력
        axes[row, col].imshow(selected_images[i])
        axes[row, col].set_title(f"Image {i+1}")
        axes[row, col].axis("off")

        # 마스크 출력
        axes[row, col + 1].imshow(selected_masks[i], cmap="jet")
        axes[row, col + 1].set_title(f"Mask {i+1}")
        axes[row, col + 1].axis("off")

    plt.tight_layout()
    plt.savefig("selected_images_and_masks.png")
    plt.show()
    
    
    
def calculate_random_accuracy(dataset, model, save_path):
    """
    무작위로 선택한 50장의 데이터에 대한 모델의 예측 정확도를 계산하고, 결과를 그래프와 함께 파일로 저장하는 함수

    Args:
        dataset (tf.data.Dataset): (image, mask) 쌍을 포함한 데이터셋
        model (tf.keras.Model): 학습된 세그멘테이션 모델
        save_path (str): 결과 그래프를 저장할 경로

    Returns:
        float: 무작위로 선택한 50장의 평균 정확도
    """
    total_accuracy = 0
    num_samples = 0
    sample_accuracies = []  # 무작위 샘플별 정확도 저장

    # 데이터셋을 리스트로 변환 후 무작위로 50개 선택
    dataset_list = list(dataset.unbatch().as_numpy_iterator())
    random_samples = np.random.choice(len(dataset_list), size=50, replace=False)

    for idx in random_samples:
        image, true_mask = dataset_list[idx]

        # 모델 예측 수행
        prediction = model.predict(np.expand_dims(image, axis=0))
        prediction = np.argmax(prediction, axis=-1)[0]  # 채널 차원에서 argmax 후 첫 번째 결과 사용

        # Flatten하여 accuracy_score 계산
        accuracy = accuracy_score(true_mask.flatten(), prediction.flatten())
        total_accuracy += accuracy
        sample_accuracies.append(accuracy)

        num_samples += 1

    # 평균 정확도 계산
    average_accuracy = total_accuracy / num_samples

    # 그래프 생성
    plt.figure(figsize=(10, 6))
    plt.plot(sample_accuracies, marker='o', label="Sample Accuracy")
    plt.axhline(y=average_accuracy, color='r', linestyle='--', label=f"Average Accuracy: {average_accuracy:.2f}")
    plt.title("Random Sample-wise Accuracy and Average Accuracy")
    plt.xlabel("Sample Index")
    plt.ylabel("Accuracy")
    plt.legend()
    plt.grid(True)

    # 그래프 파일로 저장
    current_time = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    file_name = f"{save_path.split('.')[0]}_{current_time}.png"
    plt.savefig(file_name)
    plt.close()

    print(f"Accuracy results saved to {file_name}")

    return average_accuracy

