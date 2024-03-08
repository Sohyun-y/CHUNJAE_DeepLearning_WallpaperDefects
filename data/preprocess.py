import pandas as pd
import os
import tensorflow as tf


class Preprocess:
    def __init__(self):
        self.df = pd.DataFrame()


    # 한글 폴더명을 정수로 바꿔주고, 라벨링 어떻게 했는지 데이터프레임 생성
    def change_folder_name(self, directory_path):
        label_list = []
        label_type_list = []

        for i, h in enumerate(os.listdir(directory_path)):
            try:
                os.rename(f'{directory_path}/{h}', f'{directory_path}/{i}')
                label_type_list.append(h)
                label_list.append(i)
            except:
                print(f'"{h}" 폴더명을 변경하는데 실패했습니다.')

        self.df = pd.DataFrame({'하자종류': label_type_list, 'label': label_list})
        return self.df


    # 파일 개수 몇 개인지 데이터프레임에 추가
    def count_files(self, directory_path):
        for folder, label in zip(self.df['하자종류'], self.df['label']):
            path = os.path.join(directory_path, str(label))  # 폴더의 실제 경로
            files = os.listdir(path)

            # 디렉토리 내 파일의 개수를 데이터 프레임에 추가
            self.df.loc[self.df['하자종류'] == folder, 'file_count'] = len(files)
        
        return self.df

    
    # 정규화 안 된 dataset
    def get_raw_dataset(self, dataset_type, directory_path, labels, label_mode, color_mode, batch_size, image_size, validation_split, seed):
        if dataset_type == "training":
            dataset = tf.keras.utils.image_dataset_from_directory(
                directory=directory_path,
                labels=labels,
                label_mode=label_mode,
                color_mode=color_mode,
                batch_size=batch_size,
                image_size=image_size,
                validation_split=validation_split,
                seed=seed,
                subset="training"
            )

        elif dataset_type == "validation":
            dataset = tf.keras.utils.image_dataset_from_directory(
                directory=directory_path,
                labels=labels,
                label_mode=label_mode,
                color_mode=color_mode,
                batch_size=batch_size,
                image_size=image_size,
                validation_split=validation_split,
                seed=seed,
                subset="validation"
            )

        elif dataset_type == "test":
            dataset = tf.keras.utils.image_dataset_from_directory(
                directory=directory_path,
                label_mode=label_mode,
                color_mode=color_mode,
                batch_size=batch_size,
                image_size=image_size
            )

        return dataset


    # 정규화된 dataset
    def get_norm_dataset(self, dataset_type, dataset):
        normalization_layer = tf.keras.layers.Rescaling(1./255)

        if dataset_type == "test":
            norm_dataset = dataset.map(lambda x: normalization_layer(x))
        else:
            norm_dataset = dataset.map(lambda x, y: (normalization_layer(x), y))
        
        return norm_dataset