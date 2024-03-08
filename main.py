from data import Preprocess
from model import CustomModel
from evaluation import Evaluation

if __name__=="__main__":

    # 객체 생성
    preprocess = Preprocess()
    custommodel = CustomModel()
    evaluation = Evaluation()

    # Dataset 전처리
    ### 한글폴더명을 정수로 변경
    directory_path = '/Users/user1/Downloads/papering/dataset/train'
    preprocess.change_folder_name(directory_path)

    ### 파일 개수 세서 데이터프레임에 추가
    papering_df = preprocess.count_files(directory_path)


    ### train/val dataset
    directory_path = '/Users/user1/Downloads/papering/dataset/train'
    labels = 'inferred'  # 디렉토리 구조에 따라 클래스 라벨 부여
    label_mode = 'int'  # 정수형 클래스 라벨 사용
    color_mode = 'rgb'  # RGB 색상 모드로 이미지를 로드
    batch_size = 32
    image_size = (224, 224)  # 사용하려는 모델에 맞춰서 변경
    validation_split = 0.2
    seed = 42

    ### 정규화 안 된 train 데이터셋 로드
    dataset_type = 'training'
    train_ds = preprocess.get_raw_dataset(dataset_type, directory_path, labels, label_mode, color_mode, batch_size, image_size, validation_split, seed)
    ### 정규화 된 train 데이터셋 로드
    train_ds = preprocess.get_norm_dataset(dataset_type, train_ds)

    ### 정규화 안 된 validation 데이터셋 로드
    dataset_type = 'validation'
    val_ds = preprocess.get_raw_dataset(dataset_type, directory_path, labels, label_mode, color_mode, batch_size, image_size, validation_split, seed)
    ### 정규화 된 validation 데이터셋 로드
    val_ds = preprocess.get_norm_dataset(dataset_type, val_ds)


    ### test dataset
    dataset_type = 'test' 
    directory_path = '/Users/user1/Downloads/papering/dataset/test'
    labels = None
    label_mode = None
    color_mode = 'rgb'  # RGB 색상 모드로 이미지를 로드
    batch_size = 32
    image_size = (224, 224)  # 사용하려는 모델에 맞춰서 변경
    validation_split = None
    seed = None

    ### 정규화 안 된 데이터셋 로드
    test_ds = preprocess.get_raw_dataset(dataset_type, directory_path, labels, label_mode, color_mode, batch_size, image_size, validation_split, seed)
    ### 정규화 된 데이터셋 로드
    test_ds = preprocess.get_norm_dataset(dataset_type, test_ds)



    # Custom Model 생성 & 훈련
    model_name = 'VGG16'
    weights = 'imagenet'
    include_top = False
    input_shape = (224, 224, 3)  # 사용하려는 모델에 맞는 input size
    fc_unit_list = [1000]  # 추가할 FClayer들의 유닛을 리스트로 입력
    num_classes = 19  # 분류할 클래스 수
    optimizer = 'adam'
    loss = 'sparse_categorical_crossentropy'
    metrics = 'accuracy'
    epochs=5

    ### 모델 설정
    custommodel.get_basemodel(model_name, weights, include_top, input_shape)
    custommodel.add_layer(fc_unit_list, num_classes)
    model, history = custommodel.get_headmodel(optimizer, loss, metrics, train_ds, val_ds, epochs)



    # 모델 평가 & 시각화
    eval_model = model
    history = history

    ### 예측한 y값과 실제 y값 비교하는 데이터 프레임 생성
    y_df, y_pred, y_true = evaluation.get_prediction(eval_model, val_ds)

    ### precision, recall, accuracy, f1
    precision, recall, accuracy, f1 = evaluation.get_score()
    print(precision, recall, accuracy, f1)

    ### Loss & Accuracy 시각화
    evaluation.get_confusion_matrix()
    evaluation.get_visualization(history)




