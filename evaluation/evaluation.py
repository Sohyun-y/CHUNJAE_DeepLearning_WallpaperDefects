from sklearn.metrics import precision_score, recall_score, accuracy_score, f1_score, confusion_matrix
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

class Evaluation:
    def __init__(self):
        self.y_pred = None
        self.y_true = None


    def get_prediction(self, eval_model, val_ds):
        pred = eval_model.predict(val_ds)
        self.y_pred = np.argmax(pred, axis=1)  # 클래스 인덱스로 변환

        self.y_true = []  # 실제 라벨
        try:
            for images, labels in val_ds:
                self.y_true.extend(labels.numpy())
        except:
            pass
    
        df = pd.DataFrame(data=[self.y_pred, self.y_true]).T
        df.columns = ['pred', 'true']
        m = df['pred'] == df['true']
        df['result'] = m

        return df, self.y_pred, self.y_true


    def get_score(self):
        precision = precision_score(self.y_true, self.y_pred, average='macro')
        recall = recall_score(self.y_true, self.y_pred, average='macro')
        accuracy = accuracy_score(self.y_true, self.y_pred)
        f1 = f1_score(self.y_true, self.y_pred, average='macro')

        return precision, recall, accuracy, f1


    def get_confusion_matrix(self):
        conf_matrix = confusion_matrix(self.y_true, self.y_pred)

        # Confusion Matrix를 히트맵으로 시각화
        plt.figure(figsize=(10, 10))
        sns.heatmap(conf_matrix, annot=True, cmap='Blues', fmt='d')
        plt.xlabel('Predicted Labels')
        plt.ylabel('True Labels')
        plt.title('Confusion Matrix')

        plt.show()


    def get_visualization(self, history):
        # train, validation 손실 & 정확도 추출
        train_loss =  history.history['loss']
        val_loss =  history.history['val_loss']
        train_accuracy =  history.history['accuracy']
        val_accuracy =  history.history['val_accuracy']

        # 손실 그래프
        epochs = range(1, len(train_loss) + 1)
        plt.figure(figsize=(12, 6))
        plt.subplot(1, 2, 1)
        plt.plot(epochs, train_loss, label='Training loss')
        plt.plot(epochs, val_loss, label='Validation loss')
        plt.title('Training and Validation Loss')
        plt.xlabel('Epochs')
        plt.ylabel('Loss')
        plt.legend()

        # 정확도 그래프
        plt.subplot(1, 2, 2)
        plt.plot(epochs, train_accuracy, label='Training accuracy')
        plt.plot(epochs, val_accuracy, label='Validation accuracy')
        plt.title('Training and Validation Accuracy')
        plt.xlabel('Epochs')
        plt.ylabel('Accuracy')
        plt.legend()

        plt.show()