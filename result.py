import librosa
import numpy as np
import tensorflow as tf
import os

model = tf.keras.models.load_model('models//cnn 正常 learningrate 8.7e-9.h5')

persons = os.listdir('dataset//正常')

def load_data(data_path):
    y1, sr1 = librosa.load(data_path, duration=4)
    ps = librosa.feature.melspectrogram(y=y1, sr=sr1)
    ps = ps[np.newaxis, ..., np.newaxis]
    return ps


def infer(audio_path):
    data = load_data(audio_path)
    result = model.predict(data)
    lab = tf.argmax(result, 1)
    return lab


if __name__ == '__main__':
    list_path='dataset//test//'
    children = os.listdir(list_path)
    ans=0
    length=0
    for child in children:
        #label=infer(list_path+child)
        #print(persons[int(str(label)[11])])
        ans_0=0
        datas=os.listdir(list_path+child)
        length+=len(datas)
        for data in datas:
            label = infer(list_path+child+'//'+data)
            print(label)
            print('音频：%s 的预测结果标签为：%s' % (data, persons[int(str(label)[11])]))
            if(persons[int(str(label)[11])]==child):
                ans+=1
                ans_0+=1
        print(ans_0/len(datas))
    print("综合准确率为"+str(ans/length))
