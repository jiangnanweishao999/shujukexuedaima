import os
import librosa
import numba
def get_data_list(audio_path, list_path):
    sound_sum = 0
    persons = os.listdir(audio_path)
    print(persons)
    f_train = open(os.path.join(list_path, 'train_list.txt'), 'w')
    f_test = open(os.path.join(list_path, 'test_list.txt'), 'w')

    for i in range(len(persons)):
        sounds = os.listdir(os.path.join("dataset\\正常", persons[i]))
        for sound in sounds:
            sound_path = os.path.join("dataset\\正常", persons[i], sound)
            t = librosa.get_duration(filename=sound_path)
            if t >= 3:
                if sound_sum % 100 == 0:
                    f_test.write('%s\t%s\n' % (sound_path, i))
                else:
                    f_train.write('%s\t%s\n' % (sound_path, i))
                sound_sum += 1

    f_test.close()
    f_train.close()
   
if __name__ == '__main__':
    get_data_list('dataset/正常', 'dataset')
