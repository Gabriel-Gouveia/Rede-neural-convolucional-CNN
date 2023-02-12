import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import zipfile

path = 'personagens.zip'
zip_object = zipfile.ZipFile(file=path, mode='r')
zip_object.extractall('./')
zip_object.close()

gerador_treinamento = ImageDataGenerator(rescale=1./255,
                                        rotation_range=7,
                                        horizontal_flip=True,
                                        zoom_range=0.2)



base_treinamento = gerador_treinamento.flow_from_directory('training_set',
                                                            target_size = (64,64),
                                                            batch_size = 8,
                                                            class_mode = 'categorical')




gerador_teste = ImageDataGenerator(rescale=1./255)


base_teste = gerador_teste.flow_from_directory('test_set',
                                                            target_size = (64,64),
                                                            batch_size = 8,
                                                            class_mode = 'categorical',
                                                            shuffle = False)


rede_neural = Sequential()
rede_neural.add(Conv2D(32, (3,3), input_shape = (64,64,3), activation='relu'))
rede_neural.add(MaxPooling2D(pool_size=(2,2)))


rede_neural.add(Conv2D(32, (3,3), activation='relu'))
rede_neural.add(MaxPooling2D(pool_size=(2,2)))

rede_neural.add(Flatten())

rede_neural.add(Dense(units = 4, activation='relu'))
rede_neural.add(Dense(units = 4, activation='relu'))
rede_neural.add(Dense(units = 2, activation='softmax'))



rede_neural.compile(optimizer='adam', loss='categorical_crossentropy',
                    metrics = ['accuracy'])


rede_neural.fit_generator(base_treinamento, 
epochs=100, 
validation_data=base_teste)


previsoes = rede_neural.predict(base_teste)
print(previsoes)


import numpy as np
previsoes2 = np.argmax(previsoes, axis = 1)
previsoes2

from sklearn.metrics import accuracy_score
accuracy_score(previsoes2, base_teste.classes)

