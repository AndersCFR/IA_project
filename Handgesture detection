import cv2
import os
import sys
import glob
import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
from tensorflow.python.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras import optimizers
from tensorflow.python.keras.models import Sequential
from tensorflow.python.keras.layers import Dropout, Flatten, Dense, Activation
from tensorflow.python.keras.layers import Convolution2D, MaxPooling2D
from tensorflow.python.keras import backend as k

from tensorflow.python.keras import applications

from keras.preprocessing.image import load_img, img_to_array
from keras.models import load_model

from sklearn.metrics import confusion_matrix
from mlxtend.plotting import plot_confusion_matrix

img_source = cv2.imread("dataset_imagenes/train/papel/4Bimc2E5E9jTh1Fh.png")
ejemplo_img_papel = cv2.cvtColor(img_source, cv2.COLOR_BGR2RGB)

ancho = ejemplo_img_papel.shape[1]
alto = ejemplo_img_papel.shape[0]

plt.imshow(ejemplo_img_papel)
plt.title('Imagen Original Gesto Papel')
plt.show()

print("Ancho de la imagen: {}".format(ancho))
print("Alto de la imagen: {}".format(alto))

num_img_papel_train=len(glob.glob("dataset_imagenes/train/papel/*.png"))
num_img_papel_test=len(glob.glob("dataset_imagenes/test/papel/*.png"))

num_img_roca_train=len(glob.glob("dataset_imagenes/train/roca/*.png"))
num_img_roca_test=len(glob.glob("dataset_imagenes/test/roca/*.png"))

num_img_tijera_train=len(glob.glob("dataset_imagenes/train/tijera/*.png"))
num_img_tijera_test=len(glob.glob("dataset_imagenes/test/tijera/*.png"))

num_train = num_img_tijera_train
num_test = num_img_tijera_test

print('\n Número de imágenes para entrenamiento: {}'.format(num_train))
print(' Número de imágenes para test: {} \n'.format(num_test))

x = np.array(['papel','roca','tijera'])
y = np.array([num_img_papel_train,num_img_roca_train,num_img_tijera_train])
x2 = np.array(['papel ','roca ','tijera '])
y2 = np.array([num_img_papel_test,num_img_roca_test,num_img_tijera_test])


plt.bar(x,y,color="orange",align="center")
plt.bar(x2,y2,color="g",align="center")
plt.title("Cantidad de imágenes de todo el dataset")
plt.legend(["Entrenamiento", "Evaluación"], loc = 'best')
plt.show()

n_imgs = [num_img_roca_train,num_img_roca_test]
nombres = ["Train","Test"]
plt.pie(n_imgs, labels=nombres, autopct="%0.1f %%")
plt.axis("equal")
plt.title("Proporciones de Train y Test")
plt.show()

k.clear_session()
data_entrenamiento = "dataset_imagenes/train"
data_test = "dataset_imagenes/test"



k.clear_session()
entrenamiento_datagen = ImageDataGenerator(
    rescale=1./255,
    shear_range=0.3,
    zoom_range=0.3,
    horizontal_flip=True
)
test_datagen = ImageDataGenerator(
    rescale=1./255
)

altura, longitud = 100,100
pasos = 1000
pasos_validacion = 200
filtrosConv1 = 32
filtrosConv2 = 64
filtrosConv3 = 128
tamano_filtro1 = (3,3)
tamano_filtro2 = (2,2)
tamano_pool = (2,2)
clases = 3
lr = 0.001
batch_size=10 #número de imágenes a procesar en cada época

imagen_entrenamiento = entrenamiento_datagen.flow_from_directory(
    data_entrenamiento,
    target_size=(altura, longitud),
    batch_size=batch_size,
    #color_mode='grayscale',
    class_mode='categorical'
)

imagen_test = test_datagen.flow_from_directory(
    data_test,
    target_size=(altura, longitud),
    batch_size=batch_size,
    #color_mode='grayscale',
    class_mode='categorical'
)

# definición de la arquitectura del modelo

epocas = 25
cnn = Sequential()
opt = optimizers.Adam(learning_rate=lr)

cnn.add(Convolution2D(filtrosConv1, tamano_filtro1, padding ="same", input_shape=(longitud, altura, 3), activation='relu'))
cnn.add(MaxPooling2D(pool_size=tamano_pool))

cnn.add(Convolution2D(filtrosConv2, tamano_filtro1, padding ="same", activation='relu'))
cnn.add(MaxPooling2D(pool_size=tamano_pool))

cnn.add(Convolution2D(filtrosConv2, tamano_filtro1, padding ="same",activation='relu'))
cnn.add(MaxPooling2D(pool_size=tamano_pool))

cnn.add(Convolution2D(filtrosConv2, tamano_filtro2, padding ="same",activation='relu'))
cnn.add(MaxPooling2D(pool_size=tamano_pool))

cnn.add(Convolution2D(filtrosConv3, tamano_filtro1, padding ="same",activation='relu'))
cnn.add(MaxPooling2D(pool_size=tamano_pool))

cnn.add(Convolution2D(filtrosConv3, tamano_filtro1,padding ="same", activation='relu'))
cnn.add(MaxPooling2D(pool_size=tamano_pool))

cnn.add(Flatten())
cnn.add(Dense(256, activation='relu')) 
cnn.add(Dropout(0.5))
cnn.add(Dense(clases, activation='softmax'))
cnn.summary()

#fase de entrenamiento
opt = optimizers.Adam(learning_rate=lr)
cnn.compile(loss='categorical_crossentropy',optimizer=opt,metrics=['accuracy'])

history = cnn.fit(imagen_entrenamiento, steps_per_epoch=100, epochs=epocas, validation_data=imagen_test,validation_steps=20)

dir = './modelo'

if not os.path.exists(dir):
    os.mkdir(dir)

cnn.save('./modelo/modelo.h5')
cnn.save_weights('./modelo/pesos.h5')

#impresión de resultados

# History variables
loss_train= history.history['loss']
loss_test = history.history['val_loss']

acc_train = history.history['accuracy']
acc_test = history.history['val_accuracy']

epochs_plot = range(1,epocas+1)

# Gráfica de acc vs epochs
plt.plot(acc_train, 'r', label='Training accuracy')
plt.plot(acc_test,'y', label='Test accuracy')
plt.xticks(epochs_plot)
plt.title('Accuracy para Entrenamiento y Prueba')
plt.xlabel('Epochs')
plt.ylabel('ACC')
plt.legend()
plt.show()

# Gráfica de loss vs epochs
plt.plot(loss_train, 'g', label='Training loss')
plt.plot(loss_test,'b', label='Validation loss')
plt.xticks(epochs_plot)
plt.title('Loss para Entrenamiento y Prueba')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()
plt.show()


#Confusion Matrix and Classification Report
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from mlxtend.plotting import plot_confusion_matrix
import seaborn as sns

Y_pred = cnn.predict(imagen_test)
y_pred = np.argmax(Y_pred, axis=1)

print('Confusion Matrix')
plot_confusion_matrix(confusion_matrix(imagen_test.classes, y_pred))

#print('Classification Report')
#target_names = ['Cats', 'Dogs']
#print(classification_report(validation_generator.classes, y_pred, target_names=target_names))

#Poniendo a prueba el modelo
modelo_entrenado = './modelo/modelo.h5'
pesos = './modelo/pesos.h5'
red = load_model(modelo_entrenado)
red.load_weights(pesos)

def predict(file):
    img_source = cv2.imread(file)
    ejemplo_img_papel = cv2.cvtColor(img_source, cv2.COLOR_BGR2RGB)
    plt.imshow(ejemplo_img_papel)
    
    flag = False
    x=load_img(file, target_size=(longitud, altura))
    x=img_to_array(x)
    x=np.expand_dims(x,axis=0)
    arreglo = cnn.predict(x) #[1,0,0]
    
    resultado=arreglo[0] 
    #print('Arreglo :',type(resultado),'  ', arreglo)

    for i in range(0,3):
        #print(' - ',i,' - ',resultado[i])
        if resultado[i]>0.7:
            flag=True
           
    if flag==True:        
        respuesta = np.argmax(resultado)
        if respuesta==0:
            print("Papel")
        elif respuesta == 1:
            print("Roca")
        elif respuesta == 2:
            print("Tijera")
    else:
        respuesta=3
        print("Nada")
    #return respuesta

predict('dataset_imagenes/probando/1.png')
predict('dataset_imagenes/probando/2.png')

# mis propias fotografías
predict('dataset_imagenes/probando/4.jpeg')

#cualquier imagen

predict('dataset_imagenes/probando/web1.jpg')

'''
Problemas No Resuelto - Inquietudes
• Learning Rate como factor crítico

• Transfer Learning incompatibilidad de capas

• Como manejar las predicciones nulas para implementación en tiempo real • Manejos de pines y pulsos hasta esperar recepción de cámara es la raspberry

• Capas muy profundas no funcionaron sobre este proyecto • Tamaño de la imagen/velocidad/calidad de entrenamiento

Conclusiones
• La relación de batch size y número de pasos afectan a la velocidad del aprendizaje pero no a la calidad del aprendizaje • El tener capas altamente densas no garantiza buenos resultados, lo que lo hace es el anpalisis correcto de que capa utilizar y con que densidad y función de activación. • Las redes neuronales convolucionales son muy importantes y permiten realizar aplicaciones en todas las áreas. • El número de impagenes para entrenamiento es muy importante se debe realizar data aucmentation • Los datos deben estar limpios y equilibrados para conseguir un buen resultado
'''

