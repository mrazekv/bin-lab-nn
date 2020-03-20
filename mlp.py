#!/usr/bin/env python
# coding: utf-8

# # Neuronové sítě
# V tomto cvičení se zaměříme na efektivitu neuronových sítí. Naším úkolem nebude jen se seznámit se základem NN (který možná máte z jiných kurzů), ale 
# 
# 
# ## Práce s neuronovou sítí
# V prvním kroce musíme načíst požadované knihovny. Budeme pracovat s knihovnou [TensorFlow](https://tesorflow.org) a rozhraním Keras, které by mělo tvořit univerzální rozhraní pro práci s NN (i s jinými frameworky jako je např PyTorch). Toto rozhraní je v nové verzi TF 2.0 implicní, ale i ve starších je podporováno. Při načítání můžeme varování týkající se zastaralosti volání funkcí ``numpy`` ignorovat, jsou způsobeny využitím starší verze TensorFlow 1.14.
# 

# In[17]:


import keras
from keras.datasets import mnist
from keras.models import Sequential
import keras.layers as layers
from keras.optimizers import RMSprop


# Knihovny jsou úspěšně načtené. Dalším krokem bude stáhnout dataset MNIST obsahující ručně psaná čísla a který je základním benchmarkem pro NN. Tento dataset je (jako ostatní datasety) rozdělen na trénovací data a testovací data. 

# In[18]:


# the data, split between train and test sets
(x_train, y_train), (x_test, y_test) = mnist.load_data()
print(x_train.shape, x_test.shape)


# Najdeme zde 60 tisíc testovacích a 10 tisíc testovacích obrázků 28x28 pixelů a k tomu odpovídající počet labelů (neboli odpovídajících čísel, které obrázky představují). Bude se tedy jednat o tzv. supervised learning (trénování s učitelem). Tento krátký příklad ukáže, jak data v datasetu vypadají. 


# Nyní se můžeme dát do konstrukce sítě. Deklarujeme si parametry jako jsou počet batchů (dávek) a počet epoch, po které budeme trénovat

# In[20]:


batch_size = 128
epochs = 5


# Trénovací data jsou reprezentovány jako byty (0 - 255). My je převedeme do formátu float čísel 0 - 1 a pro MLP síť překonvertujeme 28x28 obrázky do 1D pole o délce 784.

# In[21]:


x_train = x_train.reshape(60000, 784)
x_test = x_test.reshape(10000, 784)
x_train = x_train.astype('float32')
x_test = x_test.astype('float32')
x_train /= 255
x_test /= 255
print(x_train.shape[0], 'train samples')
print(x_test.shape[0], 'test samples')

# convert class vectors to binary class matrices
y_train = keras.utils.to_categorical(y_train, 10)
y_test = keras.utils.to_categorical(y_test, 10)


# Data jsou připravena. Konečně se dostáváme ke konstrukci požadované sítě. Vytvoříme síť plně propojených vrstev: neuronů, kde každý vezme výstupy všech neuronů v předcházející vrstvě a vynásobí je vahou. 
# ![mlp.png](mpl neuron)
# 
# V tomto případě použijeme síť 784-200-100-10 neuronů, kdy všechny kromě poslední budou používat ReLU aktivační funkce.

# In[22]:


model = Sequential()
model.add(layers.Dense(200, activation='relu', input_shape=(784,)))
model.add(layers.Dense(100, activation='relu'))
model.add(layers.Dense(10, activation='softmax'))

model.summary()


# Nyní můžeme model zkompilovat a spustit trénování. Tato operace může zabrat nějaký čas, zejména pokud nemáte GPU.

# In[23]:


model.compile(loss='categorical_crossentropy',
              optimizer=RMSprop(),
              metrics=['accuracy'])

history = model.fit(x_train, y_train,
                    batch_size=batch_size,
                    epochs=epochs,
                    verbose=1,
                    validation_data=(x_test, y_test))
                    


# Model je nyní natrénovaný (na poměrně malý počet epoch). Nyní jej můžeme uložit nebo použít na jednom konkrétním obrázku. Výstupem je 10 hodnot, kdy každá říká pravděpodobnost toho, že obrázek spadá do určité kategorie. Na tomto příkladě můžeme vidět test čísla 9, kdy správně kategorie 9 má největší pravděpodobnost.



# In[52]:


score = model.evaluate(x_test, y_test, verbose=0)
print('Test loss:', score[0])
print('Test accuracy:', score[1])


# 
# 
