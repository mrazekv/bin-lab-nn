#!/usr/bin/env python
# coding: utf-8

# # Konvoluční neuronové sítě
# 
# 
# 
# Jaké jsou výhody konvolučních sítí? Zkuste si otevřít [https://playground.tensorflow.org/](https://playground.tensorflow.org/). Jedná se o jednoduchou MLP síť, která se snaží klasifikovat data na vstupu. Pokud vybete nalevo jiný dataset, uvidíte, že klasifikace není možná. Pokud ale vstupem MLP sítě bude nějaký jiný příznak než jen x1 a x2 (např kvadráty), po nějakém době se NN natrénuje na požadovaný výstup. V komplexních datech je to ještě složitější. Nešlo by to ale dělat nějak automaticky? K tomu slouží tzv. konvoluční sítě (a hluboké neuronové sítě - DNN). První takovou síť představil v roce 1997 Yann LeChun (síť LeNet) a tento krok se stal milníkem, který změnil pohled na strojové učení. V tomto cvičení se na takovou NN síť podíváme.
# 
# ## Práce s neuronovou sítí
# Předpokládá se, že jste si prošli řešení první [mlp.ipynb](mlp.ipynb) MLP sítě. V této ukázce zpracujeme síť konvoluční ve variantě LeNet.
# 
# V prvním kroce musíme načíst požadované knihovny. Budeme pracovat s knihovnou Tensorflow a rozhraním Keras, které je v nové verzi TF 2.0. Varování týkající se zastaralosti volání funkcí ``numpy`` můžeme ignorovat, jsou způsobeny využitím starší verze TensorFlow 1.14.
# 

# In[1]:


import keras
from keras.datasets import mnist
from keras.models import Sequential
import keras.layers as layers
from keras.optimizers import RMSprop


# Knihovny jsou úspěšně načtené. Dalším krokem bude znovu stáhnout dataset MNIST. Tento dataset je rozdělen na trénovací data a testovací data. 

# In[2]:


# the data, split between train and test sets
(x_train, y_train), (x_test, y_test) = mnist.load_data()
print(x_train.shape, x_test.shape)


# Nyní se můžeme dát do konstrukce sítě. Deklarujeme si parametry jako jsou počet batchů (dávek) a počet epoch, po které budeme trénovat

# In[3]:


batch_size = 128
epochs = 5


# Převedeme trénovací data do formátu float čísel 0 - 1. Zde je první **rozdíl vůči MLP síti**. Vstupní obrázky necháme v rozměru 28x28, ale přidáme ještě jednu dimenzi - barevný kanál (v našem případě je jen 1 barva - takže data se prakticky nezmění).

# In[4]:


# reshape nedelame, protoze chceme, aby se jednalo o obrazky 28x28 kvuli konvoluci
x_train = x_train.reshape(60000, 28, 28, 1)
x_test = x_test.reshape(10000, 28, 28, 1)
x_train = x_train.astype('float32')
x_test = x_test.astype('float32')
x_train /= 255
x_test /= 255
print(x_train.shape[0], 'train samples')
print(x_test.shape[0], 'test samples')

# convert class vectors to binary class matrices
y_train = keras.utils.to_categorical(y_train, 10)
y_test = keras.utils.to_categorical(y_test, 10)
y_test.shape


# Vytvoříme síť konvolučních vrstev následovaných plně propojenými. Každá konvoluce má určeno, jak velký je filtr a kolik těchto filtrů ve výsledku je. Podle nastavení okrajů se buď obrázek zmenší (šířka - velikost filtru + 1) nebo se kraje doplní nulami. Protože jsou výsledné obrázky příznaků (features) moc velké, použije se zmenšení pomocí ```AveragePooling2D``` vrstvy - tato vrstva rozdělí obrázek do čtvrečků 2x2 a vypočítá průměr. Ve svých výpočtech počtu operací tuto operaci ignorujte. V další vrstvě se 6 feature obrázků zase prožene 3x3 filtry. Pozor, na vstupu je 6 kanálů - pro každý filtr a vstupní kanál máme vlastní konvoluční filtr (v druhé vstvě máme tedy 6 * 16 3x3 filtrů), jejichž výstupy se pro všechny kanály sečtou (jak je naznačeno na obrázku)
# ![https://i.stack.imgur.com/uDgke.gif](https://i.stack.imgur.com/uDgke.gif)
# 
# Výstup druhého filtru je zase zmenšen. Pomocí Flatten vrstvy dojde k překódování 2D obrázků o 16 kanálech na 1D vektor příznaků (pouhým přeskládáním) a tento vektor je vstupem do závěrečné části, která je tvořena plně propojenými sítěmi.

# In[5]:


model = Sequential()
model.add(layers.Conv2D(filters=6, kernel_size=(3, 3), activation='relu', input_shape=(28,28, 1)))
model.add(layers.AveragePooling2D())

model.add(layers.Conv2D(filters=16, kernel_size=(3, 3), activation='relu'))
model.add(layers.AveragePooling2D())

model.add(layers.Flatten())

model.add(layers.Dense(units=120, activation='relu'))
model.add(layers.Dense(units=84, activation='relu'))
model.add(layers.Dense(units=10, activation = 'softmax'))

model.summary()


# Nyní můžeme model zkompilovat a spustit trénování. Tato operace může zabrat nějaký čas, zejména pokud nemáte GPU.

# In[6]:


model.compile(loss='categorical_crossentropy',
              optimizer=RMSprop(),
              metrics=['accuracy'])

history = model.fit(x_train, y_train,
                    batch_size=batch_size,
                    epochs=epochs,
                    verbose=1,
                    validation_data=(x_test, y_test))
                    


# Model je nyní natrénovaný (na poměrně malý počet epoch). Nyní jej můžeme uložit nebo použít na jednom konkrétním obrázku

# In[7]:


score = model.evaluate(x_test, y_test, verbose=0)
print('Test loss:', score[0])
print('Test accuracy:', score[1])

