

import numpy as np

# importing image and label data as numpy array
p=np.load(r'C:/Users/Avinash/Documents/soft_assignment/paper.npy')
pi=np.load(r'C:/Users/Avinash/Documents/soft_assignment/paper_0.npy')
s=np.load(r'C:/Users/Avinash/Documents/soft_assignment/stone.npy')
si=np.load(r'C:/Users/Avinash/Documents/soft_assignment/stone_0.npy')
sc=np.load(r'C:/Users/Avinash/Documents/soft_assignment/scissors.npy')
sci=np.load(r'C:/Users/Avinash/Documents/soft_assignment/scissors_0.npy')

p=p.reshape(50,50,50,1)
pi=pi.reshape(50,3)
s=s.reshape(50,50,50,1)
si=si.reshape(50,3)
sc=sc.reshape(50,50,50,1)
sci=sci.reshape(50,3)
inp=np.concatenate((s,p,sc))
out=np.concatenate((si,pi,sci))
#testing 
import matplotlib.pyplot as plt
k=p[0]
k=k.reshape(50,50)
plt.imshow(k)
plt.show()
k=s[0]
k=k.reshape(50,50)
plt.imshow(k)
plt.show()
k=sc[0]
k=k.reshape(50,50)
plt.imshow(k)
plt.show()

#CNN network to recognise the gestures.


from keras.layers import Conv2D, MaxPooling2D, Dense, Flatten
from keras.models import Sequential
net = Sequential()
net.add(Conv2D(32, kernel_size=(5, 5), strides=(1, 1),
                     activation='relu',
                     input_shape=(50,50,1)))
net.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))
net.add(Conv2D(64, (5, 5), activation='relu'))
net.add(MaxPooling2D(pool_size=(2, 2)))
net.add(Flatten())
net.add(Dense(1000, activation='relu'))
net.add(Dense(3, activation='softmax'))
net.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
net.fit(inp1,out1,epochs=3)

#shuffling the data to avoid over training of model.

from random import shuffle 
y=[]
for i in range(0,150):
    y.append(i)
shuffle(y)
#print(y)

ind=[]
for i in range(0,150):
    k=[y[i], inp[i], out[i]]
    ind.append(k)
ind.sort()
out1=[]
inp1=[]
for i in range(0,150):
    inp1.append(ind[i][1])
    out1.append(ind[i][2])
inp1=np.array(inp1).astype('float32')/255
out1=np.array(out1)


#getting real time user input

import urllib.request as ur
import cv2
import numpy as np
import time
import matplotlib.pyplot as plt
url='http://192.168.43.1:8080/shot.jpg'
imgResp = ur.urlopen(url)
imgNp = np.array(bytearray(imgResp.read()),dtype=np.uint8)
i = cv2.imdecode(imgNp,-1)
plt.imshow(i)
plt.show()
g = cv2.cvtColor(i, cv2.COLOR_BGR2GRAY)
plt.imshow(g)
plt.show()
g = cv2.threshold(g,100, 255, cv2.THRESH_BINARY)[1]
plt.imshow(g)
plt.show()
g=cv2.resize(g,(50,50))
plt.imshow(g)
plt.show()
k=g.reshape(1,50,50,1)
e=net.predict(k)
#c=input('')
print(e)


#LSTM model to predict the moves


import urllib.request as ur
import cv2
import numpy as np
from numpy import array
from keras.models import Sequential
from keras.layers import LSTM
from keras.layers import Dense
import time
import matplotlib.pyplot as plt
from time import sleep
url='http://192.168.43.1:8080/shot.jpg'
raw_seq = [1,3,2,1]

#getting user data in real

def input_():
    imgResp = ur.urlopen(url)
    imgNp = np.array(bytearray(imgResp.read()),dtype=np.uint8)
    i = cv2.imdecode(imgNp,-1)
    g = cv2.cvtColor(i, cv2.COLOR_BGR2GRAY)
    g = cv2.threshold(g,100, 255, cv2.THRESH_BINARY)[1]
    g=cv2.resize(g,(50,50))
    img1=g
    k=g.reshape(1,50,50,1)
    e=net.predict(k)
    if(e[0][0]==1):
        return 1
    if(e[0][1]==1):
        return 2
    if(e[0][2]==1):
        return 3
    return 0
    
def output_(o):
    if(o==1):
        ki='stone'
    elif(o==2):
        ki='paper'
    elif(o==3):
        ki='scissor'
    else:
        ki='null'
    return(ki)

def result_(a,b):
    print(a)
    print(b)
    k1=inp[a*50-5]
    k1=k1.reshape(50,50)
    k2=inp[b*50-5]
    k2=k2.reshape(50,50)
    fig=plt.figure(figsize=(2, 1))
    fig.add_subplot(1,2,1)
    plt.imshow(k1)
    plt.xlabel("player")
    fig.add_subplot(1,2,2)
    plt.imshow(k2)
    plt.xlabel("computer")
    plt.show()
    if((a==1 and b==2)or(a==2 and b==3)or(a==3 and b==1)):
        r=1
        
        raw_seq.append(b)
    elif((b==1 and a==2)or(b==2 and a==3)or(b==3 and a==1)):
        r=2
        if(a==1):raw_seq.append(2)
        if(a==2):raw_seq.append(3)    
        if(a==3):raw_seq.append(1)
    else:
        r=0
        if(a==1):raw_seq.append(2)
        if(a==2):raw_seq.append(3)    
        if(a==3):raw_seq.append(1)
    return(r)
    

c=0
m=0 

def split_sequence(sequence, n_steps):
	X, y = list(), list()
	for i in range(len(sequence)):
		end_ix = i + n_steps
		if end_ix > len(sequence)-1:
			break
		seq_x, seq_y = sequence[i:end_ix], sequence[end_ix]
		X.append(seq_x)
		y.append(seq_y)
	return array(X), array(y)



for i in range(0,15):
    n_steps = 3
    X, y = split_sequence(raw_seq, n_steps)
    n_features = 1
    X = X.reshape((X.shape[0], X.shape[1], n_features))

    
    model = Sequential()
    model.add(LSTM(50, activation='relu', return_sequences=True, input_shape=(n_steps, n_features)))
    model.add(LSTM(50, activation='relu'))
    model.add(Dense(1))
    model.compile(optimizer='adam', loss='mse')
    model.fit(X, y, epochs=50, verbose=0)
    # demonstrate prediction
    a=len(raw_seq)
    
    i=input_()
    x= array(raw_seq[a-3:a:1])
    x=x.reshape((1,3,1))
    out= model.predict(x, verbose=0)
    out=round(float(out))
    if(out<=3):
        r=result_(i,out)

        print('computer:    '+output_(out))
        print('you:         '+output_(i))

        if(r==1):
            c=c+1
        if(r==2):
            m=m+1

        print('computer: '+str(c)+'           you: '+str(m))