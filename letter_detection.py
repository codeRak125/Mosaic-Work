import tensorflow as tf
from tensorflow import keras
import string
import cv2
immport random
import os

PATH='characters'
all_symbols=string.ascii_uppercase + string.ascii_lowercase +'0123456789' + '^%/*+-'

traindata=[]
for char in all_symbols:
	path=os.path.join(PATH,char)
	for img in os.lisdir(path):
		symb_index=all_symbol.index(char)
		img_array=cv2.immread(os.path.join(path,img),cv2.IMREAD_GRAYSCALE)
		image_array=cv2.resize(img_array,(100,100))
		traindata.append([image_array,symb_index])
random.shuffle(traindata)
Xfull=[]
yfull=[]

for freatures,lables in traindata:
	Xfull.append(freatues)
	yfull.append(labels)

X=np.array(Xfull,dtype='float64')
y=np.array(yfull,dtype='float64')

def model():
	inputs = keras.Input(shape=(100,100,1))
	x = keras.layers.Conv2D(200, (3,3),activation='relu')(inputs)
	x_shortcut = x
	x = keras.layers.MaxPooling2D((2,2))(x)
	x = keras.layers.Add()([x,x_shortcut])
	x = keras.layers.Conv2D(500, (3,3),activation='relu')(x)
	x_shortcut = x
	x = keras.layers.MaxPooling2D((2,2) , padding='same')(x)
	x = keras.layers.Add()([x,x_shortcut])
	x = keras.layers.Conv2D(500, (3,3),activation='relu')(x)
	x_shortcut = x
	x = keras.layers.MaxPooling2D((2,2))(x)
	x = keras.layers.Add()([x,x_shortcut])
	x = keras.layers.BatchNormalization()(x)
	out_flat= keras.layers.Flatten()(x)
	dense_1 = keras.layers.Dense(64 , activation='relu')(out_flat)
	out_1 = keras.layers.Dense(len_symbols , activation='sigmoid')(dense_1)
	model_out = keras.Model(inputs=inputs , outputs=out_1)
	model.compile(loss='sparse_categorical_crossentropy', optimizer='adam',metrics=['accuracy'])

	return model_out

model1=model()
model1.fit(X,y)
model.save('m1.h5')
#newmodel=keras.models.load_model('m1.h5')
#prediction=newmodel.predict(Xpredict)
