# import all libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier
from sklearn import tree
from sklearn.feature_selection import RFE
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.metrics import mean_squared_error
from sklearn.metrics import mean_absolute_error
from sklearn.metrics import r2_score
from keras.preprocessing import sequence
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation, Embedding
from keras.layers import LSTM, SimpleRNN, GRU


df = pd.read_csv('kdd.csv',header=0)
df=df.drop(['service','flag'], axis=1)
print(df)
df['class'] = df['class'].replace(['anomaly'], 1)
df['class'] = df['class'].replace(['normal'], 0)

df['protocol_type'] = df['protocol_type'].replace(['udp'], 0)
df['protocol_type'] = df['protocol_type'].replace(['tcp'], 1)
df['protocol_type'] = df['protocol_type'].replace(['icmp'], 2)

print(df)
X=df.drop(columns=['class'])
y=df['class']
print(X)


names = df.head()
dtree = tree.DecisionTreeClassifier()
rfe = RFE(estimator=dtree, n_features_to_select=16)
rfe.fit(X, y)
# summarize the selection of the attributes
print(rfe.support_)
print("Rank")
print(rfe.ranking_)
print ("Features sorted by their rank:")
print (sorted(zip(map(lambda x: round(x, 4), rfe.ranking_), names)))
mm=sorted(zip(map(lambda x: round(x, 4), rfe.ranking_), names))
print(mm)
cols=[mm[0][1],mm[1][1],mm[2][1],mm[3][1],mm[4][1],mm[5][1],mm[6][1],mm[7][1],mm[8][1],mm[9][1],mm[10][1],mm[11][1],mm[12][1],mm[13][1],mm[14][1],mm[15][1]]
print(cols)
print(df[cols])
X=df[cols]


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.05, random_state=42)




sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.fit_transform(X_test)


 
# Set the n_components=3
principal=PCA(n_components=3)
X_train=principal.fit_transform(X_train)
X_test=principal.fit_transform(X_test)
 
# Check the dimensions of data after PCA
print(X_train)
print("PCAA")
explained_variance = principal.explained_variance_ratio_

print("PCAA DONE")





model= Sequential()

model.add(Dense(16, kernel_initializer='uniform', activation='relu', input_dim=3))
model.add(Dense(14, kernel_initializer='uniform', activation='relu'))
model.add(Dense(1, activation="sigmoid"))
model.compile(optimizer='rmsprop', loss='binary_crossentropy', metrics=['accuracy'])
hist=model.fit(X_train,y_train, epochs=10,batch_size=10,validation_data=(X_test, y_test))

#model.add(Dense(16, input_dim=3))
#model.add(Dense(14))
#model.add(Dense(1, activation="sigmoid"))
#model.compile(optimizer='rmsprop', loss='binary_crossentropy', metrics=['accuracy'])
#hist=model.fit(X_train,y_train, epochs=10,batch_size=10,validation_data=(X_test, y_test))

#train and validation loss
plt.plot(hist.history['loss'])
plt.plot(hist.history['val_loss'])
plt.title('Model Loss')
plt.ylabel('Loss')
plt.xlabel('Epoch')
plt.legend(['train','Validation'],loc='upper left')
plt.savefig('results/DNN Loss_kdd.png') 
plt.pause(5)
plt.show(block=False)
plt.close()

#train and validation accuracy
plt.plot(hist.history['accuracy'])
plt.plot(hist.history['val_accuracy'])
plt.title('Model Accuracy')
plt.ylabel('Accuracy')
plt.xlabel('Epoch')
plt.legend(['train','Validation'],loc='upper left')
plt.savefig('results/DNN Accuracy_kdd.png') 
plt.pause(5)
plt.show(block=False)
plt.close()


y_pred=model.predict(X_test)
y_pred = [np.argmax(x) for x in y_pred]



