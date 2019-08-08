import pandas as pd
import keras
from.keras.models import Sequential
from keras.layers import Dense
from sklearn.model_selection import train_test_split

previsores = pd.read_csv('entradas-breast.csv')
classe = pd.read_csv('saidas-breast.csv')


previsores_treinamento, previsores_teste,classe_treinamento, classe_teste = train_test_split(previsores, classe, test_size = 0.25) 


classificador = Sequential()

#Configurando a rede neural
classificador.add(Dense(units = 16, 
			activation = "relu",
			kernel_initializer = "random_uniform",
			imput_dim = 30)

classificador.add(Dense(units = 1,
			activation = "sigmoid")

#configuração e execução
classificador.compile(	optimizer = "adam",
			loss = "binary_crossentropy",
			metrics = ["binary_accurancy"])

classificador.fit(	previsores_treinamento,
			classe_treinamento,
			batch_size = 10
			epochs = 100)

#Previsão e Avaliação da Rede Neural
previsoes = classificador.predict(previsores_teste)
previsoes = (previsoes > 0.5)

from sklearn.metrics import confusion_matrix, accuracy_score

precisão = accuracy_score(classe_teste, previsoes)
matriz = confusion_matrix(classe_teste, previsoes)

#utilizando keras
resultado = classificador.evaluate(previsores_teste, classe_teste)




