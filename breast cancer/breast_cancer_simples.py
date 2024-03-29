import pandas as pd

previsores = pd.read_csv('entradas-breast.csv')
classe = pd.read_csv('saidas-breast.csv')

from sklearn.model_selection import train_test_split
previsores_treinamento, previsores_teste,classe_treinamento, classe_teste = train_test_split(previsores, classe, test_size = 0.25) 

import keras
from keras.models import Sequential #cria as camadas sequencialmente
from keras.layers import Dense
classificador = Sequential()

#Configurando a rede neural
classificador.add(Dense(units = 16, activation = 'relu',
			kernel_initializer = 'random_uniform', input_dim = 30))
#Adicionando mais uma camada na rede Neural
#input_dim vai apaenas na primeira camada oculta
#em geral utiliza-se a mesma quantidade de neurônios (16), porém, deve ser testada a melhor configuração
classificador.add(Dense(units = 16, activation = 'relu',
			kernel_initializer = 'random_uniform'))

classificador.add(Dense(units = 1, activation = 'sigmoid'))

#configuração e execução

otimizador = keras.optimizers.Adam(lr=0.001, decay = 0.0001, clipvalue = 0.5)
classificador.compile(	optimizer = otimizador,loss = 'binary_crossentropy', metrics = ['binary_accuracy'])


#classificador.compile(	optimizer = 'adam',loss = 'binary_crossentropy', metrics = ['binary_accuracy'])

classificador.fit(	previsores_treinamento,
		classe_treinamento,
			batch_size = 10,
			epochs = 100)

#Previsão e Avaliação da Rede Neural
previsoes = classificador.predict(previsores_teste)
previsoes = (previsoes > 0.5)

from sklearn.metrics import confusion_matrix, accuracy_score

precisão = accuracy_score(classe_teste, previsoes)
matriz = confusion_matrix(classe_teste, previsoes)

#utilizando keras
resultado = classificador.evaluate(previsores_teste, classe_teste)




