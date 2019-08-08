import numpy as np

#Step Funcion - Apenas para problemas linearmente separaveis
def stepFunction(soma):
    if(soma >= 1):
        return 1
    return 0

#Sigmoid - utilizada para problemas binários  
def sigFunction(soma):
    return 1 / (1+np.exp(-soma))

#Hyperbolic Tanget - retorna valores entre -1 e 1
def hypTangFunction(soma):
    return (np.exp(soma) - np.exp(-soma))/(np.exp(soma) + np.exp(-soma))

#ReLU - Retorna 0 ou um número maior que zero
def reLUFunction(soma):
   if(soma >= 0):
       return soma
   return 0

#Linear - Retorna o valor que foi passado
def linearFunction(soma):
    return soma

#Softmax - retorna probabilidades para problemas com mais de duas classes
def softmaxFunction(x):
    ex = np.exp(x)
    return ex/ex.sum()
    


  
teste = stepFunction(10)
teste = sigFunction(2.1)
teste = hypTangFunction(2.1)
teste = reLUFunction(2.1)
teste = linearFunction(-10)
valores = [5, 2, 1.3]
print(softmaxFunction(valores))