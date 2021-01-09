
# importing libraries 
from PyQt5 import QtCore, QtWidgets, QtGui
from PyQt5.QtWidgets import * 
from PyQt5 import QtCore, QtGui 
from PyQt5.QtGui import * 
from PyQt5.QtCore import * 
import sys 

import yfinance as yf

# Importar bibliotecas 
import pandas as pd
import numpy as np
import math
import torch
import matplotlib.pyplot as plt

import tkinter as tk
import easygui


class MyWindow(QScrollArea):
    def __init__(self):
        super(MyWindow, self).__init__()
        self.initUI()
        
    def initUI(self):
        self.x = 600
        self.y = 600
        self.setMinimumSize(QSize(self.x, self.y))
        self.setWindowTitle("Análise de Séries temporais")
        

        # Criando um QLabel
        self.texto = QLabel("Simbolo da Ação: ", self)
        self.texto.adjustSize()
        self.texto.setAlignment(QtCore.Qt.AlignCenter)
        self.larguraTexto = self.texto.frameGeometry().width()
        self.alturaTexto = self.texto.frameGeometry().height()
        self.texto.setAlignment(QtCore.Qt.AlignCenter)
        self.texto.move( 80 , 30 - self.alturaTexto/2)

        # Criando QLineEdit
        self.largura = 150
        self.altura = 40
        self.positionX = 80
        self.positionY = 50
        self.line_edit = QLineEdit(self) #criando
        self.line_edit.setGeometry(self.positionX, self.positionY, self.largura, self.altura) #posicionando
        self.line_edit.setAlignment(QtCore.Qt.AlignCenter) #centralizando texto

        # Criando um QLabel - Data inicio
        self.textoDate = QLabel("Data Inicio: ", self)
        self.textoDate.adjustSize()
        self.textoDate.setAlignment(QtCore.Qt.AlignCenter)
        self.larguraTextoDateStart = self.textoDate.frameGeometry().width()
        self.alturaTextoDateStart = self.textoDate.frameGeometry().height()
        self.textoDate.setAlignment(QtCore.Qt.AlignCenter)
        self.textoDate.move( 250 , 30 - self.alturaTextoDateStart/2)

        # Calendario Inico
        self.calendarStart = QCalendarWidget(self)
        self.calendarStart.adjustSize()
        self.larguraCalendarStart = self.calendarStart.frameGeometry().width()
        self.alturaCalendarStart = self.calendarStart.frameGeometry().height()
        self.positionXCalendarStart = 250
        self.positionYCalendarStart = 50
        self.calendarStart.move(self.positionXCalendarStart, self.positionYCalendarStart)

        # Criando um QLabel - Data Fim
        self.textoDate = QLabel("Data Fim: ", self)
        self.textoDate.adjustSize()
        self.textoDate.setAlignment(QtCore.Qt.AlignCenter)
        self.larguraTextoDateFinal = self.textoDate.frameGeometry().width()
        self.alturaTextoDateFinal = self.textoDate.frameGeometry().height()
        self.textoDate.setAlignment(QtCore.Qt.AlignCenter)
        self.textoDate.move( 580 , 30 - self.alturaTextoDateFinal/2)

        # Calendario Fim
        self.calendarFinal = QCalendarWidget(self)
        self.calendarFinal.adjustSize()
        self.larguraCalendarFinal = self.calendarFinal.frameGeometry().width()
        self.alturaCalendarFinal = self.calendarFinal.frameGeometry().height()
        self.positionXCalendarFinal = 580
        self.positionYCalendarFinal = 50
        self.calendarFinal.move(self.positionXCalendarFinal, self.positionYCalendarFinal)

        # Criando um QLabel
        self.titleAction = QLabel("Name", self)
        self.titleAction.setWordWrap(True) 
        self.titleAction.adjustSize()
        self.titleAction.setAlignment(QtCore.Qt.AlignCenter)
        self.larguraTitleAction = self.titleAction.frameGeometry().width()
        self.alturaTitleAction = self.titleAction.frameGeometry().height()
        self.titleAction.setAlignment(QtCore.Qt.AlignCenter)
        self.titleAction.move(80, 110)

        # Criando um QLabel
        self.titleCountry = QLabel("Country", self)
        self.titleCountry.setWordWrap(True) 
        self.titleCountry.adjustSize()
        self.titleCountry.setAlignment(QtCore.Qt.AlignCenter)
        self.larguraTitleCountry = self.titleCountry.frameGeometry().width()
        self.alturaTitleCountry = self.titleCountry.frameGeometry().height()
        self.titleCountry.setAlignment(QtCore.Qt.AlignCenter)
        self.titleCountry.move(80, 150)

        # Criando um QLabel
        self.titleSector = QLabel("Sector", self)
        self.titleSector.setWordWrap(True) 
        self.titleSector.adjustSize()
        self.titleSector.setAlignment(QtCore.Qt.AlignCenter)
        self.larguraTitleSector = self.titleSector.frameGeometry().width()
        self.alturaTitleSector = self.titleSector.frameGeometry().height()
        self.titleSector.setAlignment(QtCore.Qt.AlignCenter)
        self.titleSector.move(80, 190)

        # Criando um QLabel
        self.titleTimes = QLabel("Épocas", self)
        self.titleTimes.setWordWrap(True) 
        self.titleTimes.adjustSize()
        self.titleTimes.setAlignment(QtCore.Qt.AlignCenter)
        self.larguraTitleTimes = self.titleTimes.frameGeometry().width()
        self.alturaTitleTimes = self.titleTimes.frameGeometry().height()
        self.titleTimes.setAlignment(QtCore.Qt.AlignCenter)
        self.titleTimes.move(910, 30)

        # Criando QLineEdit
        self.larguraTimes = 150
        self.alturaTimes = 25
        self.positionXTimes = 910
        self.positionYTimes = 50
        self.line_editTimes = QLineEdit(self) #criando
        self.line_editTimes.setGeometry(self.positionXTimes, self.positionYTimes, self.larguraTimes, self.alturaTimes) #posicionando
        self.line_editTimes.setAlignment(QtCore.Qt.AlignCenter) #centralizando texto

        # Criando um QLabel
        self.titleNeurons = QLabel("Neuronios Ocultos", self)
        self.titleNeurons.setWordWrap(True) 
        self.titleNeurons.adjustSize()
        self.titleNeurons.setAlignment(QtCore.Qt.AlignCenter)
        self.larguraTitleNeurons = self.titleNeurons.frameGeometry().width()
        self.alturaTitleNeurons = self.titleNeurons.frameGeometry().height()
        self.titleNeurons.setAlignment(QtCore.Qt.AlignCenter)
        self.titleNeurons.move(910, 85)

        # Criando QLineEdit
        self.larguraNeurons = 150
        self.alturaNeurons = 25
        self.positionXNeurons = 910
        self.positionYNeurons = 105
        self.line_editNeurons = QLineEdit(self) #criando
        self.line_editNeurons.setGeometry(self.positionXNeurons, self.positionYNeurons, self.larguraNeurons, self.alturaNeurons) #posicionando
        self.line_editNeurons.setAlignment(QtCore.Qt.AlignCenter) #centralizando texto

        # Criando um QLabel
        self.titleLearning = QLabel("Learning Rate", self)
        self.titleLearning.setWordWrap(True) 
        self.titleLearning.adjustSize()
        self.titleLearning.setAlignment(QtCore.Qt.AlignCenter)
        self.larguraTitleLearning = self.titleLearning.frameGeometry().width()
        self.alturaTitleLearning = self.titleLearning.frameGeometry().height()
        self.titleLearning.setAlignment(QtCore.Qt.AlignCenter)
        self.titleLearning.move(910, 140)

        # Criando QLineEdit
        self.larguraLearning = 150
        self.alturaLearning = 25
        self.positionXLearning = 910
        self.positionYLearning = 160
        self.line_editLearning = QLineEdit(self) #criando
        self.line_editLearning.setGeometry(self.positionXLearning, self.positionYLearning, self.larguraLearning, self.alturaLearning) #posicionando
        self.line_editLearning.setAlignment(QtCore.Qt.AlignCenter) #centralizando texto

        # Criando um QLabel
        self.titleMomentum = QLabel("Momentum", self)
        self.titleMomentum.setWordWrap(True) 
        self.titleMomentum.adjustSize()
        self.titleMomentum.setAlignment(QtCore.Qt.AlignCenter)
        self.larguraTitleMomentum = self.titleMomentum.frameGeometry().width()
        self.alturaTitleMomentum = self.titleMomentum.frameGeometry().height()
        self.titleMomentum.setAlignment(QtCore.Qt.AlignCenter)
        self.titleMomentum.move(910, 190)

        # Criando QLineEdit
        self.larguraMomentum = 150
        self.alturaMomentum = 25
        self.positionXMomentum = 910
        self.positionYMomentum = 210
        self.line_editMomentum = QLineEdit(self) #criando
        self.line_editMomentum.setGeometry(self.positionXMomentum, self.positionYMomentum, self.larguraMomentum, self.alturaMomentum) #posicionando
        self.line_editMomentum.setAlignment(QtCore.Qt.AlignCenter) #centralizando texto
    
        # Criando um Botão
        self.btnSubmit = QtWidgets.QPushButton(self)
        self.btnSubmit.setText("Submit")
        self.larguraBtnSubmit = self.btnSubmit.frameGeometry().width()
        self.alturaBtnSubmit = self.btnSubmit.frameGeometry().height()
        self.positionXBtnSubmit = 1080
        self.positionYBtnSubmit = 120
        self.btnSubmit.move(self.positionXBtnSubmit, self.positionYBtnSubmit)
        self.btnSubmit.clicked.connect(self.button_clicked)

       # Criando um QLabel
        self.titleGrafTemporal = QLabel("Gráficos Temporais", self)
        self.titleGrafTemporal.setWordWrap(True) 
        self.titleGrafTemporal.adjustSize()
        self.titleGrafTemporal.setAlignment(QtCore.Qt.AlignCenter)
        self.larguraTitleGrafTemporal = self.titleGrafTemporal.frameGeometry().width()
        self.alturaTitleGrafTemporal = self.titleGrafTemporal.frameGeometry().height()
        self.titleGrafTemporal.setAlignment(QtCore.Qt.AlignCenter)
        self.titleGrafTemporal.move(80, 250)

        # Grafico Todo
        self.grafTempTotal = QLabel(self)

        # Grafico Treinamento
        self.grafTempTreino = QLabel(self)

        # Grafico Teste
        self.grafTempTeste = QLabel(self)

        # Grafico Errors
        self.grafErrors = QLabel(self)

        # Grafico LatsErrors
        self.grafLastErrors = QLabel(self)

        # Grafico Tests
        self.grafTests = QLabel(self)

        self.actions = [
            "FNAM11.SA",
            "OIBR3.SA",
            "FNOR11.SA",
            "PETR4.SA",
            "AVIP5L.SA",
            "VVAR3.SA",
            "CIEL3.SA",
            "BBDC4.SA",
            "IRBR3.SA",
            "MGLU3.SA",
            "MEAL3.SA",
            "VALE3.SA",
            "BEEF3.SA",
            "ITUB4.SA",
            "JBSS3.SA",
            "MRFG3.SA",
            "ITSA4.SA",
            "BRFS3.SA",
            "BRML3.SA",
            "POMO4.SA",
            "SUZB3.SA",
            "EMBR3.SA",
            "USIM5.SA",
        ]
        

    def button_clicked(self):

        self.actionName = self.line_edit.text().upper() + ".SA"
        print("Nome da Ação: " + self.actionName)

        for a in self.actions:
            # Nome da Ação
            if a == self.actionName:
                self.existe = "TRUE"
                break
            else:
                self.existe = "FALSE"
                
                   
        if self.existe == "TRUE":

            # Setando o Nome da Ação Escolhida
            self.valueAction = yf.Ticker(self.line_edit.text().upper() + ".SA").info['longName']
            self.titleAction.setText( self.valueAction )
            self.titleAction.adjustSize()
            self.titleAction.setAlignment(QtCore.Qt.AlignCenter)
            self.larguraTitleAction = self.titleAction.frameGeometry().width()
            self.alturaTitleAction = self.titleAction.frameGeometry().height()
            self.titleAction.setAlignment(QtCore.Qt.AlignCenter)
            self.titleAction.move(80, 110)

            # Setando o País da Ação Escolhida
            self.valueCountry = yf.Ticker(self.line_edit.text().upper() + ".SA").info['country']
            self.titleCountry.setText( self.valueCountry )
            self.titleCountry.adjustSize()
            self.titleCountry.setAlignment(QtCore.Qt.AlignCenter)
            self.larguraTitleCountry = self.titleCountry.frameGeometry().width()
            self.alturaTitleCountry = self.titleCountry.frameGeometry().height()
            self.titleCountry.setAlignment(QtCore.Qt.AlignCenter)
            self.titleCountry.move(80, 150)

            # Setando o Setor da Ação Escolhida
            self.valueSector = yf.Ticker(self.line_edit.text().upper() + ".SA").info['sector']
            self.titleSector.setText( self.valueSector )
            self.titleSector.adjustSize()
            self.titleSector.setAlignment(QtCore.Qt.AlignCenter)
            self.larguraTitleSector = self.titleSector.frameGeometry().width()
            self.alturaTitleSector = self.titleSector.frameGeometry().height()
            self.titleSector.setAlignment(QtCore.Qt.AlignCenter)
            self.titleSector.move(80, 190)

            # Retornando valor data de inicio
            self.dateStart = self.calendarStart.selectedDate()
            print("Selected Date: " + self.dateStart.toString(Qt.ISODate))

            # Retornando valor data final
            self.dateFinal = self.calendarFinal.selectedDate()
            print("Selected Date: " + self.dateFinal.toString(Qt.ISODate))

            self.data = yf.download(self.actionName, start=self.dateStart.toString(Qt.ISODate), end=self.dateFinal.toString(Qt.ISODate))
            self.data = self.data.Close
            print(f'Total: {self.data.size}')
            print(f'Treinamento: {round(self.data.size*.8)}')
            print(f'Teste: {self.data.size - round(self.data.size*.8)}')


            # Plotar o Gráfico
            plt.figure(figsize=(4, 2))
            plt.plot(self.data, '-')
            plt.xlabel('DIAS')
            plt.ylabel('VALOR R$')
            plt.title(self.actionName)
            self.nameImage = "grafTempTodo.png"
            plt.savefig('./images/' + self.nameImage, format="png")
            self.dirGrafTempTodo = QtGui.QPixmap('./images/' + self.nameImage)
            self.grafTempTotal.setPixmap(self.dirGrafTempTodo)
            self.grafTempTotal.setGeometry(40, 280, 600, 200)
        
            # Plotar treinamento e teste
            plt.figure(figsize=(4, 2))
            plt.plot(self.data[:850], 'r-')
            plt.plot(self.data[850:], 'g-')
            plt.xlabel('DIAS')
            plt.ylabel('VALOR R$')
            plt.title(self.actionName)
            plt.axvline(self.data.index[850], 0, 30, color='k', linestyle='dashed', label='Teste')
            plt.text(self.data.index[320], 25, 'Treinamento', fontsize='x-large')
            plt.text(self.data.index[910], 15, 'Testes', fontsize='x-large')
            self.nameImageGrafTreino = "grafTempTreino.png"
            plt.savefig('./images/' + self.nameImageGrafTreino, format="png")
            self.dirGrafTempTreino = QtGui.QPixmap('./images/' + self.nameImageGrafTreino)
            self.grafTempTreino.setPixmap(self.dirGrafTempTreino)
            self.grafTempTreino.setGeometry(460, 280, 600, 200)
            # plt.show()

            # Plotar apenas teste
            plt.figure(figsize=(4, 2))
            plt.plot(self.data[850:], 'g-')
            plt.xlabel('DIAS')
            plt.ylabel('VALOR R$')
            plt.title(self.actionName)
            self.nameImageGrafTeste = "grafTempoTeste.png"
            plt.savefig('./images/' + self.nameImageGrafTeste, format="png")
            self.dirGrafTempTeste = QtGui.QPixmap('./images/' + self.nameImageGrafTeste)
            self.grafTempTeste.setPixmap(self.dirGrafTempTeste)
            self.grafTempTeste.setGeometry(880, 280, 600, 200)
            # plt.show()

            # Criar janela deslizante
            self.janelas = 50

            self.data_final = np.zeros([self.data.size - self.janelas, self.janelas + 1])

            for i in range(len(self.data_final)):
                for j in range(self.janelas+1):
                    self.data_final[i][j] = self.data.iloc[i+j]

            # Normalizar entre 0 e 1
            self.max = self.data_final.max()
            self.min = self.data_final.min()
            self.dif = self.data_final.max() - self.data_final.min()
            self.data_final = (self.data_final - self.data_final.min())/self.dif

            self.x = self.data_final[:, :-1]
            self.y = self.data_final[:, -1]
            print(self.max, self.min, self.dif)


            # Converter para tensor
            #Entrada do treinamento
            #Saída do treinamento
            self.training_input = torch.FloatTensor(self.x[:850, :])
            self.training_output = torch.FloatTensor(self.y[:850])

            #Entrada do teste
            #Saída do teste
            self.test_input = torch.FloatTensor(self.x[850: , :])
            self.test_output = torch.FloatTensor(self.y[850:])

            print(self.test_input)
            print(self.test_output)

            # Classe do modelo da Rede Neural
            class Net(torch.nn.Module):
                def __init__(self, input_size, hidden_size):
                    super(Net, self).__init__()
                    self.input_size = input_size
                    self.hidden_size = hidden_size
                    self.fc1 = torch.nn.Linear(self.input_size, self.hidden_size)
                    self.relu = torch.nn.ReLU()
                    self.fc2 = torch.nn.Linear(self.hidden_size, 1)
                def forward(self, x):
                    hidden = self.fc1(x)
                    relu = self.relu(hidden)
                    output = self.fc2(relu)
                    output = self.relu(output)
                    return output

            # Criar a instância do modelo
            self.input_size = self.training_input.size()[1]
            self.hidden_size = int(self.line_editNeurons.text())
            self.model = Net(self.input_size, self.hidden_size)
            print(f'Entrada: {self.input_size}')
            print(f'Escondida: {self.hidden_size}')
            print(self.model)

            # Critério de erro
            self.criterion = torch.nn.MSELoss()

            # Criando os paramêtros (learning rate[obrigatória] e momentum[opcional])
            self.lr = float(self.line_editLearning.text()) #0.09
            self.momentum = float(self.line_editMomentum.text()) #0.03
            self.optimizer = torch.optim.SGD(self.model.parameters(), self.lr, self.momentum)

            # Para visualizar os pesos
            for self.param in self.model.parameters():
                print(self.param)
                pass

            # Treinamento
            self.model.train()
            self.epochs = int(self.line_editTimes.text()) # 100001
            self.errors = []
            for self.epoch in range(self.epochs):
                self.optimizer.zero_grad()
                # Fazer o forward
                self.y_pred = self.model(self.training_input)
                # Cálculo do erro
                self.loss = self.criterion(self.y_pred.squeeze(), self.training_output)
                self.errors.append(self.loss.item())
                if self.epoch % 1000 == 0:
                    print(f'Epoch: {self.epoch}. Train loss: {self.loss.item()}.')
                # Backpropagation
                self.loss.backward()
                self.optimizer.step()
            
            # Testar o modelo já treinado
            self.model.eval()
            self.y_pred = self.model(self.test_input)
            self.after_train = self.criterion(self.y_pred.squeeze(), self.test_output)
            print('Test loss after Training' , self.after_train.item())


            # Gráficos de erro e de previsão
            def plotcharts(errors):
                self.errors = np.array(errors)
                lasterrors = np.array(self.errors[-2500:])
                plt.figure(figsize=(12, 2))
                graf01 = plt.subplot(1, 3, 1) # nrows, ncols, index
                graf01.set_title('Errors')
                plt.plot(self.errors, '-')
                plt.xlabel('Epochs')
                graf02 = plt.subplot(1, 3, 2) # nrows, ncols, index
                graf02.set_title('Last 25k Errors')
                plt.plot(lasterrors, '-')
                plt.xlabel('Epochs')
                graf03 = plt.subplot(1, 3, 3)
                graf03.set_title('Tests')
                a = plt.plot(self.test_output.numpy(), 'y-', label='Real')
                #plt.setp(a, markersize=10)
                a = plt.plot(self.y_pred.detach().numpy(), 'b-', label='Predicted')
                #plt.setp(a, markersize=10)
                plt.legend(loc=7)

                self.nameImageErrors = "grafErrors.png"
                plt.savefig('./images/' + self.nameImageErrors, format="png")
                self.dirGrafErrors = QtGui.QPixmap('./images/' + self.nameImageErrors)
                self.grafErrors.setPixmap(self.dirGrafErrors)
                self.grafErrors.setGeometry(0, 580, 1400, 200)
                # plt.show()
            plotcharts(self.errors)
        else:
            easygui.msgbox("Ação não Encontrada", title="Alerta!")
            print("Ação não Existe!")

def window():
    app = QApplication(sys.argv)
    win = MyWindow()
    win.show()
    sys.exit(app.exec_())

window()