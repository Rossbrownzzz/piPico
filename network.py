import random
random.seed(123)

class Network:
    def __init__(self, numLayers=4, inputDim=11, outputDim=9, learningRate=0.08, lrDecay=0.0001, momentum=0.6, momentumDecay = 0.01, initialWeightsMax=0.15, trainBatchSize=1):
        self.numLayers = numLayers
        self.inputDim = inputDim
        self.outputDim = outputDim
        self.learningRate = learningRate
        self.lrDecay = lrDecay
        self.momentum = momentum
        self.momentumDecay = momentumDecay
        self.initialWeightsMax = initialWeightsMax
        self.trainBatchSize = trainBatchSize

        self.inputLayer = Layer(layerNodes=inputDim)
        self.outputLayer = Layer(layerNodes=outputDim)
        self.layers = []
        for i in range(numLayers):
            self.layers.append(Layer())

        self.inputLayer.link(previous=None, next=self.layers[0])
        for i in range(numLayers):
            layer : Layer
            layer = self.layers[i]
            if i == 0:
                layer.link(previous=self.inputLayer, next=self.layers[i+1])
            elif i == numLayers-1:
                layer.link(previous=self.layers[i-1], next=self.outputLayer)
            else:
                layer.link(previous=self.layers[i-1], next=self.layers[i+1])
        
        self.outputLayer.link(previous=self.layers[numLayers-1], next=None)


    def predict(self, input):
        return self.layers[0].activations(input)


    def updateWeights(self, trainingGames):
        for games in trainingGames:
            gameResult = games[-1]
            gameinfo = games[:-1]
            for inputOutput in gameinfo:
                boardState = inputOutput[0]
                moveMade = inputOutput[1]

                losses = self.loss(boardState, gameResult)
                self.backPropogate(losses, moveMade)


    def loss(self, pred, result):
        # if won, it should have been higher
        if result == 1:
            return (pred + pred * self.learningRate) - pred
        # if lost it should have been lower
        elif result == -1:
            return -(pred - (pred - pred * self.learningRate))
        # if tie it should have been slightly lower
        elif result == 0:
            return -(pred - (pred - pred * self.learningRate/4))
        

    def backPropogate(self, loss, outNode):
        hiddenErrors = [0]*self.layerNodes
        for i in range(self.layerNodes):
            accum = 0
            accum += self.outputWeights[outNode][i] * loss
        
        for i in range(self.layerNodes):
            hiddenErrors[i] = accum * self.hiddenValue[i] * (1- self.hiddenValue[i])
            for j in range(self.previousLayerNodes):
                self.hiddenWeights[i][j] += hiddenErrors[i] * self.learningRate * self.momentum
            self.outputWeights[outNode][i] += loss * self.learningRate * self.momentum
        # decay momentum
        if self.momentum > self.momentumDecay:
            self.momentum = self.momentum - self.momentumDecay
        if self.learningRate > self.lrDecay:
            self.learningRate = self.learningRate - self.lrDecay




class Layer:
    def __init__(self, layerNodes=11, initialWeightsMax=0.15):
        self.layerNodes = layerNodes      
        self.initialWeightsMax = initialWeightsMax     


    def link(self, next: 'Layer', previous: 'Layer'):
        self.nextLayer = next
        self.previousLayer = previous
        if self.previousLayer is not None:
            self.previousLayerNodes = previous.layerNodes
            self.hiddenWeights = [[random.randint(1,100)/100 * self.initialWeightsMax for i in range(self.previousLayerNodes)] for j in range(self.layerNodes)]
            #self.hiddenBias = [random.randint(1,100)/100 * initialWeightsMax for i in range(layerNodes)]
        
        if self.nextLayer is not None:
            self.nextLayerNodes = next.layerNodes
            self.outputWeights = [[random.randint(1,100)/100 * self.initialWeightsMax for i in range(self.layerNodes)] for j in range(self.nextLayerNodes)]
            #self.outputBias = [random.randint(1,100)/100 * initialWeightsMax for i in range(outputNodes)]
       


    def activations(self, inputs):
        outputValue = [0]*self.layerNodes
        for i in range(self.layerNodes):
            accum = 0
            for j in range(self.previousLayerNodes):
                accum += inputs[j] * self.hiddenWeights[i][j] #+ self.hiddenBias[i]
            outputValue[i] = 1/(1+pow(2.718, -accum))
        if self.nextLayer is not None:
            return self.nextLayer.activations(outputValue)
        else:    
            return outputValue
        

test = Network()
print(test.predict([0,1,1,1,1,2,1,0,1,1,0,1]))