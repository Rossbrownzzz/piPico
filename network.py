import random
random.seed(123)

class Network:
    def __init__(self, numLayers=4, layerDim=[15,13,14,12], inputDim=11, outputDim=9, learningRate=0.1, lrDecay=0.00001, \
                 momentum=0.6, momentumDecay = 0.01, initialWeightsMax=0.15, trainBatchSize=1):
        self.numLayers = numLayers
        self.layerDim = layerDim
        self.inputDim = inputDim
        self.outputDim = outputDim
        self.learningRate = learningRate
        self.lrDecay = lrDecay
        self.momentum = momentum
        self.momentumDecay = momentumDecay
        self.initialWeightsMax = initialWeightsMax
        self.trainBatchSize = trainBatchSize

        #create and link layers
        self.inputLayer = Layer(layerNodes=inputDim)
        self.outputLayer = Layer(layerNodes=outputDim)
        self.layers = []
        for i in range(numLayers):
            self.layers.append(Layer(layerNodes=self.layerDim[i]))

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
        act = input
        for layer in self.layers:
            act = layer.activations(act)
        return self.outputLayer.activations(act)



    def train(self, trainData):
        for sample, result in trainData:
            activations = []
            act = sample
            for layer in self.layers:
                act = layer.activations(act)
                activations.append(act)
            activations.append(self.outputLayer.activations(act))

            error = self.loss(activations[-1], result)
            self.backPropogate(error, activations)

    ###### TODO try setting another neural network as the board state judge. make it regression rather than classification, and use the prospective increase or decrease in
    ###### board state value to get loss from every output, rather than just the move made.
    ###### train the regression network on board states encountered and whether it wound up being a win or a lose for that side.
    def loss(self, pred, true):
        if len(pred)==1:
            errors = [pred[0] - true]
        else:
            errors = []
            for i in range(len(pred)):
                errors.append(pred[i] - true[i])
        return errors


    def backPropogate(self, error, activations):
        activations.reverse()
        deltas = [[]]
        for outNode in range(len(activations[0])):
            deltas[0].append(error[outNode] * activations[0][outNode] * (1-activations[0][outNode]))
        i = 1
        #print('DELTAS 0', deltas[0])
        for layer in self.layers[::-1]:
            #print(layer.hiddenWeights)
            delta = []
            for j in range(len(activations[i])):
                #print('-------act', activations[i])
                D = 0
                for prevDelta in deltas[i-1]:
                    D += activations[i][j] * prevDelta
                D = D * activations[i][j] * (1 - activations[i][j]) 
                delta.append(D)
            deltas.append(delta)
            #print(deltas)
            for j in range((len(activations[i]))):
                for node in range(len(layer.hiddenWeights[j])):
                    #print(layer.hiddenWeights[j][node])
                    layer.hiddenWeights[j][node] += -activations[i][j] * deltas[i][j] * self.learningRate
                    #print('activ, delta, lr', -activations[i][j], deltas[i][j], self.learningRate)
            #print("new", layer.hiddenWeights)
            i += 1 




class Layer:
    def __init__(self, layerNodes=11, initialWeightsMax=0.15):
        self.layerNodes = layerNodes      
        self.initialWeightsMax = initialWeightsMax     


    def link(self, next: 'Layer', previous: 'Layer'):
        self.nextLayer = next
        self.previousLayer = previous
        if self.previousLayer is not None:
            self.previousLayerNodes = previous.layerNodes
            self.hiddenWeights = [[random.randint(1,100)/100 * (-1 if random.random()>0.5 else 1) * self.initialWeightsMax for i in range(self.previousLayerNodes)] for j in range(self.layerNodes)]
        
        if self.nextLayer is not None:
            self.nextLayerNodes = next.layerNodes
       

    def activations(self, inputs):
        outputValue = [0]*self.layerNodes
        for i in range(self.layerNodes):
            accum = 0
            for j in range(self.previousLayerNodes):
                accum += inputs[j] * self.hiddenWeights[i][j] 
            outputValue[i] = 1/(1+pow(2.718, -accum)) 
        #print('o', outputValue)
        return outputValue
        


test = Network(inputDim=2, numLayers=2, layerDim=[2,2], outputDim=2)
prev = test.predict([1,0])#[0,1,1,1,1,0,1,0,1,1,0,1])
print(prev)

for i in range(100):
    test.train([[[1,0],[10,0]]])
    #test.train([[[0,1,1,1,1,0,1,0,1,1,0,1], [1, 0]]])
    curr = test.predict([1,0])
    #curr = test.predict([0,1,1,1,1,0,1,0,1,1,0,1])
    #print(curr)
    print(curr[0]-prev[0], curr[1]-prev[1])
    prev=curr
print(curr)