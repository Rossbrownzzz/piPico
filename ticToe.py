import random

# for Q, maybe predict likelihood of win with another net for each board state, use that as a true value to use when updating to update all output nodes of playing net?

# My hidden error calculation won't work with higher batch sizes

# i dont use biases

# i don't have to option to make p2 a net too

class Network:
    def __init__(self, inputNodes=10, hiddenNodes=11, outputNodes=9, learningRate=0.08, lrDecay=0.0001, momentum=0.6, momentumDecay = 0.01, initialWeightsMax=0.15, trainBatchSize=1, epsilon=0.75, epsilonDecay=0.01):
        self.inputNodes = inputNodes
        self.hiddenNodes = hiddenNodes
        self.outputNodes = outputNodes
        self.learningRate = learningRate
        self.lrDecay = lrDecay
        self.momentum = momentum
        self.momentumDecay = momentumDecay
        self.initialWeightsMax = initialWeightsMax
        self.trainBatchSize = trainBatchSize
        self.epsilon = epsilon
        self.epsilonDecay = epsilonDecay


        self.hiddenWeights = [[random.randint(1,100)/100 * initialWeightsMax for i in range(inputNodes)] for j in range(hiddenNodes)]
        #self.hiddenBias = [random.randint(1,100)/100 * initialWeightsMax for i in range(hiddenNodes)]
        self.outputWeights = [[random.randint(1,100)/100 * initialWeightsMax for i in range(hiddenNodes)] for j in range(outputNodes)]
        #self.outputBias = [random.randint(1,100)/100 * initialWeightsMax for i in range(outputNodes)]
    
    def activations(self, inputs):
        self.hiddenValue = [0]*self.hiddenNodes
        outputValue = [0]*self.outputNodes

        # calculate activations of hidden nodes
        for i in range(self.hiddenNodes):
            accum = 0
            for j in range(self.inputNodes):
                accum += inputs[0][j] * self.hiddenWeights[i][j] #+ self.hiddenBias[i]
            self.hiddenValue[i] = 1/(1+pow(2.718, -accum))
        
        #calculate output layer activations
        for i in range(self.outputNodes):
            accum = 0
            for j in range(self.hiddenNodes):
                accum += self.hiddenValue[j] * self.outputWeights[i][j] #+ self.outputBias[i]
            outputValue[i] = 1/(1+pow(2.718, -accum))
        return outputValue

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
        hiddenErrors = [0]*self.hiddenNodes
        for i in range(self.hiddenNodes):
            accum = 0
            accum += self.outputWeights[outNode][i] * loss
        
        for i in range(self.hiddenNodes):
            hiddenErrors[i] = accum * self.hiddenValue[i] * (1- self.hiddenValue[i])
            for j in range(self.inputNodes):
                self.hiddenWeights[i][j] += hiddenErrors[i] * self.learningRate * self.momentum
            self.outputWeights[outNode][i] += loss * self.learningRate * self.momentum
        # decay momentum
        if self.momentum > self.momentumDecay:
            self.momentum = self.momentum - self.momentumDecay
        if self.learningRate > self.lrDecay:
            self.learningRate = self.learningRate - self.lrDecay





#check if someone has won
def Winner(board):
    #in a row
    for row in board:
        if  row[0] != '-' and (row[0] == row[1] == row[2]):
            return row[0]
    #in a diagonal
    if  board[1][1] != '-' and ((board[0][0] == board[1][1] == board[2][2]) or (board[0][2] == board[1][1] == board[2][0])):
        return board[1][1]
    #in a column
    for i in range(3):
        if board[0][i] != '-' and (board[0][i] == board[1][i] == board[2][i]):
            return board[0][i]
    #no one wins yet
    return 'n'




def game(p1: Network, p2):
    markValues = {'-':1, 'x':2, 'o':3}
    board = [['-']*3,['-']*3,['-']*3]
    winner = 'n'
    gamePlayed = []

    p1turn = random.randint(0,1)
    p2turn = 1-p1turn
    turn = 1
    p1mark = 'x' if p1turn==1 else 'o'
    p2mark = 'x' if p2turn==1 else 'o'
    
    x=0
    # max of 9 moves, unless someone wins
    while x<9 and winner == 'n':
        x += 1
        #p1's turn, the ai
        if p1turn == turn:
            if type(p1) == Network:
                # calculate outputs, and which move to use. input board state, and which piece you are
                boardFlat = [markValues[col] for row in board for col in row]
                boardFlat.append(markValues[p1mark])
                outputs = p1.activations([boardFlat])
                moveWeights = [outputs.index(i) for i in sorted(outputs, reverse=True)]
                        
                # iterate through each move spot suggested, making the highest legal move and passing turn
                for spot in moveWeights:
                    if turn == p1turn:
                        if board[int(spot/3)][spot%3] == '-':
                            board[int(spot/3)][spot%3] = p1mark
                            turn = p2turn
                            gamePlayed.append([outputs[spot],spot,boardFlat])
            

        # if it's p2's turn, random move. If it's allowable, place it and swap turn
        elif p2turn == turn:
            if p2 == 'random':
                while turn == p2turn:
                    spot = random.randint(0,8)
                    if board[int(spot/3)][spot%3] == '-':
                        board[int(spot/3)][spot%3] = p2mark
                        turn = p1turn
            elif p2 =='user':
                print(f'move {x}')
                for row in board:
                    print(f'|{row[0]}|{row[1]}|{row[2]}|')
                while turn == p2turn:
                    spot = int(input('move: '))
                    if board[int(spot/3)][spot%3] == '-':
                        board[int(spot/3)][spot%3] = p2mark
                        turn = p1turn


        # print board state
        if printing and p2 != 'user':
            print(f'move {x}')
            for row in board:
                print(f'|{row[0]}|{row[1]}|{row[2]}|')

        #check for winner, if winner, log win
        winner = Winner(board)
        if winner != 'n':
            if winner == p1mark:
                gamePlayed.append(1)
            else:
                gamePlayed.append(-1)
        
    # no winner and 9 moves played, log draw
    if winner == 'n':
        gamePlayed.append(0)
        if printing:
            print('cat\'s game')
    if printing:
        print(f"AI played as {p1mark}")
    return gamePlayed





if __name__ == '__main__':
    printing = False
    training = True
    gamesToPlay = 2000
 
    p1net = Network()
 #   results = []
    trainGames = []
    while gamesToPlay > 0:
        gamePlayed = game(p1net, 'random') #can also pass in 'user' as second argument to play manually
 #       results.append(gamePlayed)
        trainGames.append(gamePlayed)

        if len(trainGames) == p1net.trainBatchSize:
            p1net.updateWeights(trainGames)
            trainGames = []
        gamesToPlay -= 1
        print(gamesToPlay)
    while True:
        printing = True
        game(p1net, 'user')

'''
    with open('pico\\learningAIresultsBigger.txt', 'w') as f:
        for entry in results:
            f.write(f'{entry[0]},{entry[1]},{entry[2]}\n')
    f.close
'''