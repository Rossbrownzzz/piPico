import matplotlib.pyplot as plt


ngames = 200000

p1wins = [0]
p2wins = [0]
draws = [0]
p1winp = [0]
p2winp = [0]
drawp = [0]
with open('pico\\learningAIresults.txt', 'r') as f:
    f.readline()
    f.readline()
    lines = f.readlines()
    i = 0
    for line in lines:
        if i < ngames:
            line = line.strip('\n')
            entry = line.split(',')

            p1wins.append(p1wins[-1])
            p2wins.append(p2wins[-1])
            draws.append(draws[-1])

            if entry[1] == 'p1':
                p1wins[-1] = p1wins[-1] +1
            elif entry[1] == 'p2':
                p2wins[-1] = p2wins[-1] +1
            else:
                draws[-1] = draws[-1]+1
            
            p1winp.append(p1wins[-1]/len(p1wins))
            p2winp.append(p2wins[-1]/len(p2wins))
            drawp.append(draws[-1]/len(draws))

        i+= 1
        
    f.close()

'''
p1winsd = [0]
p2winsd = [0]
drawsd = [0]
p1winpd = [0]
p2winpd = [0]
drawpd = [0]
with open('pico\\learningAIresultsx2.txt', 'r') as f:
    f.readline()
    f.readline()
    lines = f.readlines()
    i = 0
    for line in lines:
        if i < ngames:
            line = line.strip('\n')
            entry = line.split(',')

            p1winsd.append(p1winsd[-1])
            p2winsd.append(p2winsd[-1])
            drawsd.append(drawsd[-1])

            if entry[1] == 'p1':
                p1winsd[-1] = p1winsd[-1] +1
            elif entry[1] == 'p2':
                p2winsd[-1] = p2winsd[-1] +1
            else:
                drawsd[-1] = drawsd[-1]+1
            
            p1winpd.append(p1winsd[-1]/len(p1winsd))
            p2winpd.append(p2winsd[-1]/len(p2winsd))
            drawpd.append(drawsd[-1]/len(drawsd))

        i+= 1
        
    f.close()
'''

fig, ax = plt.subplots()

ax.plot(p1winp, label='AIwin%')
ax.plot(p2winp, label='randomwin%')
ax.plot(drawp, label='draw%')
'''
ax.plot(p1winpd, label='noMark_AIwin%')
ax.plot(p2winpd, label='noMark_randomwin%')
ax.plot(drawpd, label='noMark_draw%')
'''

ax.legend()
plt.show()