import numpy
import json
import urllib.request
import torch
import torch.nn as nn
from math import sqrt
# import trueskill
from trueskill import BETA
from trueskill.backends import cdf


def pRating(p):
    return p['afterMean'] - 3 * p['afterDeviation']


def isWinner(team):
    return max(team[0]['score'], team[1]['score'], team[2]['score'], team[3]['score']) > 0


def validate(m):
    for p in m:
        if p['afterMean'] is None:
            return False
        if p['faction'] > 4:
            return False
    if match.__contains__(0):           # needed for some entries where 2 players possess same spot?
        return False
    if isWinner(match[0::2]) == isWinner(match[1::2]):
        return False
    return True


def trueskill(m):
    delta_mean = m[0]['afterMean'] + m[2]['afterMean'] + m[4]['afterMean'] + m[6]['afterMean'] - \
                 m[1]['afterMean'] - m[3]['afterMean'] - m[5]['afterMean'] - m[7]['afterMean']
    dev_a = m[0]['afterDeviation'] + m[2]['afterDeviation'] + m[4]['afterDeviation'] + m[6]['afterDeviation']
    dev_b = m[1]['afterDeviation'] + m[3]['afterDeviation'] + m[5]['afterDeviation'] + m[7]['afterDeviation']
    denom = sqrt(2 * (BETA * BETA) + pow(dev_a, 2) + pow(dev_b, 2))
    return delta_mean / denom


url = "https://api.faforever.com/data/gamePlayerStats?" \
      "fields[gamePlayerStats]=afterDeviation,afterMean,faction,score,startSpot,game" \
      "&filter=game.featuredMod.id==0;" \
              "game.mapVersion.id==560;" \
              "game.validity=='VALID';" \
              "afterDeviation<100;" \
              "scoreTime>'2000-01-01T12%3A00%3A00Z'" \
      "&page[size]=10000" \
      "&page[number]="

download = False

with open('/home/brett/PycharmProjects/data/setons.json', 'r') as infile:
    data = json.loads(infile.read())
if download:
    for p in range(43, 60):  # pages. pg 42 was incomplete
        with urllib.request.urlopen(url + str(p)) as j:
            print(url + str(p))
            new = json.loads(j.read())
            new = new['data']
            data = data + new
        if new.__len__() == 0:
            break

    with open('setons.json', 'w') as outfile:
        json.dump(data, outfile)


fullGames = []
i = 0
while i < data.__len__() - 1:
    gid = data[i]['relationships']['game']['data']['id']
    players = []
    while i < data.__len__() - 1 and data[i]['relationships']['game']['data']['id'] == gid:
        players.append(data[i]['attributes'])
        i += 1
    if players.__len__() == 8:
        fullGames.append(players)


matches = []
results = []
estimates = []
for players in fullGames:
    match = [0, 0, 0, 0, 0, 0, 0, 0]
    for player in players:
        i = int(player['startSpot']) - 1
        match[i] = player
    if validate(match):
        x = [0, 0, 0, 0, 0, 0, 0, 0]
        for i in range(8):
            x[i] = match[i]['afterMean']
        matches.append(torch.Tensor(x).float())
        results.append(isWinner(match[0::2]))
        estimates.append(trueskill(match))


results = torch.Tensor(results).float()
estimates = numpy.array(estimates)
print('dl\'ed', data.__len__())
print('8 pls', fullGames.__len__())
print('valid', matches.__len__())


# nn

'''
ReLU        4974
ReLU6       4973
LeakyReLU   4960
Tanh        4959
RReLU       4736
Sigmoid     4639
SELU        4615
ELU         4540

'''


# numpy.random.seed(26)
device = torch.device("cuda:0")


net = nn.Sequential(
    nn.LayerNorm(8, elementwise_affine=False),
    nn.Linear(8, 16),
    nn.ReLU(),
    nn.Linear(16, 1)
)

loss_fn = nn.MSELoss()

for i in range(5):
    for x, y in zip(matches, results):
        y_pred = net(x)

        # print(x)
        # print(y_pred)

        loss = loss_fn(y_pred, y)

        net.zero_grad()

        loss.backward()

        with torch.no_grad():
            for param in net.parameters():
                param -= .008 * param.grad
        # print(y.item(), y_pred.item())

    print(i, loss.item())


perf = []
n = []
t = []
for match, mu, r in zip(matches, estimates, results):
    perf.append(float(net(match)))
    n.append((float(net(match)) > .5) == r)
    t.append((float(cdf(mu)) > .5) == r)
    # print(r.item(), cdf(mu) > .5, float(net(match)) > .5)
n = numpy.array(n)
t = numpy.array(t)

for param in net.parameters():
    print(param.data)
print('trueskill perf: ', t.mean(), numpy.corrcoef(estimates, results)[0][1])
print('neuralnet perf: ', n.mean(), numpy.corrcoef(perf, results)[0][1])
print('ts | nn', numpy.corrcoef(estimates, perf)[0][1])
