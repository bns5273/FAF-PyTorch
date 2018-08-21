import numpy
import json
import urllib.request
import plotly
import torch
import torch.nn as nn
from math import sqrt
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


# numpy.random.seed(26)
device = torch.device("cuda:0")


url = "https://api.faforever.com/data/gamePlayerStats?" \
      "fields[gamePlayerStats]=afterDeviation,afterMean,faction,score,startSpot,game" \
      "&filter=game.featuredMod.id==0;" \
              "game.mapVersion.id==560;" \
              "game.validity=='VALID';" \
              "scoreTime>'2000-01-01T12%3A00%3A00Z'" \
      "&page[size]=10000" \
      "&page[number]="

download = False

with open('/home/brett/PycharmProjects/data/setons.json', 'r') as infile:
    data = json.loads(infile.read())
if download:
    for p in range(1, 100):  # pages. 87?
        with urllib.request.urlopen(url + str(p)) as j:
            print(url + str(p))
            new = json.loads(j.read())
            new = new['data']
            data = data + new
        if new.__len__() == 0:
            break

    with open('/home/brett/PycharmProjects/data/setons.json', 'w') as outfile:
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

    if validate(match):                 # null values, player position errors, one win + one loss
        #       players
        # 2x4x8
        x = [[[0, 0, 0, 0, 0, 0, 0, 0],     # faction, mean
              [0, 0, 0, 0, 0, 0, 0, 0],
              [0, 0, 0, 0, 0, 0, 0, 0],
              [0, 0, 0, 0, 0, 0, 0, 0]],
             [[0, 0, 0, 0, 0, 0, 0, 0],     # faction, dev
              [0, 0, 0, 0, 0, 0, 0, 0],
              [0, 0, 0, 0, 0, 0, 0, 0],
              [0, 0, 0, 0, 0, 0, 0, 0]]]
        for i in range(8):
            f = int(match[i]['faction']) - 1
            m = match[i]['afterMean']
            d = match[i]['afterDeviation']
            # x[i][f][0] = match[i]['afterMean']
            # x[i][f][1] = match[i]['afterDeviation']
            x[0][f][i] = m
            x[1][f][i] = d
        results.append(isWinner(match[0::2]))
        estimates.append(trueskill(match))
        matches.append(torch.Tensor(x).float())

results = torch.Tensor(results).float()
# print('dl\'ed', data.__len__())
# print('8 pls', fullGames.__len__())
# print('valid', matches.__len__())


# nn


torch.set_printoptions(precision=3, linewidth=200)

# class for reshaping within sequential net
class View(nn.Module):
    def __init__(self, *shape):
        super(View, self).__init__()
        self.shape = shape
    def forward(self, input):
        return input.view(self.shape)


net = nn.Sequential(
    nn.GroupNorm(1, 2, affine=False),            # 2x4x8 normalizes mean + dev separately
    nn.Conv1d(4, 1, 4, dilation=2, bias=False),  # 2x1x2 weights for faction-position
    View(4),
    nn.Linear(4, 1),                             # 1x1x1 weights for mean vs dev by team
    nn.Sigmoid()
)

loss_fn = nn.MSELoss(size_average=False)
optimizer = torch.optim.Adam(net.parameters(), lr=.008)

for i in range(10):
    predictions = []
    for x, y in zip(matches, results):
        y_pred = net(x)
        # print(y_pred)

        loss = loss_fn(y_pred, y)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        predictions.append((float(y_pred) > .5) == bool(y))

    print(i, '\t', numpy.mean(predictions))
    for param in net.parameters():
        print(param.data)


# analysis


perf = []
nnet = []
tsk = []
for match, mu, r in zip(matches, estimates, results):
    perf.append(float(net(match)))
    nnet.append((float(net(match)) > .5) == r)
    tsk.append((cdf(mu) > .5) == r)
nnet = numpy.array(nnet)
tsk = numpy.array(tsk)

for param in net.parameters():
    print(param.data)

print('trueskill perf: ', tsk.mean(), numpy.corrcoef(estimates, results)[0][1])
print('neuralnet perf: ', nnet.mean(), numpy.corrcoef(perf, results)[0][1])
print('ts | nn', numpy.corrcoef(estimates, perf)[0][1])

neuralvstrueskill = [plotly.graph_objs.Scatter(
    x=estimates,
    y=perf,
    mode='markers'
)]

plotly.plotly.plot(neuralvstrueskill, filename='neuralvstrueskill')
