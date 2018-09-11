import numpy as np
import random
import json
import urllib.request
import plotly.graph_objs as go
import plotly.plotly as py
import torch as torch
import torch.nn as nn
from math import sqrt
from trueskill import BETA  # == 4.1666_
from trueskill.backends import cdf


# calculates rating shown in game
def pRating(p):
    return p['afterMean'] - 3 * p['afterDeviation']


# used for validation
def isWinner(team):
    return max(team[0]['score'], team[1]['score'], team[2]['score'], team[3]['score']) > 0


# used for dataset
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


# old algorithm for analysis
def trueskill(m):
    delta_mean = m[0]['afterMean'] + m[2]['afterMean'] + m[4]['afterMean'] + m[6]['afterMean'] - \
                 m[1]['afterMean'] - m[3]['afterMean'] - m[5]['afterMean'] - m[7]['afterMean']
    dev_a = m[0]['afterDeviation'] + m[2]['afterDeviation'] + m[4]['afterDeviation'] + m[6]['afterDeviation']
    dev_b = m[1]['afterDeviation'] + m[3]['afterDeviation'] + m[5]['afterDeviation'] + m[7]['afterDeviation']
    denom = sqrt(2 * (BETA * BETA) + pow(dev_a, 2) + pow(dev_b, 2))
    return delta_mean / denom


# class for reshaping within sequential net
class View(nn.Module):
    def __init__(self, *shape):
        super(View, self).__init__()
        self.shape = shape

    def forward(self, input):
        return input.view(self.shape)


download = False

url = "https://api.faforever.com/data/gamePlayerStats?" \
      "fields[gamePlayerStats]=afterDeviation,afterMean,faction,score,startSpot,game" \
      "&filter=game.featuredMod.id==0;" \
              "game.mapVersion.id==560;" \
              "game.validity=='VALID';" \
              "scoreTime>'2000-01-01T12%3A00%3A00Z'" \
      "&page[size]=10000" \
      "&page[number]="
with open('setons.json', 'r') as infile:
    data = json.loads(infile.read())
    if download:
        for p in range(1, 10):  # pages. 87?
            with urllib.request.urlopen(url + str(p)) as j:
                print(url + str(p))
                data += json.loads(j.read())['data']
            if new.__len__() == 0:
                break
        with open('setons.json', 'w') as outfile:
            json.dump(data, outfile)


# validating. check that all 8 players are present
fullGames = []
i = 0
while i < len(data) - 1:
    gid = data[i]['relationships']['game']['data']['id']
    players = []
    while i < len(data) - 1 and data[i]['relationships']['game']['data']['id'] == gid:
        players.append(data[i]['attributes'])
        i += 1
    if len(players) == 8:
        fullGames.append(players)


# validating
matches = []
results = []
estimates = []
for players in fullGames:
    match = [0, 0, 0, 0, 0, 0, 0, 0]
    for player in players:
        i = int(player['startSpot']) - 1
        match[i] = player

    if validate(match):  # check for null values, player position errors, one win + one loss
        # 2x4x8 -> mean-dev xx faction xx position
        r = isWinner(match[0::2])
        x = [[[0, 0, 0, 0, 0, 0, 0, 0],     # mean
              [0, 0, 0, 0, 0, 0, 0, 0],
              [0, 0, 0, 0, 0, 0, 0, 0],
              [0, 0, 0, 0, 0, 0, 0, 0]],
             [[0, 0, 0, 0, 0, 0, 0, 0],     # deviation
              [0, 0, 0, 0, 0, 0, 0, 0],
              [0, 0, 0, 0, 0, 0, 0, 0],
              [0, 0, 0, 0, 0, 0, 0, 0]]]
        for i in range(8):                  # fill in x
            f = int(match[i]['faction']) - 1
            m = match[i]['afterMean']
            d = match[i]['afterDeviation']
            x[0][f][i] = m
            x[1][f][i] = d

        matches.append(torch.FloatTensor(x))
        results.append(r)
        estimates.append(cdf(trueskill(match)))


''' Neural Network '''


torch.set_printoptions(precision=3, linewidth=200)
batch_size = 50
epochs = 500

net = nn.Sequential(
    nn.BatchNorm2d(2),  # Bx2x4x8
    View(-1, 64),       # 2x4x8 -> 1x1x64
    nn.Linear(64, 256),
    nn.ReLU(),
    nn.Linear(256, 1),
    View(-1),           # 1x1 int array -> int
    nn.Sigmoid()        # output layer
)

loss_fn = nn.MSELoss()
optimizer = torch.optim.Adam(net.parameters(), lr=.005)

graph_correlation = []
graph_percentage = []
graph_ts_correlation = []
graph_ts_percentage = []
for j in range(epochs):
    training_data, training_results = zip(*random.sample(list(zip(matches, results)), 40000))
    testing_data, testing_results, testing_trueskill = zip(*random.sample(list(zip(matches, results, estimates)), 1000))

    # training
    for i in range(0, len(training_data), batch_size):
        x = torch.stack(training_data[i:i + batch_size])
        y = torch.Tensor(training_results[i:i + batch_size])

        y_pred = net(x)
        loss = loss_fn(y_pred, y)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    # testing
    predictions = []
    coin_predict = []
    coin_ts_predict = []
    for t in range(0, len(testing_data), batch_size):
        x = torch.stack(testing_data[t:t + batch_size])
        predictions += list(net(x).data)
    for p, t, r in zip(predictions, testing_trueskill, testing_results):
        coin_predict.append((p.data > .5) == r)
        coin_ts_predict.append((t > .5) == r)

    graph_correlation.append(np.corrcoef(predictions, testing_results)[0][1])
    graph_percentage.append(np.mean(coin_predict))
    graph_ts_correlation.append(np.corrcoef(testing_trueskill, testing_results)[0][1])
    graph_ts_percentage.append(np.mean(coin_ts_predict))
    print(j, graph_percentage[-1], graph_correlation[-1])


''' Analysis '''

# for param in net.parameters():
#     print(param.data)

vsTrueskill = [
    go.Histogram(
        name='TrueSkill',
        x=testing_trueskill,
        opacity=.75
    ),
    go.Histogram(
        name='Neural Network',
        x=predictions,
        opacity=.75
    )
]
layout = go.Layout(barmode='overlay')
fig = go.Figure(data=vsTrueskill, layout=layout)
py.plot(fig, filename='neuralvstrueskill')

netvtime = [
    go.Scatter(
        name='percentage',
        x=list(range(len(graph_percentage))),
        y=graph_percentage,
        mode='lines'
    ),
    go.Scatter(
        name='correlation',
        x=list(range(len(graph_correlation))),
        y=graph_correlation,
        mode='lines'
    ),
    go.Scatter(
        name='trueskill percentage',
        x=list(range(len(graph_percentage))),
        y=graph_ts_percentage,
        mode='lines'
    ),
    go.Scatter(
        name='trueskill correlation',
        x=list(range(len(graph_correlation))),
        y=graph_ts_correlation,
        mode='lines'
    )
]
py.plot(netvtime, filename='netvtime')


''' Sample Run
...
491 0.912 0.8488968814028814
492 0.887 0.796355651765359
493 0.92 0.8546175814834017
494 0.906 0.8338445591501401
495 0.901 0.8343490898674357
496 0.899 0.8295045544916277
497 0.906 0.8286187419581098
498 0.904 0.8364858446798663
499 0.91 0.8412758365749676
'''
