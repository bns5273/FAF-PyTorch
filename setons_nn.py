from numpy import mean
import pandas as pd
import random
import json
import matplotlib as plt
import plotly.graph_objs as go
import plotly.plotly as py
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader


# determines the winning team based on player scores
def winner(team):
    return max([team[x]['score'] for x in range(4)]) > 0


# confirms a match as valid
def validate(players):
    for player in players:
        if player is None:                  # player not present
            return False
        if player['afterMean'] is None:     # not rated
            return False
        if player['faction'] > 4:           # modded faction
            return False
    if winner(players[0::2]) == winner(players[1::2]):  # corrupt replay
        return False
    return True


# class for reshaping within sequential net
class View(nn.Module):
    def __init__(self, *shape):
        super(View, self).__init__()
        self.shape = shape

    def forward(self, input):
        return input.view(self.shape)


class SetonsDataset(Dataset):
    def __init__(self, filename):
        print('reading...')
        self.data = []
        file = open(filename, 'r').read()
        df = pd.DataFrame.from_dict(json.loads(file))
        games = list(df.groupby(
            df.relationships.apply(lambda x: x['game']['data']['id']),
            sort=False))

        print('ingesting...')
        for game in games:
            gid, frame = game
            players = [None] * 8
            for a, i, r, t in frame.values:
                position = a['startSpot'] - 1
                players[position] = a
            if validate(players):
                # 2x4x8 -> mean-dev xx faction xx position
                x = [[[0, 0, 0, 0, 0, 0, 0, 0],  # mean
                      [0, 0, 0, 0, 0, 0, 0, 0],
                      [0, 0, 0, 0, 0, 0, 0, 0],
                      [0, 0, 0, 0, 0, 0, 0, 0]],
                     [[0, 0, 0, 0, 0, 0, 0, 0],  # deviation
                      [0, 0, 0, 0, 0, 0, 0, 0],
                      [0, 0, 0, 0, 0, 0, 0, 0],
                      [0, 0, 0, 0, 0, 0, 0, 0]]]
                y = winner(players[0::2])
                for i in range(8):  # fill in x
                    f = players[i]['faction'] - 1
                    m = players[i]['afterMean']
                    d = players[i]['afterDeviation']
                    x[0][f][i] = m
                    x[1][f][i] = d
                self.data.append((torch.FloatTensor(x), y))

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx]


# CUDA for PyTorch
use_cuda = torch.cuda.is_available()
device = torch.device("cuda:0" if use_cuda else "cpu")
print('cuda: ', use_cuda)

torch.set_printoptions(threshold=5000, precision=3, linewidth=200)
data = SetonsDataset('setons.json')
head = data[0]
print(len(data), 'matches')
print(head)

epochs = 50
training_gen = DataLoader(data, batch_size=50, shuffle=True, num_workers=2)

net = nn.Sequential(
    nn.BatchNorm2d(2),  # Bx2x4x8
    View(-1, 64),       # 2x4x8 -> 1x1x64
    nn.Linear(64, 128),
    nn.ReLU(),
    nn.Linear(128, 1),
    View(-1),           # 1x1 int array -> int
    nn.Sigmoid()        # output layer
)

loss_fn = nn.MSELoss()
optimizer = torch.optim.Adam(net.parameters(), lr=.005)

# graph_correlation = []
# graph_percentage = []
# graph_ts_correlation = []
# graph_ts_percentage = []
# for j in range(epochs):
#     training_data, training_labels = zip(*random.sample(list(zip(matches, results)), 40000))
#     testing_data, testing_labels = zip(*random.sample(list(zip(matches, results)), 1000))
#
#     # training
#     for i in range(0, len(training_data), batch_size):
#         x = torch.stack(training_data[i:i + batch_size])
#         y = torch.Tensor(training_labels[i:i + batch_size])
#
#         y_pred = net(x)
#         loss = loss_fn(y_pred, y)
#         optimizer.zero_grad()
#         loss.backward()
#         optimizer.step()
#
#     # testing
#     predictions = []
#     coin_predict = []
#     coin_ts_predict = []
#     for t in range(0, len(testing_data), batch_size):
#         x = torch.stack(testing_data[t:t + batch_size])
#         predictions += list(net(x).data)
#     for p, t, r in zip(predictions, testing_trueskill, testing_labels):
#         coin_predict.append((p.data > .5) == r)
#         coin_ts_predict.append((t > .5) == r)
#
#     graph_correlation.append(np.corrcoef(predictions, testing_labels)[0][1])
#     graph_percentage.append(np.mean(coin_predict))
#     graph_ts_correlation.append(np.corrcoef(testing_trueskill, testing_labels)[0][1])
#     graph_ts_percentage.append(np.mean(coin_ts_predict))
#     print(j, graph_percentage[-1], graph_correlation[-1])


# ''' Analysis '''
#
# # for param in net.parameters():
# #     print(param.data)
#
# vsTrueskill = [
#     go.Histogram(
#         name='TrueSkill',
#         x=testing_trueskill,
#         opacity=.75
#     ),
#     go.Histogram(
#         name='Neural Network',
#         x=predictions,
#         opacity=.75
#     )
# ]
# layout = go.Layout(barmode='overlay')
# fig = go.Figure(data=vsTrueskill, layout=layout)
# py.plot(fig, filename='neuralvstrueskill')
#
# netvtime = [
#     go.Scatter(
#         name='percentage',
#         x=list(range(len(graph_percentage))),
#         y=graph_percentage,
#         mode='lines'
#     ),
#     go.Scatter(
#         name='correlation',
#         x=list(range(len(graph_correlation))),
#         y=graph_correlation,
#         mode='lines'
#     ),
#     go.Scatter(
#         name='trueskill percentage',
#         x=list(range(len(graph_percentage))),
#         y=graph_ts_percentage,
#         mode='lines'
#     ),
#     go.Scatter(
#         name='trueskill correlation',
#         x=list(range(len(graph_correlation))),
#         y=graph_ts_correlation,
#         mode='lines'
#     )
# ]
# py.plot(netvtime, filename='netvtime')


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
