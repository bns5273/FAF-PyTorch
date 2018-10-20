# FAF-Setons-PyTorch-Project
PyTorch ● JSON ● Plotly ● Numpy ● Pandas ● RSQL


This sequential neural network retrieves data from the FAF open-source database using RSQL filters in JSON form. This data file is about 180MB and contains 813,000 records. This is processed and filtered to 26,000 valid cases. It is fed to a sequential neural network created with PyTorch. The goal of the project was to predict the outcome of strategy game matchups using only information provided about the game from the database (this info includes the player positions on the map, the "faction" of the player, and two aspects of the players rating, somewhat similar to ELO rankings in chess).  

[This is a short video describing the game](https://www.youtube.com/watch?v=lQMSPHsyWX8)  

# Details on the network
The network has two hidden layers of 128 neurons each. Due to there being multiple data channels (map position, faction, rating mean+deviation) I needed to normalize my data, which I achieved by using BatchNorm2d with a batch size of 50. Each channel is normalized separately and form a 3d input matrix of dimensions 2x4x8. I used the Adam optimizer, the ReLu activation function, and the MSE loss function. I used Sigmoid for the output layer. I performed 50 iterations through a randomized dataset of ~25,000 cases for training, and it is continuously tested using a random sample of cases. The network outperforms the in-game trueskill game balancing system by a significant margin.  

| Aspect      | TrueSkill | PyTorch |
| ----------- | --------- | -------------- |
| percentage  | 0.768     | 0.910          |
| correlation | 0.527     | 0.841          |


                          
[This is a graph of the improvement of the network over iterations](https://plot.ly/~bsse/31)  
[This graph compares the predictions of this network with the in-game "trueskill" game balancing system](https://plot.ly/~bsse/29)