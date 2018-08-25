# FAF-Setons-PyTorch-Project
Neural Networks (PyTorch) ● JSON ● Plotly ● Urllib ● RSQL filters ● Numpy


The network retrieves data from the FAF open-source database using RSQL filters in JSON form. This data file is about 180MB and contains 813,000 records. This is processed and filtered to 51,000 valid cases. It is fed to a sequential neural network created with PyTorch. The goal of the project was to predict the outcome of strategy game matchups using only information provided about the game from the database (this info includes the player positions on the map, the "faction" of the player, and two aspects of the players rating, somewhat similar to ELO in chess). 

# Details on the network
The network has two hidden layers of 128 neurons each. I had issues with over-fitting using a higher amount of neurons. Due to there being multiple data channels (map position, faction, rating mean+deviation) I needed to normalize my data, which I achieved by using BatchNorm2d with a batch size of 50. Each case has an 3d input matrix of dimensions 2x4x8. I used the Adam optimizer, the ReLu activation function, and the MSE loss function. I used Sigmoid for the output layer. I performed 1000 iterations through the entire dataset for training, which resulted in the network outperforming the in-game trueskill game balancing system. 

| Aspect      | TrueSkill | Neural Network |
| ----------- | --------- | -------------- |
| percentage  | 0.768     | 0.818          |
| correlation | 0.527     | 0.646          |


                          
[This is a graph of the improvement of the network over iterations](https://plot.ly/~bsse/31)  
[This graph compares the predictions of this network with the in-game "trueskill" game balancing system](https://plot.ly/~bsse/29)