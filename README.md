This module trains a NN and gives the weight distribution after each layer. 

The Parser has the following Arguments: 
"-a", "--activation": Choses the activation function. Possible values: 
    "ReLu"
    "Sigmoidal"
    "Tanh"
    "Softsign"
"-i", "--initialisation": Chooses the initialisation of weights. Possible values: 
    "uniform": a uniform distribution between -1 and 1
    "normal": a normal distribution around 0 with sigma=1
    "uniform/10": a uniform distribution between -0.1 and 0.1
    "normal/10": a normal distribution around 0 with sigma=0.1
    "xavier": the initialisation proposed in this paper, which this tool was originally developed to test: http://proceedings.mlr.press/v9/glorot10a/glorot10a.pdf
"--learningrate": a float, typically 0.001
"-e", "--epochs": give the integer value for the number of epochs, typically 10-100
"-d", "--depth": give the integer for the depth of the network. For a good visualisation, this should not exceed 8.
"-w", "--width": give the integer for the width of the network. Typically around 1000
"-s", "--datasetsize": give the integer how many samples of the 60000 samples dataset should be used. 60000 is adviseable, but a lower number speeds up training. 