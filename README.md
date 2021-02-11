# DeepFlu
Forecasting Influenza Susceptibility based on pre-exposure gene expression
DeepFlu layout:
  One input layer(kernel initializer: uniform, activation: relu)
  Four hidden layers with 100 nodes each(kernel initializer: uniform, activation: relu)
  One output layer(kernel initializer: uniform, activation: sigmoid)
Other important parameters or features:
  Dropout: first hidden layer only (0.1)
  Loss: binary crossentropy
  Optimizer: adam
  Epochs: 150
  Batch size: 200
