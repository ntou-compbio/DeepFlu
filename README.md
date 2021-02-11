# DeepFlu
Forecasting Influenza Susceptibility based on pre-exposure gene expression using Deep Neural Network
# Layout:
+ One input layer(kernel initializer: uniform, activation: relu)  
+ Four hidden layers with 100 nodes each(kernel initializer: uniform, activation: relu)  
+ One output layer(kernel initializer: uniform, activation: sigmoid)    
Other important parameters or features:
+ Dropout: first hidden layer only (0.1)
+ Loss: binary crossentropy
+ Optimizer: adam
+ Epochs: 150
+ Batch size: 200  
![DEEPFLU](https://github.com/ntou-compbio/DeepFlu/blob/main/DeepFlu_Layout.png)
# Dataset:
[GSE52428](https://www.ncbi.nlm.nih.gov/geo/query/acc.cgi?acc=GSE52428)  
[GSE73072](https://www.ncbi.nlm.nih.gov/geo/query/acc.cgi?acc=GSE73072)
