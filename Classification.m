%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%File Importing                            %
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

FILE_DIR = 'NeuralNet-Solver';
haptAttr = importdata(fullfile(FILE_DIR, '/HAPT/haptAttr.txt'));
haptLabel = importdata(fullfile(FILE_DIR,'/HAPT/haptLabel.txt'));
activity_labels = importdata(fullfile(FILE_DIR,'/HAPT/activity_labels.txt'));
features = importdata(fullfile(FILE_DIR,'/HAPT/features.txt'));
features_info = importdata(fullfile(FILE_DIR,'/HAPT/features_info.txt'));

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%Getting data                              %
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

t = zeros(8000,12);
for i = 1:size(haptLabel)
    t(i,haptLabel(i)) = 1;
end

%each input column entry out of 8000 has 561 attributes 
x = haptAttr'; 
%each output column entry out of 8000 has a '1' indicating the classification
t = t';

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%Getting data                              %
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

%net = patternnet([5,5]);
net = patternnet(5);
[net,tr] = train(net,x,t);
y = net(x);

%Variables to consider tuning
%Number of hidden layers
%Training method
%Number of neurons on the hidden layer
%Type of activation function

%activation function, num of neurons, hidden layers
%classify outputs
%function approximation - mapping of input to single output can use neural
%network with 1 hidden layer

%zeros() for matrix
%transpose ''
%access a matrix - index from 1
%matrix(row,column)
%matrix(:,1:4)
%1:4 = [1 2 3 4]
%; - surpress command window output
% struct - neural net is a struct
%use matlab plot function
%net = patternnet - matlab toolbox for neural networks (or fitnet)
%[net,tr] = train(net,new_______) 
%train function to train network - 

%different iterations using differnet parameters
%generic algorithm?