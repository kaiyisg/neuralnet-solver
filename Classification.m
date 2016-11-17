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

%making ur neural net
net = patternnet(5);

%training and getting results - get graph looking shit
[net,tr] = train(net,x,t);

%input into net to verify - confusion matrix
y = net(x);

%performance parameter
%training method / function
%number of layers / neurons
%dist of data set
%normalization - mapping between -1 / 1 - to handle random inputs

%not impt
%Type of activation function

%how to do it
%net = patternnet - matlab toolbox for neural networks (or fitnet)
%[net,tr] = train(net,new_______) 

%generic algorithm?