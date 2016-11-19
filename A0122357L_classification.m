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
%Setting up net                            %
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

%making neural net
net = patternnet(100);

% Set up Division of Data for Training, Validation, Testing
net.divideParam.trainRatio = 70/100;
net.divideParam.valRatio = 15/100;
net.divideParam.testRatio = 15/100;

net.trainFcn = 'trainscg';

net.trainParam.max_fail = 6;

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%Training net                              %
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

[net,tr] = train(net,x,t);
y = net(x);

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%Plotting confusion matrix                 %
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    
y_test = net(x(:,tr.testInd));
t_test = t(:,tr.testInd);
plotconfusion(t_test,y_test);

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%Plotting roc matrix                       %
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

[tpr,fpr,th] = roc(t,y);
%plotroc(t,y);