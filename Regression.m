%File Importing
FILE_DIR = 'NeuralNet-Solver';
studentAttr = importdata(fullfile(FILE_DIR, '/Students/students.csv'));
studentGrade = importdata(fullfile(FILE_DIR,'/Students/readme.txt'));

%Variables to consider tuning
%Number of hidden layers
%Training method
%Number of neurons on the hidden layer
%Type of activation function

x = studentAttr'; 
t = studentGrade(:,1)';

net = fitnet(1);
%net = train(net,x,t);
%view(net);
%y = net(x);
