%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%File Importing                            %
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

FILE_DIR = 'NeuralNet-Solver';
studentAttrStruct = importdata(fullfile(FILE_DIR, '/Students/students.csv'));
studentAttrInt = studentAttrStruct.data;
studentAttrTextData = studentAttrStruct.textdata;

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%Formating data to double[] for neural net %
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
keySet =   {'M','F','GP','MS','U','R','GT3','LE3','A','T','at_home','health','services','teacher','home','reputation','course','mother','father','yes','no'};
valueSet = [1,0,1,0,1,0,1,0,1,0,1,2,3,4,1,2,3,1,2,1,0];
mapObj = containers.Map(keySet,valueSet);

%String 'other' is 5 for cols mjob,fjob, 4 for col reason, 3 col for guardian
keySetOther =   {'Mjob','Fjob','reason','guardian'};
valueSetOther = [5,5,4,3];
mapObjOthers = containers.Map(keySetOther,valueSetOther);

t_without_g1g2 = zeros(350,30);

for i = 1:350 
    for j = 1:23
        curr = char(studentAttrTextData(i+1,j)); %skipping the header row
        [num, status] = str2num(curr);
        if status
            t_without_g1g2(i,j) = num;
        elseif strcmp('other',curr) 
            header = char(studentAttrTextData(1,j));
            t_without_g1g2(i,j) = mapObjOthers(header);
        else
            t_without_g1g2(i,j) = mapObj(curr);
        end
    end
end

%Don't need to iterate the last 3 columns of g1g2g3
for i = 1:350 
    for j = 1:7
        t_without_g1g2(i,j+23) = char(studentAttrInt(i,j));
    end
end

t_with_g1g2 = zeros(350,32);
t_with_g1g2(:,1:30)= t_without_g1g2(:,:);
for i = 1:350 
    for j = 8:9
        t_with_g1g2(i,j+23) = char(studentAttrInt(i,j));
    end
end

g3 = zeros(350,1);
for i = 1:350 
    g3(i,1) = char(studentAttrInt(i,10));
end

% training parameters 
net.trainParam.epochs = 1000;	
net.trainParam.goal = 0;	
net.trainParam.max_fail	= 6;	
snet.trainParam.min_grad = 1e-7;	
net.trainParam.mu = 0.001;	
net.trainParam.mu_dec = 0.1;
net.trainParam.mu_inc = 10;	
net.trainParam.show = 25;	
net.trainParam.showCommandLine = false;
net.trainParam.showWindow = true;	
net.trainParam.time = inf;	


x = t_without_g1g2'; 
t = g3';

% %% training the network (default setting)
net = fitnet(20);
net = train(net,x,t);
y = net(x);


% %% fitting the data using 2 hidden layers
% 
% net = fitnet([10 10]);
% net = train(net,studentAttr,studentGrade_G3);
% y = net(x);
%% fitting the data with additional G1 and G2 as inputs

%net = fitnet([1]); %the different number of layers and neurons are adjsuted here
%net = train(net,studentAttr_G1G2,studentGrade_G3);
%y = net(x);

% %% fitting the data with additional G1 and G2 as inputs and using trainbr instead of trainln
% 
% net.trainFcn = 'trainbr';
% net = fitnet(10);
% 
% net = train(net,studentAttr_G1G2,studentGrade_G3);
% y = net(x);
