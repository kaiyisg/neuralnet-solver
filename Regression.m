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
