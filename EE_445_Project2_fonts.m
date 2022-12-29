clear all;clc;
% define input for letter and fonts from A to K, 3 fonts for each letter 
% '#' is 1, '.' is -1

p=[-1 -1 1 1 -1 -1 -1, -1 -1 -1 1 -1 -1 -1, -1 -1 -1 1 -1 -1 -1, -1 -1 1 -1 1 -1 -1,...
   -1 -1 1 -1 1 -1 -1, -1 1 1 1 1 1 -1, -1 1 -1 -1 -1 1 -1, -1 1 -1 -1 -1 1 -1, 1 1 1 -1 1 1 1 1;...

   -1 -1 -1 1 -1 -1 -1, -1 -1 -1 1 -1 -1 -1, -1 -1 -1 1 -1 -1 -1, -1 -1 1 -1 1 -1 -1,...
   -1 -1 1 -1 1 -1 -1, -1 1 -1 -1 -1 1 -1, -1 1 1 1 1 1 -1, -1 1 1 1 1 1 -1, -1 1 -1 -1 -1 1 -1 1;...

   -1 -1 -1 1 -1 -1 -1, -1 -1 -1 1 -1 -1 -1, -1 -1 1 -1 1 -1 -1, -1 -1 1 -1 1 -1 -1,...
   -1 1 -1 -1 -1 1 -1, -1 1 1 1 1 1 -1, 1 -1 -1 -1 -1 -1 1, 1 -1 -1 -1 -1 -1 1, 1 1 -1 -1 -1 1 1 1;... % A fonts and letter
   
   
   1 1 1 1 1 1 -1, -1 1 -1 -1 -1 -1 1, -1 1 -1 -1 -1 -1 1, -1 1 -1 -1 -1 -1 1,...
   -1 1 1 1 1 1 -1, -1 1 -1 -1 -1 -1 1, -1 1 -1 -1 -1 -1 1, -1 1 -1 -1 -1 -1 1, 1 1 1 1 1 1 -1 1;...

   1 1 1 1 1 1 -1, 1 -1 -1 -1 -1 -1 1, 1 -1 -1 -1 -1 -1 1, 1 -1 -1 -1 -1 -1 1,...
   1 1 1 1 1 1 -1, 1 -1 -1 -1 -1 -1 1, 1 -1 -1 -1 -1 -1 1, 1 -1 -1 -1 -1 -1 1, 1 1 1 1 1 1 -1 1;...
   
   1 1 1 1 1 1 -1, -1 1 -1 -1 -1 -1 1, -1 1 -1 -1 -1 -1 1, -1 1 1 1 1 1 -1,...
   -1 1 -1 -1 -1 -1 1, -1 1 -1 -1 -1 -1 1, -1 1 -1 -1 -1 -1 1, -1 1 -1 -1 -1 -1 1, 1 1 1 1 1 1 -1 1;... %B 
   
   
   -1 -1 1 1 1 1 1, -1 1 -1 -1 -1 -1 1, 1 -1 -1 -1 -1 -1 -1, 1 -1 -1 -1 -1 -1 -1,...
   1 -1 -1 -1 -1 -1 -1, 1 -1 -1 -1 -1 -1 -1, 1 -1 -1 -1 -1 -1 -1, -1 1 -1 -1 -1 -1 1, -1 -1 1 1 1 1 -1 1;...
   
   -1 -1 1 1 1 -1 -1, -1 1 -1 -1 -1 1 -1, 1 -1 -1 -1 -1 -1 1, 1 -1 -1 -1 -1 -1 -1,...
   1 -1 -1 -1 -1 -1 -1, 1 -1 -1 -1 -1 -1 -1, 1 -1 -1 -1 -1 -1 1, -1 1 -1 -1 -1 1 -1, -1 -1 1 1 1 -1 -1 1;...
   
   -1 -1 1 1 1 -1 1, -1 1 -1 -1 -1 1 -1, 1 -1 -1 -1 -1 -1 1, 1 -1 -1 -1 -1 -1 -1,...
   1 -1 -1 -1 -1 -1 -1, 1 -1 -1 -1 -1 -1 -1, 1 -1 -1 -1 -1 -1 1, -1 1 -1 -1 -1 1 -1, -1 -1 1 1 1 -1 -1 1;... %C
   
   
   1 1 1 1 1 -1 -1, 1 1 -1 -1 -1 1 -1, -1 1 -1 -1 -1 -1 1, -1 1 -1 -1 -1 -1 1,...
   -1 1 -1 -1 -1 -1 1, -1 1 -1 -1 -1 -1 1, -1 1 -1 -1 -1 -1 1, -1 1 -1 -1 -1 1 -1, 1 1 1 1 1 -1 -1 1;...

   1 1 1 1 1 -1 -1, 1 -1 -1 -1 -1 1 -1, 1 -1 -1 -1 -1 -1 1, 1 -1 -1 -1 -1 -1 1,...
   1 -1 -1 -1 -1 -1 1, 1 -1 -1 -1 -1 -1 1,1 -1 -1 -1 -1 -1 1, 1 -1 -1 -1 -1 1 -1, 1 1 1 1 1 -1 -1 1;

   1 1 1 1 1 -1 -1, -1 1 -1 -1 -1 1 -1, -1 1 -1 -1 -1 -1 1, -1 1 -1 -1 -1 -1 1,...
   -1 1 -1 -1 -1 -1 1, -1 1 -1 -1 -1 -1 1, -1 1 -1 -1 -1 -1 1, -1 1 -1 -1 -1 1 -1, 1 1 1 1 1 -1 -1 1;... %D
   
   
   1 1 1 1 1 1 1, -1 1 -1 -1 -1 -1 1, -1 1 -1 -1 -1 -1 -1, -1 1 -1 1 -1 -1 -1,...
    -1 1 1 1 -1 -1 -1, -1 1 -1 1 -1 -1 -1, -1 1 -1 -1 -1 -1 -1, -1 1 -1 -1 -1 -1 1,1 1 1 1 1 1 1 1;...
    
    1 1 1 1 1 1 1, 1 -1 -1 -1 -1 -1 -1, 1 -1 -1 -1 -1 -1 -1, 1 -1 -1 -1 -1 -1 -1,...
    1 1 1 1 1 -1 -1, 1 -1 -1 -1 -1 -1 -1, 1 -1 -1 -1 -1 -1 -1, 1 -1 -1 -1 -1 -1 -1,1 1 1 1 1 1 1 1;...
    
    1 1 1 1 1 1 1, -1 1 -1 -1 -1 -1 1, -1 1 -1 -1 1 -1 -1, -1 1 -1 1 -1 -1 -1,...
    -1 1 1 1 -1 -1 -1, -1 1 -1 1 -1 -1 -1, -1 1 -1 -1 -1 -1 -1, -1 1 -1 -1 -1 -1 1,1 1 1 1 1 1 1 1; %E
  
    
    
    -1 -1 -1 1 1 1 1, -1 -1 -1 -1 -1 1 -1, -1 -1 -1 -1 -1 1 -1, -1 -1 -1 -1 -1 1 -1,...
    -1 -1 -1 -1 -1 1 -1, -1 -1 -1 -1 -1 1 -1, -1 1 -1 -1 -1 1 -1, -1 1 -1 -1 -1 1 -1, -1 -1 1 1 1 -1 -1 1;
    
    -1 -1 -1 -1 -1 1 -1, -1 -1 -1 -1 -1 1 -1, -1 -1 -1 -1 -1 1 -1, -1 -1 -1 -1 -1 1 -1,...
    -1 -1 -1 -1 -1 1 -1, -1 -1 -1 -1 -1 1 -1, -1 1 -1 -1 -1 1 -1, -1 1 -1 -1 -1 1 -1, -1 -1 1 1 1 -1 -1 1;
    
    -1 -1 -1 -1 1 1 1, -1 -1 -1 -1 -1 1 -1, -1 -1 -1 -1 -1 1 -1, -1 -1 -1 -1 -1 1 -1,...
    -1 -1 -1 -1 -1 1 -1, -1 -1 -1 -1 -1 1 -1, -1 1 -1 -1 -1 1 -1, -1 1 -1 -1 -1 1 -1, -1 -1 1 1 1 -1 -1 1;%J
    
    
    
    1 1 1 -1 -1 1 1, -1 1 -1 -1 1 -1 -1, -1 1 -1 1 -1 -1 -1, -1 1 1 -1 -1 -1 -1,... 
    -1 1 1 -1 -1 -1 -1, -1 1 -1 1 -1 -1 -1, -1 1 -1 -1 1 -1 -1, -1 1 -1 -1 -1 1 -1, 1 1 1 -1 -1 1 1 1;
    
   1 -1 -1 -1 -1 1 -1, 1 -1 -1 -1 1 -1 -1, 1 -1 -1 1 -1 -1 -1, 1 -1 1 -1 -1 -1 -1, ...
    1 1 -1 -1 -1 -1 -1, 1 -1 1 -1 -1 -1 -1, 1 -1 -1 1 -1 -1 -1, 1 -1 -1 -1 1 -1 -1,1 -1 -1 -1 -1 1 -1 1;
    
   1 1 1 -1 -1 1 1, -1 1 -1 -1 -1 1 -1, -1 1 -1 -1 1 -1 -1, -1 1 -1 1 -1 -1 -1,... 
    -1 1 1 -1 -1 -1 -1, -1 1 -1 1 -1 -1 -1, -1 1 -1 -1 1 -1 -1, -1 1 -1 -1 -1 1 -1, 1 1 1 -1 -1 1 1 1]; %K
p=p';
t=[-1 -1; -1 1; 1 -1 ; 
   -1 -1; -1 1; 1 -1; 
   -1 -1; -1 1; 1 -1;
   -1 -1; -1 1; 1 -1;
  -1 -1; -1 1; 1 -1;
  -1 -1; -1 1; 1 -1;
 -1 -1; -1 1; 1 -1];
t=t';


% Training
no_nodes=10;
fprintf('Number of nodes: %2.0f\n',no_nodes);

net=feedforwardnet(no_nodes,...
    'traingda');
net=configure(net,p,t);
net.divideFcn='';
net.trainParam.show=100;
net.trainParam.epochs=5000;
net.trainParam.goal=1e-7;
[net,tr]=train(net,p,t);
a=sim(net,p);
fprintf('Training result                   Target\n')
disp([a',t'])
%% Testing
ptest=[1 1 1 -1 -1 1 1
    -1 1 -1 -1 1 -1 -1
    -1 1 -1 1 -1 -1 -1
    -1 1 1 -1 -1 -1 -1
    -1 1 1 -1 -1 -1 -1
    -1 1 -1 1 -1 -1 -1
    -1 1 -1 -1 1 -1 -1
    -1 1 -1 -1 -1 1 -1
    1 1 1 -1 -1 1 1 ];
hintonw(ptest);
ptest=[1 1 1 -1 -1 1 1, -1 1 -1 -1 1 -1 -1, -1 1 -1 1 -1 -1 -1, -1 1 1 -1 -1 -1 -1,... 
    -1 1 1 -1 -1 -1 -1, -1 1 -1 1 -1 -1 -1, -1 1 -1 -1 1 -1 -1, -1 1 -1 -1 -1 1 -1, 1 1 1 -1 -1 1 1 1];
ptest=ptest';
atest=sim(net,ptest)
