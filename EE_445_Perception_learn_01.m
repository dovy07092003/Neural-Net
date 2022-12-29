%Perc_learnp_01.m
%New with learnp command
%% Each of the four column vectors
% in P defines a 2-element input vectors;
%% Input Section
clear all;clc;cla;clf;
pause_flag=1;
P=[-0.5 -0.5 0.3 -0.1;...
    -0.5 0.5 -0.5 1.0;...
    1.0 1.0 1.0 1.0];
T=[1 1 0 0];
% Set input layer size R and neuron layer size S
[R,Q]=size(P); [S,Q]=size(T);

%% Initialize network parameters
figure(1);
plotpv(P(1:R-1,:),T); % don't want to plot the bias so get R-1
change_marker % in shared folder Matlab hints

%The perceptron must properly classify the 4 inputs vectors in P
%into the two categories defined by T
%Perceptrons have HARDLIM neurons
% These neurons are capable of separating an input space into two segments
% using a straight one segment target value of 1 and the other 0


W=rand(S,R)
Wp=W(:,1:R-1);
Bp=W(:,R);

%% display initial values
% The input vectors are replotted...
plotpv(P(1:R-1,:),T); %Plot perceptron input/target vectors
%... with the neuron's intial
% random attempt at classification
plotpc(Wp,Bp);
%... we are going to train it!
watchon;%Sets the current figure pointer to the watch
cla;
plotpv(P(1:R-1,:),T);

pause(3);
figure(1);
E=1;
linehandle=plotpc(Wp,Bp);
disp('hit a key to continue')

%Learnp returns a new delta W that performs as
% a better classifier, and the error.This loop allows 
% the network to adapt for one pass,plots the 
% classification line, and continues until the 
%error is zero.
max_epoch=10;
%network learning section

while (sse(E) & (epoch<= max_epoch))
    Ai=hardlim(W*P)
    % Learning Phase
    Ei=T-Ai; %Error
    dWq=learnp(W,P,[],[],[],[],Ei,[],[],[],[],[])
    % Presentation Phase
    disp('pause')
    W=W+dWq;
    Wp= W(:,1:R-1)% Update weight and 
    Bp=W(:,R) %Bias
    linehandle=plotpc(Wp,Bp,linehandle);
    lines= findobj(gcf,'Types','Line')
    change_linewidth %in shared dir Hints
    change_marker
    drawnow;
    if (pause_flag ==1)
        pause(1)
    end
    A=hardlim(W*P);% Calculate the weight of the output
    E=T-A; % Calculate the error
    clc;home;
    fprintf(' Tp= %3i %3i %3i %3i \n',Tp);
    fprintf(' Ai= %3i %3i %3i %3i \n',A);
    fprintf(' Ei= %3i %3i %3i %3i \n',Ei);
    pause(1);
end
watchoff;
fprintf('Input values for Perception example #0  \n')
for j=1:R-1
    for i=1:Q
        fprintf('  %+5.4f ',P(j,i))
    end
    fprintf(' \n')
end
fprintf('Target VAlues: \n')
for j=1:S
    for i=1:Q
        fprintf('%+5.4f ',T(j,i)) 
    end
    fprintf(' \n')
end
fprintf('Net Output Values: \n')
for j=1:S
    for i=1:Q
        fprintf( '  %+5.4f ',A(j,i))
    end
    fprintf(' \n')
end
%% Network testing section
% Now hardlims is used to classify
% any other input vector, like [0.7;1.2]:
% A plot of this new point with 
% the original training set shows
% how the network performs. First,
% plot the new point:
%p=[0.7;1.2;1]
% p=[-0.7;1;1]
p=[0.5;0.5;1];
a= hardlim(W*p); %provide binary output
plotpv(p(1:2),a);
ThePoint=findobj(gca,'type','line');
set(ThePoint,'Color','red');
hold on
plotpv(P(1:R-1,:),T);
Wp=W(:,1:R-1)
Bp=W(:,R)
plotpc(Wp,Bp);
change_linewidth
change_marker
hold off;



