% Convert "#"= 1, "." = "-1"
% loop for training, loop for testing( add a bunch of noisy data inside)


%% 1. Tranning by Hebb
clear all;clc;clf;
pauseflag=1;
pausetime=0.5;
minw=0; %set up min and max weight
maxw=20;
% P is an input matrix
P=[1  1  1  1  1  1 -1 -1 -1 -1  1  1  1  1 -1  1 -1 -1 -1 -1  1  1  1  1  1 1;%E+bias
   1  1  1  1 -1  1 -1 -1 -1 -1  1  1  1 -1 -1  1 -1 -1 -1 -1  1 -1 -1 -1 -1 1]; %F+bias
   
fprintf('Input Matrix P \n');
fprintf('%2i %2i %2i %2i %2i %2i %2i %2i %2i %2i %2i %2i %2i %2i %2i %2i %2i %2i %2i %2i %2i %2i %2i %2i %2i %2i \n',P);
%The first and second line are inputs
% The third line is bias
% The first column is X1,second column is X2
P=P'; % transpose P
T=[1 -1]; %Bipolar output
fprintf('Output Matrix T \n');
fprintf('%2i %2i \n',T)
[R,Q]= size(P); [S,Q] =size(T);
fprintf('R=%2i Q=%2i S=%2i \n',R,Q,S)
W0=zeros(S,R); % Initialize all weight equal to 0
B0= ones(S,1);
W=W0;
B=B0;
max_epoch=20;% the number of time you want to show the network your input
lp.lr=1;%define learing rate
lp.dr=0;%and decay rate
disp_freq=1;

for epoch=1:max_epoch
    fprintf('epoch %i\n',epoch)
    for q=1:Q %Q is the number of column of the input matrix
      
        A=T(:,q);  %choose X1, X2 inputs

        dW=learnhd(W,P(:,q),[],[],A,[],[],[],[],[],lp,[]); %calculating the weight change

        W=W+dW; %calculating the weight when matlab is learning
%         fprintf('input= %2i, W= %2i \n',q,W); %Print the number of input and the weight has been added
        if pauseflag==1
            pause(pausetime);
            figure(1)
        end
    end
    if rem(epoch,disp_freq) == 0 %display how the matlab learn by graph
        pause(pausetime);
        hintonw(W,maxw,minw)
        title('Weights Function (i,j)')
    end
end

% Plotting the final results
hintonw(W,maxw,minw) %graph the weight of the matrix
% Summarize results
disp('The Final Weights Function');
W
disp('');
P
disp('');
disp('The neuron responds')
A=hardlims(W*P) % return the results in bipolar format

fprintf('Input values P \n')
for j=1:R-1
    for i=1:Q
        fprintf('%+5.4f ',P(j,i))
    end
    fprintf(' \n')
end
fprintf('Target Values T: \n')
for j=1:S
    for i=1:Q
        fprintf('%+5.4f ',T(j,i)) 
    end
    fprintf(' \n')
end
fprintf('Net Output Values: \n')
for j=1:S
    for i=1:Q
        fprintf( '%+5.4f ',A(j,i))
    end
    fprintf(' \n')
end

%% Testing
E=[+1 +1 +1 +1 +1 ;...
    +1 -1 -1 -1 -1 ;...
    1  1  1  1 -1 ;...
    1 -1 -1 -1 -1 ;...
    1 1 1 1 1  ];
figure(3)
subplot(2,2,1)

%display E
hintonw(E)
% Ev E as a vector
Ev= reshape(E,25,1);

%flip pixel
nfp=5; % how many pixel messing up
Evt=Ev;
r_ind=randi(25,nfp,1);
Evt(r_ind,1)=Evt(r_ind,1)*-1;
E_n=reshape(Evt,5,5);
% figure(4)
subplot(2,2,3)

hintonw(E_n)
Ev(26,1)=1; %Add a bias in to the Ev (testing input vector)

%testing
A_test=hardlims(W*Ev);
if  A_test==1
    fprintf('This is an E');
    Emap=[1 1 1 1 1
       1 -1 -1 -1 -1
       1  1  1  1 -1
       1 -1 -1 -1 -1
       1 1 1 1 1 ]; % Input for E
%        figure(5)
       subplot(2,2,4)

       hintonw(Emap) 
elseif A_test==-1
    fprintf('This is an F');
Fmap=[1  1  1  1 -1
      1 -1 -1 -1 -1 
      1  1  1 -1 -1
      1  -1 -1 -1 -1
      1  -1 -1 -1 -1];
% figure(6)
subplot(2,2,4)

hintonw(Fmap)


else
    fprintf('This is not neither E nor F\n')
end




%% Question 3
% code to turn off pixels: Emap/turn off pixels/ still realized/ turn off/ 
E=[+1 +1 +1 +1 +1 ;...
    +1 -1 -1 -1 -1 ;...
    1  1  1  1 -1 ;...
    1 -1 -1 -1 -1 ;...
    1 1 1 1 1  ];

figure(7)
%display E
subplot(2,2,1)
hintonw(E)
% Ev E as a vector
Ev= reshape(E,25,1);

%turn off pixels randomly
k=1
    nfp=1; % how many pixel turn off
    Evt=Ev;
    r_ind=randi(25,nfp,1);
    Evt(r_ind,1)=0;
    E_n=reshape(Evt,5,5);
%     figure(8)
    subplot(2,2,3)
    hintonw(E_n)
    E_n=reshape(E_n,25,1);
    E_n(26,1)=1;% add bias
    A_test_off=hardlims(W*E_n);
   fprintf('A_test_off = %2i\n',A_test_off);

if A_test_off==1
    while A_test_off==1
    E_n=zeros(26,1);
    nfp=nfp+1;
    r_ind=randi(25,nfp,1);
    Evt(r_ind,1)=0;
    E_n=reshape(Evt,5,5);
%     figure(8)
    subplot(2,2,4)
    hintonw(E_n)
    E_n=reshape(E_n,25,1);
    E_n(26,1)=1;
    A_test_off=hardlims(W*E_n);
        for j=1:2
            T(k,j)=nfp;
        end
    end
elseif A_test_off==-1
    while A_test_off==-1 
    E_n=zeros(26,1);
    nfp=nfp+1;
    r_ind=randi(25,nfp,1);
    Evt(r_ind,1)=0;
    E_n=reshape(Evt,5,5);
%     figure(8)
    subplot(2,2,4)
    hintonw(E_n)
    E_n=reshape(E_n,25,1);
    E_n(26,1)=1;
    A_test_off=hardlims(W*E_n);
    
        for j=1:2
            T(k,j)=nfp;
        end
    end
end
fprintf("Limits of pixels: %2i \n",nfp)



%% Question 4
%(20,21)

E=[+1 +1 +1 +1 +1;...
    +1 -1 -1 -1 -1;...
    1  1  1  1 -1;...
    1 -1 -1 -1 -1;...
    1 1 1 1 1 ];
% Ev E as a vector
Ev= reshape(E,25,1);
figure(20)
subplot(2,2,1)
hintonw(E)

%flip pixel
nfp=1;% how many pixel messing up
k=1; %number of times
Evt=Ev;
r_ind=randi(25,nfp,1);
Evt(r_ind,1)=Evt(r_ind,1)*-1;
E_n=reshape(Evt,5,5);
% figure(21)
subplot(2,2,3)
hintonw(E_n)
E_n=reshape(E_n,25,1);
E_n(26,1)=1;
number_mistakes=0;
A_test_flip=hardlims(W*E_n);
fprintf("The value of A is %2i \n ", A_test_flip);

% for n=1:100
if A_test_flip==1
    while A_test_flip==1
    E_n=zeros(26,1);
    nfp=nfp+1;
    r_ind=randi(25,nfp,1);
    Evt(r_ind,1)=Evt(r_ind,1)*-1;
    E_n=reshape(Evt,5,5);
%     figure(22)
    subplot(2,2,4)
    hintonw(E_n)
    E_n=reshape(E_n,25,1);
    E_n(26,1)=1;
    A_test_flip=hardlims(W*E_n);
            for j=1:2
            T(k,j)=nfp;
            end
    end
elseif A_test_flip==-1
    while A_test_flip==-1
    E_n=zeros(26,1);
    nfp=nfp+1;
    r_ind=randi(25,nfp,1);
    Evt(r_ind,1)=Evt(r_ind,1)*-1;
    E_n=reshape(Evt,5,5);
%     figure(23)
    subplot(2,2,4)
    hintonw(E_n)
    E_n=reshape(E_n,25,1);
    E_n(26,1)=1;
    A_test_flip=hardlims(W*E_n);
        for j=1:2
            T(k,j)=nfp;
        end
    end
end

% end

% fprintf("Number of times: %2i \n",k)

fprintf("Number of mistakes: %2i \n",nfp)



