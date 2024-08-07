
%% Statrt of Program
clc
clear
close all


%% Read Score Data

FolderAdress2 = "D:\MyWork\2-ParsCoders\TrainTest\score.xlsx";
scoreRslt = readtable(FolderAdress2,"Sheet","result");
scoreAdd  = readtable(FolderAdress2,"Sheet","addition");
scoreSubt = readtable(FolderAdress2,"Sheet","subtraction");
scoreY1 = readtable(FolderAdress2,"Sheet","y1");
scoreY2 = readtable(FolderAdress2,"Sheet","y2");

scRslt = scoreRslt{:,:};
scAdd  = scoreAdd{:,:};
scSubt = scoreSubt{:,:};
scY1   = scoreY1{:,:};
scY2   = scoreY2{:,:};

%% Read Main Data and Select Random for Train and Test

% Folder Adress
FolderAdress  = "D:\MyWork\2-ParsCoders\TrainTest\SplitData.xlsx";
AllData = readtable(FolderAdress,"Sheet","a1");


x1 = AllData{[2:end],[1:30]};
y1 = AllData{[2:end],[31:32]};;

x = x1';
y = y1';



responseNum1 = 1; % Equivalent to Clmn 31 in Main Data
responseNum2 = 2; % Equivalent to Clmn 32 in Main Data

[m n] = size(x);
numofTrainData = 800;
numofTestData = n - numofTrainData;
p = randperm(n,numofTrainData);

X = x(:,p);
T31 = y(responseNum1,p);
T32 = y(responseNum2,p);


jj = 1;
for ii = 1:n
    if (isempty(find(p==ii)))
        Xtst(:,jj) = x(:,ii);
        T31tst(:,jj) = y(responseNum1,ii);
        T32tst(:,jj) = y(responseNum2,ii);
        jj = jj+1;
    end
end

%% Set Up Model

% Implement k-NN Model on the Training data
mdl_31 = fitcknn(X', T31', 'NumNeighbors',1);
mdl_32 = fitcknn(X', T32', 'NumNeighbors',1);

% Predict Values on the Train data
predicted_values_31 = predict(mdl_31, X');
predicted_values_32 = predict(mdl_32, X');

% Predict Values on the Test data
predicted_values_31_tst = predict(mdl_31, Xtst');
predicted_values_32_tst = predict(mdl_32, Xtst');

% Ensure Predicted Values are Positive
%predicted_values(predicted_values <= 0) = 0 ;

%% Calculation

% Calculate Total Error for Train Data
total_error_train_31 = sum(abs(T31' - predicted_values_31)) / numel(T31);
total_error_train_32 = sum(abs(T32' - predicted_values_32)) / numel(T32);

% Calculate Total Error for Test Data
total_error_tst_31 = sum(abs(T31tst' - predicted_values_31_tst)) / numel(T31tst);
total_error_tst_32 = sum(abs(T32tst' - predicted_values_32_tst)) / numel(T32tst);

a31=[T31tst'-predicted_values_31_tst];
b31= find(a31==1);
c31= find(a31==-1);
d31 = find(a31==0);
e31 = find(a31>1);
f31 = find(a31<-1);

a32=[T32tst'-predicted_values_32_tst];
b32= find(a32==1);
c32= find(a32==-1);
d32 = find(a32==0);
e32 = find(a32>1);
f32 = find(a32<-1);


%% Rating

% Initialization
Rslt=zeros(numofTestData,1);
Addition=zeros(numofTestData,1);
Subtraction=zeros(numofTestData,1);
rateY1=zeros(numofTestData,1);
rateY2=zeros(numofTestData,1);

% Rating Result
dif_y1y2_act = T32tst - T31tst;
dif_y1y2_prd = predicted_values_32_tst - predicted_values_31_tst;

for ii=1:numofTestData
    if ( (abs(dif_y1y2_act(ii)-dif_y1y2_prd(ii))<=1) &&  (abs(T31tst(ii)-predicted_values_31_tst(ii))<=1) && (abs(T32tst(ii)-predicted_values_32_tst(ii))<=1))
        
        loc1 = find(scRslt(:,1)==predicted_values_31_tst(ii));
        loc2 = 0;
        for jj=1:length(loc1)
            if(scRslt(loc1(jj),2)==predicted_values_32_tst(ii))
                loc2 = jj;
            else
                loc2 = 0;
            end
        end
        if loc2>0
            Rslt(ii,1) = scRslt(loc1(loc2),3);
        end
    end
end

% Rating Addition
sumy1y2_act = T32tst + T31tst;
sumy1y2_prd = predicted_values_32_tst + predicted_values_31_tst;

for ii=1:numofTestData
    if ( (abs(sumy1y2_act(ii)-sumy1y2_prd(ii))<=1) )
        loc1 = find(scAdd(:,1)==sumy1y2_prd(ii));
        if loc1>0
            Addition(ii,1) = scAdd(loc1,2);
        end
    end
end



% Rating Subtraction
suby1y2_act = T32tst - T31tst;
suby1y2_prd = predicted_values_32_tst - predicted_values_31_tst;

for ii=1:numofTestData
    if ( (abs(suby1y2_act(ii)-suby1y2_prd(ii))<=1) )
        loc1 = find(scSubt(:,1)==suby1y2_prd(ii));
        if loc1>0
            Subtraction(ii,1) = scSubt(loc1,2);
        end
    end
end


% Rating y2
for ii=1:numofTestData
    if ( (abs(T31tst(ii)-predicted_values_31_tst(ii))<=1) )
        loc1 = find(scY2(:,1)==predicted_values_31_tst(ii));
        if loc1>0
            rateY2(ii,1) = scY2(loc1,2);
        end
    end
end


% Rating y1
for ii=1:numofTestData
    if ( (abs(T32tst(ii)-predicted_values_32_tst(ii))<=1) )
        loc1 = find(scY1(:,1)==predicted_values_32_tst(ii));
        if loc1>0
            rateY1(ii,1) = scY1(loc1,2);
        end
    end
end


%% Plots & Results
figure(1)
% Plot Prediction Results Clmn 31
subplot(2,2,1);
plot(1:length(T31), T31, 'bo', 1:length(predicted_values_31), predicted_values_31, 'rs');
legend('Actual', 'Predicted');
xlabel('Actual Results');
ylabel('Prediction');
title(' Train Data: Prediction vs. Actual Results ---> Column 31');

subplot(2,2,3);
plot(1:length(T31tst), T31tst, 'bo', 1:length(predicted_values_31_tst), predicted_values_31_tst, 'rs');
legend('Actual', 'Predicted');
xlabel('Actual Results');
ylabel('Prediction');
title(' Test Data: Prediction vs. Actual Results ---> Column 31');

% Plot Prediction Results Clmn 32
subplot(2,2,2);
plot(1:length(T32), T32, 'bo', 1:length(predicted_values_32), predicted_values_32, 'rs');
legend('Actual', 'Predicted');
xlabel('Actual Results');
ylabel('Prediction');
title(' Train Data: Prediction vs. Actual Results ---> Column 32');

subplot(2,2,4);
plot(1:length(T32tst), T32tst, 'bo', 1:length(predicted_values_32_tst), predicted_values_32_tst, 'rs');
legend('Actual', 'Predicted');
xlabel('Actual Results');
ylabel('Prediction');
title(' Test Data: Prediction vs. Actual Results ---> Column 32');

fprintf('************************************\n');
fprintf('Modeling Results:\n\n');
fprintf('                Clmn 31      Clmn 32\n');
fprintf('Train Error is: %.5f      %.5f \nTest Error is:  %.5f      %.5f',total_error_train_31,total_error_train_32,total_error_tst_31,total_error_tst_32);
fprintf('\n\nExact:          %.2f %%      %.2f %%  \nAcceptable:     %.2f %%      %.2f %%\nOut of Range:   %.2f %%      %.2f %%\n',100*length(d31)/length(a31),100*length(d32)/length(a32),100*(length(b31)+length(c31))/length(a31),100*(length(b32)+length(c32))/length(a32),100*(length(e31)+length(f31))/length(a31),100*(length(e32)+length(f32))/length(a32));
fprintf('************************************\n');
fprintf('Rating Results:\n\n');
fprintf('Rate Y1:          %.0f\n',sum(rateY1));
fprintf('Rate Y2:          %.0f\n',sum(rateY2));
fprintf('Rate Result:      %.0f\n',sum(Rslt));
fprintf('Rate Addition:    %.0f\n',sum(Addition));
fprintf('Rate Subtraction: %.0f\n',sum(Subtraction));
fprintf('----------------------\n');
fprintf('Rate Total:       %.0f\n',sum(rateY1)+sum(rateY2)+sum(Rslt)+ sum(Addition)+sum(Subtraction));

figure(2)
subplot(2,2,1);
confusionchart(T31',predicted_values_31);
title(' Actual Data (Case a1): confusionchart ---> Column 31');

subplot(2,2,3);
confusionchart(T31tst',predicted_values_31_tst);
title(' Test Data: confusionchart ---> Column 31');

subplot(2,2,2);
confusionchart(T32',predicted_values_32);
title(' Actual Data (Case a1): confusionchart ---> Column 32');

subplot(2,2,4);
cm = confusionchart(T32tst',predicted_values_32_tst);
title(' Test Data: confusionchart ---> Column 32');


%% Save Result
filename = 'Result_Test_Data_a1.xlsx';

Header = {'Y2 act','Y1 act','Y2 pred','Y1 pred','Rate Result','Rate Addition','Rate Subtraction','Rate Y2','Rate Y1'};
r1 = cell(numofTestData+1,9);
r1(1,:) = Header;
r1(2:end,:) = num2cell([T31tst' T32tst' predicted_values_31_tst predicted_values_32_tst Rslt Addition Subtraction rateY2 rateY1]);
writecell(r1,filename,'Sheet','Result_TestData')

