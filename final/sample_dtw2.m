%##############################################################
% This is a sample script for evaluating the classifier quality
% of your system.
%##############################################################

clear all;
clc;

% Define lists
allFiles = 'allList.txt';
trainList = 'trainCleanList.txt';
testList = 'testCleanList.txt';

tic

% Extract features
featureDict = containers.Map;
fid = fopen(allFiles);
myData = textscan(fid,'%s');
fclose(fid);
myFiles = myData{1};
distance = zeros(293,293);
for m = 58:293
    for n = (m+1):293
        [snd,fs] = audioread(myFiles{m});
        [F0,lik] = fast_mbsc_fixedWinlen_tracking(snd,fs);
        F = F0.*(lik>0.45);
        %featureDict(myFiles{i}) = mean(F0(lik>0.45));

        [snd2,fs2] = audioread(myFiles{n});
        [F02,lik2] = fast_mbsc_fixedWinlen_tracking(snd2,fs2);
        F2 = F02.*(lik2>0.45);
        %featureDict(myFiles{i}) = mean(F0(lik>0.45));
        distance(m-57,n-57) =dtw(F,F2);
        
    end
    
end

% Train the classifier
fid = fopen(trainList);
myData = textscan(fid,'%s %s %f');
fclose(fid);
fileList1 = myData{1};
fileList2 = myData{2};
labels = myData{3};
scores = zeros(length(labels),1);
for(i = 1:length(labels))
    scores(i) = -abs(featureDict(fileList1{i})-featureDict(fileList2{i}));
end
[~,threshold] = compute_eer(scores,labels);

% Test the classifier
fid = fopen(testList);
myData = textscan(fid,'%s %s %f');
fclose(fid);
fileList1 = myData{1};
fileList2 = myData{2};
labels = myData{3};
scores = zeros(length(labels),1);
for(i = 1:length(labels))
    scores(i) = -abs(featureDict(fileList1{i})-featureDict(fileList2{i}));
end
prediction = (scores>threshold);
FPR = sum(~labels & prediction)/sum(~labels);
FNR = sum(labels & ~prediction)/sum(labels);
disp(['The false positive rate is ',num2str(FPR*100),'%.'])
disp(['The false negative rate is ',num2str(FNR*100),'%.'])

toc





