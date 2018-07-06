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
%fid = fopen(trainList);
myData = textscan(fid,'%s');
fclose(fid);
myFiles = myData{1};
allF0 = []
for(i = 1:293)
%for(i = 1:length(myFiles))
    [snd,fs] = audioread(myFiles{i});
    [F0,lik] = fast_mbsc_fixedWinlen_tracking(snd,fs);
    tmp = [F0', zeros(1,600-length(F0))];
    allF0 = [allF0; tmp];
    
    featureDict(myFiles{i}) = mean(F0(lik>0.45));
    if(mod(i,10)==0)
        disp(['Completed ',num2str(i),' of ',num2str(length(myFiles)),' files.']);
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





