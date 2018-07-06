%##############################################################
% This is a sample script for evaluating the classifier quality
% of your system.
%##############################################################

clear all;
clc;

% Define lists
allFiles = 'allList.txt';
%trainList = 'trainCleanList.txt';
trainList = 'trainMultiList.txt';
%testList = 'testCleanList.txt';
testList = 'testBabbleList.txt';

tic

% Extract features
featureDict = containers.Map;
NUM_MFCCs = 12; 
fid = fopen(allFiles);
myData = textscan(fid,'%s');
fclose(fid);
myFiles = myData{1};
mfccMeans = zeros(length(myFiles),12);

    Tw = 25;                % analysis frame duration (ms)
    Ts = 10;                % analysis frame shift (ms)
    alpha = 0.97;           % preemphasis coefficient
    M = 20;                 % number of filterbank channels 
    C = 12;                 % number of cepstral coefficients
    L = 22;                 % cepstral sine lifter parameter
    LF = 50;               % lower frequency limit (Hz)
    HF = 500;              % upper frequency limit (Hz)
    %wav_file = 'sp10.wav';  % input audio filename


    % Read speech samples, sampling rate and precision from file
    %[ speech, fs] = audioread( wav_file );


    % Feature extraction (feature vectors as columns)
    %[ MFCCs, FBEs, frames ] = ...
    %                mfcc( speech, fs, Tw, Ts, alpha, @hamming, [LF HF], M, C+1, L );

for(i = 1:length(myFiles))
    [snd,fs] = audioread(myFiles{i});
     %snd = deNoise(snd);
     %mfccs = melcepst(snd, fs, 'M', NUM_MFCCs, 26, 100, 80);
     % Feature extraction (feature vectors as columns)
    %[ MFCCs, FBEs, frames ] = mfcc( snd, fs, Tw, Ts, alpha, @hamming, [LF HF], M, C+1, L );
    [ MFCCs, FBEs, frames ] = ...
                    mfcc( snd, fs, Tw, Ts, alpha, @hamming, [LF HF], M, C+1, L );
    %MFCCs = MFCCs';

    %mfccMeans(i,:) = mean(mfccs);
    %[F0,lik] = fast_mbsc_fixedWinlen_tracking(snd,fs);
    %feature=[F0,MFCCs];
    %feature=[MFCCs];
    featureDict(myFiles{i}) = mean(MFCCs');
    %featureDict(myFiles{i}) = mean(MFCCs(lik>0.45));
    %if(mod(i,10)==0)
    %    disp(['Completed ',num2str(i),' of ',num2str(length(myFiles)),' files.']);
    %end
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
    scores(i) = -sum(abs(featureDict(fileList1{i})-featureDict(fileList2{i})).^2); % L1 distance
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
    scores(i) = -sum(abs(featureDict(fileList1{i})-featureDict(fileList2{i})).^2);
end
prediction = (scores>threshold);
FPR = sum(~labels & prediction)/sum(~labels);
FNR = sum(labels & ~prediction)/sum(labels);
disp(['The false positive rate is ',num2str(FPR*100),'%.'])
disp(['The false negative rate is ',num2str(FNR*100),'%.'])

toc





