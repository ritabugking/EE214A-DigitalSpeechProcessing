clear all;
close all;

% Trains GMM for each male and female training utterances from TIMIT
% MFCCs are used as features
% Classifies new utterances as either male or female based on GMMs

addpath('VOICEBOX');
addpath('DATA');

        %% Create Multiple GMM
allFiles = 'allList.txt';
trainList = 'trainCleanList.txt';

FID = fopen(allFiles);
myData = textscan(FID,'%s');
fclose(FID);
files = myData{1};
NUM_MFCCs = 12;                 % number of MFCC coeffients to use
NUM_MIXTURES = 40;               % numbre of mixtures in GMMs
%GMModels = cell(40,1);
IDs=[];
MaleMFCCs=[];

Tw = 25;                % analysis frame duration (ms)
Ts = 10;                % analysis frame shift (ms)
alpha = 0.97;           % preemphasis coefficient
M = 20;                 % number of filterbank channels 
C = 12;                 % number of cepstral coefficients
L = 22;                 % cepstral sine lifter parameter
LF = 50;               % lower frequency limit (Hz)
HF = 500;              % upper frequency limit (Hz)
    
for i = 1:length(files)
    F = files{i};
    [speech, fs] = audioread(F);
    %[ MFCCs, FBEs, frames ] = mfcc2( speech, fs, Tw, Ts, alpha, @hamming, [LF HF], M, C+1, L ); 
    MFCCs = melcepst(speech, fs, 'M', NUM_MFCCs, 26);   
    MaleMFCCs = [MaleMFCCs; MFCCs];
end
% for i=58:293
%     file=char(files(i));
%     loc=find(file=='/')+1;
%     id = file(loc(2):loc(2)+2);
%     IDs = [IDs; string(id)];
% end
% ID_name = unique(IDs);
% j=1;
% num_model = 1;
% MaleMFCCs = [];
% while(j<=length(IDs))
%     k=j;
%     start_num = j;
%     while(k<=length(IDs)&&IDs(k)==IDs(j))
%         k=k+1;
%     end
%     k=k-1;
%     end_num = k;
%     
%     FileLength = end_num - start_num +1; 
%     
% 
%         % Get MFCCs
%     for i = start_num:start_num+FileLength-1
% 
%         F = files{i+57};
%         [speech, fs] = audioread(F);
%         MFCCs = melcepst(speech, fs, 'M', NUM_MFCCs, 26);   
%         MaleMFCCs = [MaleMFCCs; MFCCs];
%     end
% 
% 
%     
%     j = k+1;
% end

        % Fit GMM model to MFCCs
    options = statset('MaxIter', 1000);         % limit max itterations without convergence
    cInd = kmeans(MaleMFCCs, NUM_MIXTURES, 'Options', options, 'EmptyAction', 'singleton');
    gmm_model = fitgmdist(MaleMFCCs, NUM_MIXTURES, 'Options', options, 'CovType', 'diagonal', 'Start', cInd);
    %GMModels{num_model} =gmm_model;
    %num_model = num_model+1;
    
    %% Train Classifier

fid = fopen(trainList);
myData = textscan(fid,'%s %s %f');
fclose(fid);
fileList1 = myData{1};
fileList2 = myData{2};
labels = myData{3};
scores = zeros(length(labels),1);


FileLength=length(labels);

for i = 1:FileLength
    F1 = fileList1{i};
    F2 = fileList2{i};
    [speech1, fs1] = audioread(F1);
    [speech2, fs2] = audioread(F2);
    [ MFCCs1, FBEs, frames ] = ...
                    mfcc( speech1, fs1, Tw, Ts, alpha, @hamming, [LF HF], M, C+1, L );
    [ MFCCs2, FBEs, frames ] = ...
                    mfcc( speech2, fs2, Tw, Ts, alpha, @hamming, [LF HF], M, C+1, L );
    %MFCCs1 = melcepst(speech1, fs1, 'M', NUM_MFCCs, 26);    % Get MFCCs of classified data
    %MFCCs2 = melcepst(speech2, fs2, 'M', NUM_MFCCs, 26);
    
    % build a gmm for the first new speaker
    NUM_MIXTURES = 3;
    options = statset('MaxIter', 100);         % limit max itterations without convergence
    cInd = kmeans(MFCCs1, NUM_MIXTURES, 'Options', options, 'EmptyAction', 'singleton');
    gmm_model_new = fitgmdist(MFCCs1, NUM_MIXTURES, 'Options', options, 'CovType', 'diagonal', 'Start', cInd);
    
    % Calculate PDF for second speaker
%     file=char(fileList2(i));
%     loc=find(file=='/')+1;
%     id = file(loc(2):loc(2)+2);
%     skip = find(id==ID_name);
    
%     ML_All=[];
%     for j=1:length(ID_name)
%         ProbsMale = pdf(GMModels{j}, MFCCs2); 
%         % Calculate log Maximum Likelihood
%         if j~= skip
%             ML_male = sum(log(ProbsMale));
%             ML_All = [ML_All; ML_male];
%         end
%     end
    
    ProbsMale = pdf(gmm_model, MFCCs2);
    ML_ID2 = sum(log(ProbsMale));
    %pred_ID2 = find(ML_All==max(ML_All));
    % get the maximal log-likelihood of 2nd speaker from GMM-Universe
    %ML_ID2 = ML_All(pred_ID2);
    
    % Calculate PDF again
    ProbsMale = pdf(gmm_model_new, MFCCs2); 
    % get the maximal log-likelihood of 2nd speaker from GMM of the first
    % speaker
    ML_male = sum(log(ProbsMale));
        
    scores(i) = ML_male-ML_ID2;
    
%     if (pred_ID1 == pred_ID2)
%         classification{i} = 1;
%     else
%         classification{i} = 0;
%     end
end
%     for(i = 1:length(labels))
%         scores(i) = -abs(featureDict(fileList1{i})-featureDict(fileList2{i}));
%     end
%     prediction = (scores>threshold);
[~,threshold] = compute_eer(scores,labels);



% testing
testList = 'testCleanList.txt';
fid = fopen(testList);
myData = textscan(fid,'%s %s %f');
fclose(fid);
fileList1 = myData{1};
fileList2 = myData{2};
labels = myData{3};
scores = zeros(length(labels),1);

FileLength=length(labels);
classification = cell(FileLength, 1);   % used to hold classifications

for i = 1:FileLength
    F1 = fileList1{i};
    F2 = fileList2{i};
    [speech1, fs1] = audioread(F1);
    [speech2, fs2] = audioread(F2);
    [ MFCCs1, FBEs, frames ] = ...
                    mfcc( speech1, fs1, Tw, Ts, alpha, @hamming, [LF HF], M, C+1, L );
    [ MFCCs2, FBEs, frames ] = ...
                    mfcc( speech2, fs2, Tw, Ts, alpha, @hamming, [LF HF], M, C+1, L );
    %MFCCs1 = melcepst(speech1, fs1, 'M', NUM_MFCCs, 26);    % Get MFCCs of classified data
    %MFCCs2 = melcepst(speech2, fs2, 'M', NUM_MFCCs, 26);
    
    % build a gmm for the first new speaker
    NUM_MIXTURES = 3;
    options = statset('MaxIter', 100);         % limit max itterations without convergence
    cInd = kmeans(MFCCs1, NUM_MIXTURES, 'Options', options, 'EmptyAction', 'singleton');
    gmm_model_new = fitgmdist(MFCCs1, NUM_MIXTURES, 'Options', options, 'CovType', 'diagonal', 'Start', cInd);
    
    % Calculate PDF for second speaker
    %ML_All=[];
%     for j=1:length(ID_name)
%         ProbsMale = pdf(GMModels{j}, MFCCs2); 
%         % Calculate log Maximum Likelihood
%         ML_male = sum(log(ProbsMale));
%         ML_All = [ML_All; ML_male];
%         
%     end
    ProbsMale = pdf(gmm_model, MFCCs2);
    ML_ID2 = sum(log(ProbsMale));
    %pred_ID2 = find(ML_All==max(ML_All));
    % get the maximal log-likelihood of 2nd speaker from GMM-Universe
    %ML_ID2 = ML_All(pred_ID2);
    
    % Calculate PDF again
    ProbsMale = pdf(gmm_model_new, MFCCs2); 
    % get the maximal log-likelihood of 2nd speaker from GMM of the first
    % speaker
    ML_male = sum(log(ProbsMale));
        
    scores(i) = ML_male-ML_ID2;
end


prediction = (scores>threshold);
FPR = sum(~labels & prediction)/sum(~labels);
FNR = sum(labels & ~prediction)/sum(labels);
disp(['The false positive rate is ',num2str(FPR*100),'%.'])
disp(['The false negative rate is ',num2str(FNR*100),'%.'])



%     c = cell2mat(classification);
%     c=int8(c);
%     FPR = sum(~labels & c)/sum(~labels);
%     FNR = sum(labels & ~c)/sum(labels);
%     disp(['The false positive rate is ',num2str(FPR*100),'%.'])
%     disp(['The false negative rate is ',num2str(FNR*100),'%.'])


%     if (classification{i} ~= labels{i})
%         IncorrectCount = IncorrectCount + 1;
%         if (Labels{i} == 'F')
%             IncorrectFemale = IncorrectFemale + 1;
%         end
%         if (Labels{i} == 'M')
%             IncorrectMale = IncorrectMale + 1;
%         end
%         
%     else
%         if (Labels{i} == 'F')
%             CorrectFemale = CorrectFemale + 1;
%         end
%         if (Labels{i} == 'M')
%             CorrectMale = CorrectMale + 1;
%         end
%     end
% end
% 
% Precentage = ((FileLength - IncorrectCount)/FileLength)*100;        % final classification precentage
% 
% A1 = [CorrectMale, CorrectFemale, IncorrectMale, IncorrectFemale, Precentage];
% 
%     % Print Results to a files
% fileID = fopen('Results.txt','w');
% fprintf(fileID, 'Number of Correctly Identified Male Speakers %8.3f     \n', A1(1));
% fprintf(fileID, 'Number of Correctly Identified Female Speakers %8.3f   \n', A1(2));
% fprintf(fileID, 'Number of Incorrectly Identified Male Speakers %8.3f   \n', A1(3));
% fprintf(fileID, 'Number of Incorrectly Identified Female Speakers %8.3f \n', A1(4));
% fprintf(fileID, 'Total Precentage of Correct Classification %8.3f       \n', A1(5));