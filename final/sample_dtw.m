% clear all;
% clc;
% 
% % Define lists
allFiles = 'allList.txt';
trainList = 'trainCleanList.txt';
testList = 'testCleanList.txt';
NUM_MFCCs = 14; 
% 
% tic

% % Extract features
% featureDict = containers.Map;
fid = fopen(allFiles);
%fid = fopen(trainList);
myData = textscan(fid,'%s');
fclose(fid);
myFiles = myData{1};
allF0 = [];
allMfcc = [];
distance = zeros(293,293);
for m = 58:293
    for n = (m+1):293
%         [snd,fs] = audioread(myFiles{m});
%         [snd2,fs2] = audioread(myFiles{n});
%         snd = deNoise(snd);  % detrend and normolize data
%         snd2 = deNoise(snd2);
%         mfcc1 = mfcctest(snd)';
%         mfcc2 = mfcctest(snd2)';
        [speech1, fs1] = audioread(myFiles{m});
        [speech2, fs2] = audioread(myFiles{n});
    %     speech1 = deNoise(speech1);  % detrend and normolize data
    %     speech2 = deNoise(speech2);
         MFCCs1 = melcepst(speech1, fs1, 'M', NUM_MFCCs, 26);    % Get MFCCs of classified data
         MFCCs2 = melcepst(speech2, fs2, 'M', NUM_MFCCs, 26);

    %     mfcc1 = mfcctest(snd)';
    %     mfcc2 = mfcctest(snd2)';
        s=MFCCs1;
        t=MFCCs2;
%         s=mfcc1;
%         t=mfcc2;
        w=-Inf;
        ns=size(s,2);
        nt=size(t,2);
        if size(s,1)~=size(t,1)
            error('Error in dtw(): the dimensions of the two input signals do not match.');
        end

        %% initialization
        D=zeros(ns+2,nt+2)+Inf; % cache matrix
        D(1,1)=0;
        D(2,2)=0;

        % oost: distance matrix 
        oost = zeros(ns+1,nt+1)+Inf;
        for i=1:ns
            for j=1:nt
                %oost(i+1,j+1)=norm(s(:,i)-t(:,j)); % Euclidean distance
                oost(i+1,j+1) = acos(dot(s(:,i),t(:,j))/(norm(s(:,i),2)*norm(t(:,j),2)))/pi; %cosine distance
            end
        end

        % dynamic time warping
        %% begin dynamic programming
        % find the minimum distance between two matrix s & t
        % the start point should be aligned, but the end point doesn't need to be
        % D: Table of accumulative distance using dynamic programming
        for i=1:ns
            for j=1:nt
                D(i+2,j+2)=oost(i+1,j+1)+min([D(i,j+1)+oost(i,j+1), D(i+1,j)+oost(i+1,j), D(i+1,j+1)]);
            end
        end
%         d=max(D(:,nt+2));
%         d_len=nt+2;
% 
%         while(max(D(:,d_len))==-Inf)
%             d_len=d_len-1;
%             d=max(D(:,d_len));
%         end
        
        % get the smallest distance between pair sound file
        distance(m-57,n-57) = D(size(D,1), size(D,2))/(size(D,1)+size(D,2));
%     [F0,lik] = fast_mbsc_fixedWinlen_tracking(snd,fs);
%     F0(lik<0.45)=0;
%     tmp = [F0', zeros(1,600-length(F0))];
%     allF0 = [allF0; tmp];
%     
%     
%     featureDict(myFiles{i}) = mean(F0(lik>0.45));
%     if(mod(i,10)==0)
%         disp(['Completed ',num2str(i),' of ',num2str(length(myFiles)),' files.']);
%     end
    end
end


dd=[];
for m = 1:236
for n = m+1:236
temp=distance(m,n);
dd=[dd;temp];
end
end






% % Train the classifier
% fid = fopen(trainList);
% myData = textscan(fid,'%s %s %f');
% fclose(fid);
% fileList1 = myData{1};
% fileList2 = myData{2};
% labels = myData{3};
% % scores = zeros(length(labels),1);
% % for(i = 1:length(labels))
% %     %scores(i) = -abs(featureDict(fileList1{i})-featureDict(fileList2{i}));
% %     scores(i) = -dd(i);
% % end
% % [~,threshold] = compute_eer(scores,labels);
% % 
% % threshold = -threshold;
% dd=[];
% for m = 1:236
% for n = m+1:236
% temp=distance(m,n);
% dd=[dd;temp];
% end
% end
% SVMModel = fitcsvm(dd, labels, 'Standardize', true, 'KernelFunction', 'RBF','KernelScale','auto');
% label = predict(SVMModel,dd);
% sum(label==labels); % ans = 27148
% sum(label==labels)/length(labels); % ans = 0.9790



% Test the classifier
fid = fopen(testList);
myData = textscan(fid,'%s %s %f');
fclose(fid);
fileList1 = myData{1};
fileList2 = myData{2};
labels = myData{3};
FileLength=length(labels);
dd=[];
for i = 1:FileLength
    F1 = fileList1{i};
    F2 = fileList2{i};
    [speech1, fs1] = audioread(F1);
    [speech2, fs2] = audioread(F2);
%     speech1 = deNoise(speech1);  % detrend and normolize data
%     speech2 = deNoise(speech2);
     MFCCs1 = melcepst(speech1, fs1, 'M', NUM_MFCCs, 26);    % Get MFCCs of classified data
     MFCCs2 = melcepst(speech2, fs2, 'M', NUM_MFCCs, 26);
    
%     mfcc1 = mfcctest(snd)';
%     mfcc2 = mfcctest(snd2)';
    s=MFCCs1;
    t=MFCCs2;
    w=-Inf;
    ns=size(s,2);
    nt=size(t,2);
    if size(s,1)~=size(t,1)
        error('Error in dtw(): the dimensions of the two input signals do not match.');
    end

    %% initialization
    D=zeros(ns+2,nt+2)+Inf; % cache matrix
    D(1,1)=0;
    D(2,2)=0;

    % oost: distance matrix 
    oost = zeros(ns+1,nt+1)+Inf;
    for i=1:ns
        for j=1:nt
            %oost(i+1,j+1)=norm(s(:,i)-t(:,j)); % Euclidean distance
            oost(i+1,j+1) = acos(dot(s(:,i),t(:,j))/(norm(s(:,i),2)*norm(t(:,j),2)))/pi; %cosine distance
        end
    end

    % dynamic time warping
    %% begin dynamic programming
    % find the minimum distance between two matrix s & t
    % the start point should be aligned, but the end point doesn't need to be
    % D: Table of accumulative distance using dynamic programming
    for i=1:ns
        for j=1:nt
            D(i+2,j+2)=oost(i+1,j+1)+min([D(i,j+1)+oost(i,j+1), D(i+1,j)+oost(i+1,j), D(i+1,j+1)]);
        end
    end
%         d=max(D(:,nt+2));
%         d_len=nt+2;
% 
%         while(max(D(:,d_len))==-Inf)
%             d_len=d_len-1;
%             d=max(D(:,d_len));
%         end

    % get the smallest distance between pair sound file
    distance = D(size(D,1), size(D,2))/(size(D,1)+size(D,2));
    dd=[dd;distance];
end

prediction = predict(SVMModel,dd);
sum(prediction==labels) % ans = 1461
sum(prediction==labels)/length(labels) % ans = 0.9154
% scores = zeros(length(labels),1);
% for(i = 1:length(labels))
%     scores(i) = -abs(featureDict(fileList1{i})-featureDict(fileList2{i}));
% end
% prediction = (scores>threshold);
FPR = sum(~labels & prediction)/sum(~labels);
FNR = sum(labels & ~prediction)/sum(labels);
disp(['The false positive rate is ',num2str(FPR*100),'%.'])
disp(['The false negative rate is ',num2str(FNR*100),'%.'])

% toc





