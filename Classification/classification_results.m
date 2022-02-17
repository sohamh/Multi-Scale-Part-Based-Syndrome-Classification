%% triplet loss- fullface
clear all

base="../Multi-Scale-Part-Based-Syndrome-Classification/Embeddings/"; %Path to the folder containing embeddings

n_fold=5; 
n_class=2;

dimensions={'4','14','24','35','47','57'};

for st=1:length(dimensions)
    name=base+'GE/modular_'+dimensions{st}+'_dim.mat';      
    load(name);

    name=base+"PCA/modular_"+string(dimensions(st))+"_dim.mat";
    PCA=load(name);
    for fold=1:n_fold
        for synd=1:14
            train_preds=preds_train{fold,1};

            grps_train=PCA.sum_data{fold,1}.grps_train;
            grps=ones(1,size(train_preds,1));
            grps(grps_train'==synd)=0; % to obtain binary labels

            group_count=[length(grps)-sum(grps),sum(grps)];
            MdlLinear = fitcdiscr(train_preds,grps+1,'Weights',(1./group_count(squeeze(grps+1))));%last argument is only for weighted LDA
    
            test_preds=preds_test{fold,1};
            grps_test=PCA.sum_data{fold,1}.grps_test;
            grps=ones(1,size(test_preds,1));
            grps(grps_test'==synd)=0;% to obtain binary labels
    
            [label,score,cost]=predict(MdlLinear,test_preds);
      
            [T{st}.acc(fold,synd), T{st}.balanced_acc(fold,synd), T{st}.sensitivity(fold,synd), T{st}.specificity(fold,synd), T{st}.F1(fold,synd), T{st}.MCC(fold,synd)] = confmat(label,grps);
    
        end
    end
    base2="../Multi-Scale-Part-Based-Syndrome-Classification/";% path to your folder
    writetable(struct2table(T{st}),base2+'Results/tripletloss_fullface_results.xlsx','Sheet',st)
end
%% triplet loss- partbased

n_fold=5;
n_class=2;

dimensions={'4','14','24','35','47','57'};

for st=1:length(dimensions)
    name=base+"GE/modular_"+dimensions{st}+"_dim.mat";
    load(name);
    name=base+"PCA/modular_"+string(dimensions(st))+"_dim.mat";
    PCA=load(name);

    for fold=1:n_fold
        for synd=1:14
            
            train_preds=[preds_train{fold,1},preds_train{fold,2},preds_train{fold,3},...
            preds_train{fold,4},preds_train{fold,5},preds_train{fold,6},preds_train{fold,7}];
        
            grps_train=PCA.sum_data{fold,1}.grps_train;
            grps=ones(1,size(train_preds,1));
            grps(grps_train'==synd)=0;
    
            group_count=[length(grps)-sum(grps),sum(grps)];
            MdlLinear = fitcdiscr(train_preds,grps+1,'Weights',(1./group_count(squeeze(grps+1))));
            %this added for weighted LDA
    
           
            test_preds=[preds_test{fold,1},preds_test{fold,2},preds_test{fold,3},...
            preds_test{fold,4},preds_test{fold,5},preds_test{fold,6},preds_test{fold,7}];
            grps_test=PCA.sum_data{fold,1}.grps_test;
            grps=ones(1,size(test_preds,1));
            grps(grps_test'==synd)=0;
    
            [label,score,cost]=predict(MdlLinear,test_preds);
    
            [T{st}.acc(fold,synd), T{st}.balanced_acc(fold,synd), T{st}.sensitivity(fold,synd), T{st}.specificity(fold,synd), T{st}.F1(fold,synd), T{st}.MCC(fold,synd)] = confmat(label,grps);
            
        end
    end
    base2="../Multi-Scale-Part-Based-Syndrome-Classification/";% path to your folder
    writetable(struct2table(T{st}),base2+'Results/tripletloss_partbased_results.xlsx','Sheet',st)
end

%% baseline- partbased- one vs rest- AUC
dimensions=[4,14,24,35,47,57];
n_fold=5;
n_class=2;

for st=1:length(dimensions)

    dim=dimensions(st);

    name=base+"PCA/modular_"+string(dimensions(st))+"_dim.mat";
    load(name);
    
    for fold=1:n_fold
        for synd=1:14 
            train_preds=[];
            for lim=1:7 %7 for part-based and 1 for fullface
                train_preds=[train_preds,sum_data{fold,lim}.predicted_train(:,1:dim)];
            end
            grps_train=sum_data{fold,1}.grps_train;
            grps=ones(1,size(train_preds,1));
            grps(grps_train'==synd)=0;
            group_count=[length(grps)-sum(grps),sum(grps)];
            MdlLinear = fitcdiscr(train_preds,grps+1,'Weights',(1./group_count(squeeze(grps+1))));
            %this added for weighted LDA

            test_preds=[];
            for lim=1:7 %7 for part-based and 1 for fullface
                test_preds=[test_preds,sum_data{fold,lim}.predicted_test(:,1:dim)];
            end
            grps_test=sum_data{fold,1}.grps_test;
            grps=ones(1,size(test_preds,1));
            grps(grps_test'==synd)=0;
    
            [label,score,cost]=predict(MdlLinear,test_preds);
    
            [T{st}.acc(fold,synd), T{st}.balanced_acc(fold,synd), T{st}.sensitivity(fold,synd), T{st}.specificity(fold,synd), T{st}.F1(fold,synd), T{st}.MCC(fold,synd)] = confmat(label,grps);
        end
    end
    base2="../Multi-Scale-Part-Based-Syndrome-Classification/";% path to your folder
    writetable(struct2table(T{st}),base2+'Results/baseline_partbased_results.xlsx','Sheet',st)
end


%% mesures from confusion matric in binary classification
function [acc, balanced_acc, sensitivity, specificity, F1, MCC] = confmat(label,truth) 
    label=label'-1;

    P= sum(truth==0); % zero is positive
    N=sum(truth==1);
    
    TP=sum(truth==0 & label==0);
    TN=sum(truth==1 & label==1);
    FP=sum(truth==1 & label==0);
    FN=sum(truth==0 & label==1);
    
    sensitivity=TP/P;%TPR
    specificity=TN/N;%TNR
    
    acc=(TP+TN)/(P+N);
    balanced_acc= (sensitivity+specificity)/2;
    F1=(2*TP)/((2*TP)+FP+FN);
    MCC=(TP*TN)-(FP*FN)/sqrt((TP+FP)*(TP+FN)*(TN+FP)*(TN+FN));
end