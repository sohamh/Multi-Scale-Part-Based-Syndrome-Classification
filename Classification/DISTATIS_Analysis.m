% in this script the projections of the seven segments onto the first and
% second eigenvectors of the similarity matrix is implemented
%load in order
clear all

dimensions={'4','14','24','35','47','57'};

base="/usr/local/micapollo01/MIC/DATA/STAFF/smahdi0/tmp/Syndrome_Analysis/paper_material/Multi-Scale-Part-Based-Syndrome-Classification/Embeddings/";

fold=1;
n_segments=7;
coords = cell(1,n_segments);
for st=1:n_segments
    name=base+"GE/modular_"+string(dimensions(2))+"_dim.mat";
    load(name);
    coords{st} = preds_test{fold,st};
end
%% compute distance matrices
DMATS = cellfun(@(x) squareform(pdist(x,'Euclidean')),coords,'UniformOutput',false);
DMATS = cat(3,DMATS{:});
%% fit distatis
DS = DISTATIS();
DS.DistanceMatrices = DMATS;
DS.fit();

%% plot similarity among spaces
x = DS.StudySimilarityScores(:,1);
y = DS.StudySimilarityScores(:,2);
labels={"1","2","3","4","5","6","7"};
scatter(x,y);
hold on 
text(x-0.02,y+0.03,labels,'FontSize',44);


