function [results] = tuningSVM(X, y, Xval, yval, Cvec , sigmavec )
%tuningSVM tuning the SVM parameters
%   tuningSVM(X, y, Xval, yval, Cvec , sigmavec )tuning the SVM parameters

results = 0;

[CGrid sigmaGrid] = meshgrid(Cvec,sigmavec);
CGrid = CGrid(:);
sigmaGrid = sigmaGrid(:);
nmodels = size(CGrid)(1);

error = zeros(nmodels,1);

for i=1:nmodels
  model= svmTrain(X, y, CGrid(i), @(x1, x2) gaussianKernel(x1, x2, sigmaGrid(i))); 
  error(i) = mean(double(svmPredict(model,Xval)~= yval));
end

results = [CGrid,sigmaGrid,error];

end

%load('ex6data3.mat');
%Cvec = [0.01; 0.03; 0.1; 0.3; 1; 3; 10; 30];            
%sigmavec = [0.01; 0.03; 0.1; 0.3; 1; 3; 10; 30];        
%results = tuningSVM(X, y, Xval, yval, Cvec , sigmavec )    
%[errmin errid] = min(results(:,3));
%results(errid,:)                                         
