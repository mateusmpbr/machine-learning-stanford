function [C, sigma] = dataset3Params(X, y, Xval, yval)
%DATASET3PARAMS returns your choice of C and sigma for Part 3 of the exercise
%where you select the optimal (C, sigma) learning parameters to use for SVM
%with RBF kernel
%   [C, sigma] = DATASET3PARAMS(X, y, Xval, yval) returns your choice of C and 
%   sigma. You should complete this function to return the optimal C and 
%   sigma based on a cross-validation set.
%

% You need to return the following variables correctly.
C = 1;
sigma = 0.3;

% ====================== YOUR CODE HERE ======================
% Instructions: Fill in this function to return the optimal C and sigma
%               learning parameters found using the cross validation set.
%               You can use svmPredict to predict the labels on the cross
%               validation set. For example, 
%                   predictions = svmPredict(model, Xval);
%               will return the predictions on the cross validation set.
%
%  Note: You can compute the prediction error using 
%        mean(double(predictions ~= yval))
%


sigmaSet = [0.01, 0.03, 0.1, 0.3, 1, 3, 10, 30];
err_results = [];

for try_C = sigmaSet
    for try_sigma = sigmaSet
        model = svmTrain(X, y, try_C, @(x1, x2) gaussianKernel(x1, x2, try_sigma));
        predictions = svmPredict(model, Xval);
        err = mean(double(predictions ~= yval));
        err_results = [err_results; try_C, try_sigma, err];
    endfor
endfor

[minval, idx] = min(err_results(:, 3));
C = err_results(idx, 1);
sigma = err_results(idx, 2);

% =========================================================================

end
