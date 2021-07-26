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
% Values for C and sigma to test the model on
C_search =  [0.01;0.03;0.1;0.3;1;3;10;30]; % column vector
sigma_search =  [0.01;0.03;0.1;0.3;1;3;10;30];

prediction_error_matrix = zeros(length(C_search), length(sigma_search)); % 8x8 matrix
para_combination_list = zeros(length(C_search) * length(sigma_search), 3); % 8x8 matrix which corresponds to C and sigma combination
combination_count = 1;
for i=1:length(C_search)
    for j=1:length(sigma_search)
        C_val = C_search(i);
        sigma_val = sigma_search(j);
        
        model = svmTrain(X, y, C_val, @(x1, x2)gaussianKernel(x1, x2, sigma_val));
        predictions = svmPredict(model, Xval);
        prediction_error = mean(double(predictions ~= yval));
        %prediction_error_matrix(i,j) = prediction_error; %write into the matrix
        para_combination_list(combination_count,:,:) = [C_val,sigma_val,prediction_error]; %write into a a list of three cols: col1 = C, col2= sigma, col3 = error
        combination_count = combination_count+1;
    end
end
sorted = sortrows(para_combination_list, 3);% col 3 contains the error
selected_para = sorted(1,:,:); %first row has the best combination
C = selected_para(1);
sigma = selected_para(2); 
[C; sigma];
% =========================================================================

end
