function p = predict(theta, X)
%PREDICT Predict whether the label is 0 or 1 using learned logistic 
%regression parameters theta
%   p = PREDICT(theta, X) computes the predictions for X using a 
%   threshold at 0.5 (i.e., if sigmoid(theta'*x) >= 0.5, predict 1)

m = size(X, 1); % Number of training examples

% You need to return the following variables correctly
p = zeros(m, 1);

% ====================== YOUR CODE HERE ======================
% Instructions: Complete the following code to make predictions using
%               your learned logistic regression parameters. 
%               You should set p to a vector of 0's and 1's
%

h_theta = sigmoid(X * theta);
%h_theta
%thresholding at 0.5, if the h_theta values are bigger than 0.5 then assign
%1 or else 0.
for i = 1:length(h_theta)
   
    if h_theta(i) >= 0.5
        p(i) = 1;
    else
        p(i) = 0;
    end
end



% =========================================================================


end
