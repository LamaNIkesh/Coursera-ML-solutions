function [J, grad] = costFunctionReg(theta, X, y, lambda)
%COSTFUNCTIONREG Compute cost and gradient for logistic regression with regularization
%   J = COSTFUNCTIONREG(theta, X, y, lambda) computes the cost of using
%   theta as the parameter for regularized logistic regression and the
%   gradient of the cost w.r.t. to the parameters. 

% Initialize some useful values
m = length(y); % number of training examples

% You need to return the following variables correctly 
J = 0;
grad = zeros(size(theta));

% ====================== YOUR CODE HERE ======================
% Instructions: Compute the cost of a particular choice of theta.
%               You should set J to the cost.
%               Compute the partial derivatives and set grad to the partial
%               derivatives of the cost w.r.t. each parameter in theta

h_theta = sigmoid(X * theta);
%cost calculation
J = (1 / m) * ( sum((-y.*log(h_theta) - (1 - y).*log(1 - h_theta)))) + lambda/(2*m)*(sum(theta(2:end).^2));
%calculate theta with gradient descent 
% the first value is a later added bias with all values equal to 1. 
%so the update rule for the first theta is given below
grad(1) = 1 / m * sum((h_theta - y) .* X(:, 1));
for i = 2:size(theta, 1)


    %for the next thetas, the update rule is regularised lamda term
    grad(i) = (1 / m * sum((h_theta - y) .* X(:, i))) + (lambda/m)* theta(i);
end




% =============================================================

end
