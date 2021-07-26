function [J, grad] = cofiCostFunc(params, Y, R, num_users, num_movies, ...
                                  num_features, lambda)
%COFICOSTFUNC Collaborative filtering cost function
%   [J, grad] = COFICOSTFUNC(params, Y, R, num_users, num_movies, ...
%   num_features, lambda) returns the cost and gradient for the
%   collaborative filtering problem.
%

% Unfold the U and W matrices from params
X = reshape(params(1:num_movies*num_features), num_movies, num_features);
Theta = reshape(params(num_movies*num_features+1:end), ...
                num_users, num_features);

            
% You need to return the following values correctly
J = 0;
X_grad = zeros(size(X)); % Nm x n , Nm = no of movies, n = no of features
Theta_grad = zeros(size(Theta)); % Nu x n, Nu = no of users

% ====================== YOUR CODE HERE ======================
% Instructions: Compute the cost function and gradient for collaborative
%               filtering. Concretely, you should first implement the cost
%               function (without regularization) and make sure it is
%               matches our costs. After that, you should implement the 
%               gradient and use the checkCostFunction routine to check
%               that the gradient is correct. Finally, you should implement
%               regularization.
%
% Notes: X - num_movies  x num_features matrix of movie features (Nm x n)
%        Theta - num_users  x num_features matrix of user features (Nu x n)
%        Y - num_movies x num_users matrix of user ratings of movies (Nm x Nu)
%        R - num_movies x num_users matrix, where R(i, j) = 1 if the  
%            i-th movie was rated by the j-th user (Nm x Nu)
%
% You should set the following variables correctly:
%
%        X_grad - num_movies x num_features matrix, containing the 
%                 partial derivatives w.r.t. to each element of X
%        Theta_grad - num_users x num_features matrix, containing the 
%                     partial derivatives w.r.t. to each element of Theta
%





Error = (X*Theta') - Y; %X*Theta' is the prediction matrix 
                        % Dim: Nm x Nu 
Error_active = Error.*R ; %note we only accuumulate cost for user j 
                                 %and movie i only if  R(i,j) = 1.
                                 %Elementwise multiplication makes non
                                 %rated errors zeros and any costs later on. 

% without regularisation                                 
%compute cost 
J = (1/2)*sum(sum(Error_active.^2)); % Nm x Nu %double sum due to being a matrix
%compute grads w.r.t. to each element of X and Theta
X_grad = Error_active * Theta; % (Nm x Nu) * Nu x n = Nm x n
Theta_grad = Error_active' * X; % (NmxNu)' * Nm x n = Nu x n

%now with regularisation term for X and Theta
reg_X = (lambda/2)*sum(sum(X.^2));
reg_theta = (lambda/2)*sum(sum(Theta.^2));

J = J + reg_X + reg_theta;
% =============================================================
X_grad = X_grad + lambda*X; 
Theta_grad = Theta_grad + lambda*Theta;

grad = [X_grad(:); Theta_grad(:)];

end
