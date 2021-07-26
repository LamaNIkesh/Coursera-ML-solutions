function idx = findClosestCentroids(X, centroids)
%FINDCLOSESTCENTROIDS computes the centroid memberships for every example
%   idx = FINDCLOSESTCENTROIDS (X, centroids) returns the closest centroids
%   in idx for a dataset X where each row is a single example. idx = m x 1 
%   vector of centroid assignments (i.e. each entry in range [1..K])
%

% Set K
K = size(centroids, 1);

% You need to return the following variables correctly.
idx = zeros(size(X,1), 1);
%dist_to_centroid = zeros(K,1); %temp holder for the distances
% ====================== YOUR CODE HERE ======================
% Instructions: Go over every example, find its closest centroid, and store
%               the index inside idx at the appropriate location.
%               Concretely, idx(i) should contain the index of the centroid
%               closest to example i. Hence, it should be a value in the 
%               range 1..K
%
% Note: You can use a for-loop over the examples to compute this.
%

for i=1:size(X,1) %iteratre through data points
    dist_to_centroid = zeros(K,2); %temp holder for the distances into Kx2 array
   for j=1:K % iterate through each centroid
       dist_to_centroid (j,:) = [norm(X(i,:) - centroids(j,:))^2, j]; % store distance and the centroid idx   
   end
   sorted_dist = sortrows(dist_to_centroid,1);
   idx(i) = sorted_dist(1,2); %store just the centroid idx
   
end





% =============================================================

end

