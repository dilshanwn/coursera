fprintf('Plotting Data ...\n')
data = load('ex1data1.txt');
X = data(:, 1); y = data(:, 2);
m = length(y); % number of training examples

X = [ones(m, 1), data(:,1)]; % Add a column of ones to x
theta = zeros(2, 1); % initialize fitting parameters

% Some gradient descent settings
iterations = 1500;
alpha = 0.01;

fprintf('\nTesting the cost function ...\n')
% compute and display initial cost
J = computeCost(X, y, theta);
fprintf('\nCost computed = %f\n', J);
fprintf('\nExpected cost value (approx) 32.07\n');


J = computeCost(X, y, [-1 ; 2]);
fprintf('\nCost computed = %f\n', J);
fprintf('Expected cost value (approx) 54.24\n');

