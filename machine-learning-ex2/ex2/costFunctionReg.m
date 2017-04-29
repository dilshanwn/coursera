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


[cost_temp, grad_temp] = costFunction(theta, X, y);
theta_temp=theta(2:(size(theta,1)),1);

J=cost_temp+ ((lambda/(2*m))*(theta_temp'*theta_temp));
temp=((lambda/m).*theta_temp);
temp=[0;temp]
grad=grad_temp+temp;

% =============================================================

end
