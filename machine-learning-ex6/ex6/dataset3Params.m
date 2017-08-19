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

curr_c=C;
curr_sigma=sigma;
min_error=0;
error=0;
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
list=[0.01,0.03,0.1,0.3,1,3,10,30];

for i=1:length(list)
    curr_c=list(i);
    for j=1:length(list)
        curr_sigma=list(j);
        
        %fprintf('\ncurr_c = %f',curr_c);
        %fprintf('\ncurr_sigma = %f',curr_sigma);
        %fprintf('\ni = %f',i);
        %fprintf('\nj = %f',j);
        
        model= svmTrain(X, y, curr_c, @(x1, x2) gaussianKernel(x1, x2, curr_sigma));
        predictions=svmPredict(model,Xval);
        error=mean(double(predictions ~= yval));
        %fprintf('\nerror = %error',j);
        if i==1 && j==1
            min_error=error;
            C=curr_c;
            sigma=curr_sigma;
            %fprintf('\ninit steps');
            %fprintf('\nmin_error = %f',min_error);
            %fprintf('\nC = %f',C);
            %fprintf('\nsigma = %f',sigma);
            
        else if error<min_error
                %fprintf('\nmin_error = %f',min_error);
                %fprintf('\nC = %f',C);
                %fprintf('\nsigma = %f',sigma);
                %fprintf('\ncurr_c = %f',curr_c);
                %fprintf('\ncurr_sigma = %f',curr_sigma);
                min_error=error;
                C=curr_c;
                sigma=curr_sigma;
                %fprintf('\nC = %f',C);
                %fprintf('\nsigma = %f',sigma);
            end
        end
    end
end

            %fprintf('\nFinal');
            %fprintf('\nC = %f',C);
            %fprintf('\nsigma = %f',sigma);


% =========================================================================

end
