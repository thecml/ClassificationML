function w = train_perceptron_backprop(trainData, trainLbls, eta, nClasses, offset)

train_tilde = [ones(1,size(trainData,2));trainData];
w = rand(size(trainData,1)+1, nClasses);
nTrainImages = size(trainData,2);

%make a boundary variable for when labels start at 0
%and go to nClasses-1.
if(offset == 0) limit = 1;
else limit = 0;
end

%train perceptron
for k = offset:nClasses-limit
    wrongLabels = [];
    i = 0;
    label = 0;
    done = 0;
    nIters = 0;
    while (done == 0 && nIters < 100)
        X = [];
        for i = 1:nTrainImages
            x_i = train_tilde(:,i);
            if(trainLbls(i) == k) label = 1;
            else label = -1;
            end
            %criterion function
            if (offset == 0) f = label*w(:,k+1)'*x_i;
            else f = label*w(:,k)'*x_i;
            end
            if f < 0
               % save vector and labels that were wrong
               X = [X x_i];
               wrongLabels = [wrongLabels label];
            end
        end
        sumOfWrongs = zeros(size(trainData,1)+1,1);
        if size(X,2) > 0
            for j = 1:size(X,2)
                sumOfWrongs = sumOfWrongs + wrongLabels(j)*X(:,j);
            end
        end
        %update k of w with the wrongs we have to adjust it
        if (offset == 0) w(:,k+1) = w(:,k+1) + eta * sumOfWrongs;
        else w(:,k) = w(:,k) + eta * sumOfWrongs;
        end
        wrongLabels = [];
        sumOfWrongs = [];
        nIters = nIters+1;
        if (isempty(X))
            done = 1;
        end
    end
end



