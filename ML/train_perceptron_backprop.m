function w = train_perceptron_backprop(trainData, trainLbls, eta, nClasses)

train_tilde = [ones(1,size(trainData,2));trainData];
w = rand(size(trainData,1)+1, nClasses);
nTrainImages = size(trainData,2);

%train perceptron
for k = 1:nClasses
    wrongLabels = [];
    i = 0;
    label = 0;
    done = 0;
    while (done == 0)
        X = [];
        for i = 1:nTrainImages
            x_i = train_tilde(:,i);
            if(trainLbls(i) == k) label = 1;
            else label = -1;
            end
            %criterion function
            f = label*w(:,k)'*x_i;
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
        w(:,k) = w(:,k) + eta * sumOfWrongs;
        wrongLabels = [];
        sumOfWrongs = [];
        if (isempty(X))
            done = 1;
        end
    end
end

%confusion matrix
%testLbls -> 40x120. 0 p? alle pladser som den
%ikke er kvallet som, ellers 1.
%confusionmat(testLbls,resLabels)

%test
%g = w'*test_tilde
%r = g/norm(testData,2);




