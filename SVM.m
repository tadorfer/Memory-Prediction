clear; % clear Workspace
timekeep_start = tic;


%% Loading Free Recall Data

load('mem0.mat'); % data for class 0
load('mem1.mat'); % data for class 1

labels0 = zeros(size(mem0_d10,1),1); % labels for class 0
labels1 = ones(size(mem1_d10,1),1); % labels for class 1
all_mem = [mem0_d10 labels0;mem1_d10 labels1]; % combine data with labels


%% Defining Parameters

boxc = [0.01 0.1 1 5 10]; % misclassification cost parameter 
sig = [0.5 1 5 7 10 15 20 30 50 100]; % bandwidth 'sigma' (for RBF)
run = 2; % 10 runs (for model consistency)
outer_cv = 10; % 10 outer loops to test model on held out dataset
inner_cv = 6; % 6 inner cross-validation loops to optimize boxc and sig
avcon = 2; % 10 control runs with permutated data (control dataset)


%% Defining Folds for Nested 10x6 Cross-Validation (CV)

% Defining indices for outer training folds (10-fold outer CV)
m1 = 0;
n1 = 1;
step1 = length(all_mem);
interval1 = step1-step1/10; % interval1 size: 5040-504
interval2 = step1/10; % interval2 size = 5040/10
for i1 = 1:10
    outer_train(:,i1) = [1:interval2*m1 interval2*i1+1:step1]; % (4536x10)
    n1 = n1+interval1;
    m1 = m1+1;
end

% Defining indices for outer test folds (10-fold outer CV)
n2 = 1;
for i2 = 1:10
    outer_test(:,i2) = n2:interval2*i2; % (504x10)
    n2 = n2+interval2;
end

% Defining indices for inner training folds (6-fold inner CV)
m2 = 0;
n3 = 1;
step2 = length(all_mem)*0.9;
interval3 = step2-step2/6; % interval3 size: 4536-756
interval4 = step2/6; % interval4 size = 4536/6
for i3 = 1:6
    inner_train(:,i3) = [1:interval4*m2 interval4*i3+1:step2]; % (3780x6)
    n3 = n3+interval3;
    m2 = m2+1;
end

% Defining indices for inner validation folds (6-fold CV)
n4 = 1;
for i4 = 1:6
    inner_val(:,i4) = n4:interval4*i4; % 756 indices, 6 folds (756x6)
    n4 = n4+interval4;
end


%% SVM Model

for runs = 1:run
    rand_perm = randperm(length(all_mem))';
    mem_exp = all_mem(rand_perm,:); % randomly order data
    randperms(:,runs) = rand_perm;
    for outCV = 1:outer_cv
        mem_outer_train = mem_exp(outer_train(:,outCV),:); 
        y = 0;
        for inCV = 1:inner_cv
            for i = 1:numel(boxc) 
                for j = 1:numel(sig) 
                    b = boxc(i);
                    s = sig(j);

                    train_mem = mem_outer_train(inner_train(:,inCV),:);
                    val_mem = mem_outer_train(inner_val(:,inCV),:);
                    trainval_inner = [train_mem;val_mem];
                
                    mdl = fitcsvm(train_mem(:,1:64),...
                        train_mem(:,65),...
                        'BoxConstraint',b,...
                        'KernelFunction','gaussian',...
                        'KernelScale',s,...
                        'Standardize',true);
                    
                    trainPreds = predict(mdl,train_mem(:,1:64));
                    valPreds = predict(mdl,val_mem(:,1:64));
                    trainLoss = 1-(mean(trainPreds==train_mem(:,65)));
                    valLoss = 1-(mean(valPreds==val_mem(:,65)));
                    
                    if y <= (50*inner_cv)-1
                        y = y+1; %(boxc x sig x inCV = 300)
                    else
                        break;
                    end

                    GridSearch(y,1) = ((i-1)*(b(end)-b(1)))/(10-1)+b(1); 
                    GridSearch(y,2) = ((j-1)*(s(end)-s(1)))/(20-1)+s(1); 
                    GridSearch(y,3) = (1-trainLoss)*100; 
                    GridSearch(y,4) = (1-valLoss)*100; 

                    disp(['Runs: ',num2str(runs),' | ',...
                        'OuterCV: ',num2str(outCV),' | ',...
                        'InnerCV: ',num2str(inCV),' | ',...
                        'BoxC: ',num2str(((i-1)*(b(end)-b(1)))/(10-1)+b(1)),' | ',...
                        'Sigma: ',num2str(((j-1)*(s(end)-s(1)))/(20-1)+s(1)),' | ',...
                        'Train_Acc: ',num2str(round((1-trainLoss)*100,2)),'%',' | ',...
                        'Val_Acc: ',num2str(round((1-valLoss)*100,2)),'%'])
                end
            end
        end
        
        % Average across 6 inner CV loops
        
        r = 1:50:(50*inner_cv);
        for e = 1:50
            GridSearch_avg(e,:) = GridSearch(r,4)';
            r = r+1;
        end
        
        GridSearch_pars = [GridSearch(1:50,[1 2]) mean(GridSearch_avg,2)];

        [max_val,idx] = max(GridSearch_pars(:,3)); 

        best_box = GridSearch_pars(idx,1); 
        best_sig = GridSearch_pars(idx,2); 
        
        mdl_test = fitcsvm(trainval_inner(:,1:64),...
            trainval_inner(:,65),...
            'BoxConstraint',best_box,...
            'KernelFunction','gaussian',...
            'KernelScale',best_sig,...
            'Standardize',true);

        disp(['Best Box Constraint: ',num2str(best_box),' | ',...
            'Best Sigma: ',num2str(best_sig),' | ',...
            'Best Validation Accuracy: ',num2str(round((max_val),2)),'%'])

        
        % Test Model with Optimized Parameters

        mem_outer_test = mem_exp(outer_test(:,outCV),:); 

        testPreds = predict(mdl_test,mem_outer_test(:,1:64)); 
        testLoss = 1-(mean(testPreds==mem_outer_test(:,65))); 
        val_acc = (1-testLoss)*100; 

        disp(['Best Testing Accuracy: ',num2str(val_acc),'%'])

        results(outCV,1) = val_acc;
        results(outCV,2) = best_box;
        results(outCV,3) = best_sig;
      
        testlabels_outer(:,outCV) = mem_outer_test(:,65);
        testpreds_outer(:,outCV) = testPreds;
    end
    all_results(runs,:) = results(:,1); 
    alltestlabels(:,:,runs) = testlabels_outer;
    alltestpreds(:,:,runs) = testpreds_outer;
    all_box(:,runs) = best_box;
    all_sig(:,runs) = best_sig;
end

avg_acc = mean(mean(all_results));


%% Significance Testing

for avg_con = 1:avcon
    z = 0;
    mem_con = all_mem(randperms(:,avg_con),:);
    for a_con = 1:4
        mem_con_a = mem_con(outer_train(:,a_con),:);
        mem_con_a(:,65) = mem_con_a(randperm(size(mem_con_a,1)),65);
        test_con = mem_con(outer_test(:,a_con),:);
        test_con(:,65) = test_con(randperm(size(test_con,1)),65);

        mdl_con = fitcsvm(mem_con_a(:,1:64),mem_con_a(:,65),'BoxConstraint',
                                    all_box(avg_con),'KernelFunction','gaussian',
                                    'KernelScale',all_sig(avg_con),'Standardize',true);

        trainPreds_con = predict(mdl_con,mem_con_a(:,1:64));
        valPreds_con = predict(mdl_con,test_con(:,1:64));
        trloss_con = 1-(mean(trainPreds_con==mem_con_a(:,65)));
        valloss_con = 1-(mean(valPreds_con==test_con(:,65)));

        if z <= avcon*4-1
            z = z+1;
        else
            break;
        end

        Control_Results(z,1,avg_con) = all_box(avg_con); 
        Control_Results(z,2,avg_con) = all_sig(avg_con); 
        Control_Results(z,3,avg_con) = round((1-trloss_con)*100,2); 
        Control_Results(z,4,avg_con) = round((1-valloss_con)*100,2);

        disp(['Rd:',num2str(a_con),' | ','Av: ',num2str(avg_con),' | ',...
            'It: ',num2str(z),' | ','BoxC: ',num2str(all_box(avg_con)),' | ',...
            'Sigma: ',num2str(all_sig(avg_con)),' | ',...
            'Train_Acc: ',num2str(round((1-trloss_con)*100,2)),'%',' | ',...
            'Val_Acc: ',num2str(round((1-valloss_con)*100,2)),'%'])
    end
end

Control_Results_avg = permute(Control_Results,[1 3 2]);
Control_Results_avg = reshape(Control_Results_avg,[],size(GridSearch,2),1);

ks = kstest(avg_acc);
ks_con = kstest(Control_Results_avg(:,4));

if ks && ks_con == 1
    fprintf('\nBoth data sets are not normally distributed. Proceed with Wilcoxon Rank Sum Test...\n\n')
end

p = ranksum(avg_acc,Control_Results_avg(:,4),'Tail','right');

disp(['Average Model Accuracy: ',num2str(avg_acc),...
    '% with Standard Deviation of ',num2str(std(avg_acc))])
disp(['Average Control Accuracy: ',num2str(mean(Control_Results_avg(:,4))),...
    '% with ','Standard Deviation of ',num2str(std(Control_Results_avg(:,4))),' | ',...
    'Significance: ',num2str(p)])


timekeep_end = toc(timekeep_start);
fprintf('%d minutes and %f seconds\n', floor(timekeep_end/60), rem(timekeep_end,60));
