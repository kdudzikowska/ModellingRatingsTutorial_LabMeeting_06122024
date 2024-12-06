% Katarzyna Dudzikowska
% MSN & SDN Lab Meeting. 06.12.2024
% Computational modelling of continuous ratings: subjective experience of fatigue. 
% FULL SCRIPT
%% I. Load the data
clear all
%--CHANGE PATH HERE--%
datadir = ######################## ;
%---%
cd(datadir)

pps_files = dir('data/*.mat'); %Find all matlab files in the data directory
n_trials = 150; %Number of trials in the task
n_pps = length(pps_files); %Number of participants

%Extract participant ids
for i_pp = 1:n_pps
    ids(i_pp) = extractBetween(pps_files(i_pp).name, "s", ".");
end

% Initialise arrays for the data
ids = cell(n_pps,0);
population_fatigue = zeros(n_pps,n_trials);
population_initFatigue = zeros(n_pps,1);
population_fatigueChange = zeros(n_pps,n_trials);
population_effort = zeros(n_pps,n_trials);
population_success = zeros(n_pps,n_trials);

% Fill the arrays with data
for i_pp = 1:n_pps
    fprintf('%s\n',pps_files(i_pp).name);    
    ids(i_pp) = extractBetween(pps_files(i_pp).name, "s", "."); 
    load(pps_files(i_pp).name);
    
    %fatigue ratings collected throughout the task (75 out of 150 trials, NaN for the rest)
    population_fatigue(i_pp,:) = x.fatigue(1:end); 
    %baseline fatigue reported at the start
    population_initFatigue(i_pp,1) = x.initial_fatigue(1);
    %change in fatigue rating since the last obtained rating
    population_fatigueChange(i_pp,:) = x.fatigueChange(1:end);
    %effort level required on each trial, quantified as a proportion of Maximum 
    %Voluntary Contraction: 0 for rest trials and 0.30, 0.39 or 0.48
    population_effort(i_pp,:) = x.effort(1:end);
    %trial success: 1 if the goal of sustaining the required effort level
    %was achieved, 0 otherwise, NaN on rest trials
    population_success(i_pp,:) = x.success(1:end);
    %reward obtained on the trial: random number of points given for
    %successfully exerting effort and after rest (0 for a failure)
    population_reward(i_pp,:) = x.reward(1:end); 

end

%--Check out the data structures to know what we're working with--%

%% II. Plot the data

%% Plot fatigue ratings over trials for each participant

plo1 = tiledlayout(3,5);

title(plo1,'Fatigue in effort-based task')
xlabel(plo1,'Trial')
ylabel(plo1,'Fatigue rating')

for i_pp = 1:n_pps
    id = ids{i_pp};
    y_fatigue = population_fatigue(i_pp,:);
    x_trials = 1:n_trials;
    
    nexttile
    plot(x_trials, y_fatigue, 'c*') 
    title(['Ptcpt ', id]);
end

%% Plot average fatigue change 

% Calculating average fatigue change across participants
effort_levels = [0 0.3 0.39 0.48];
mean_fatigue_change_per_effort = NaN(length(effort_levels), 1);
err_fatigue_change_per_effort = NaN(length(effort_levels), 1);

for i_e = 1:length(effort_levels)
    e = effort_levels(i_e);
    e_fatigueChange = population_fatigueChange(population_effort==e);
    e_fatigueChange = e_fatigueChange(~isnan(e_fatigueChange));
    mean_fatigue_change_per_effort(i_e) = mean(e_fatigueChange);
    err_fatigue_change_per_effort(i_e) = std(e_fatigueChange)/ sqrt(length(e_fatigueChange));
end

% Plot
figure
errorbar(categorical(effort_levels), mean_fatigue_change_per_effort, err_fatigue_change_per_effort)

title('Fatigue change in effort-based task')
xlabel('Effort Level')
ylabel('Fatigue Change')

%% III. Hypothesised model: Full Fatigue Model with Unrecoverable & Recoverable Fatigue Components

% The function for the model is in the file: fatigue_estimate_UfRfRr.m

%--Check out the function script now--%

% Create a function handle for the model
fatigue_func = @(params, E, ratings, baseline) fatigue_estimate_UfRfRr(params, E, ratings, baseline);
% Give the model id and a title
i_model = 1;
model_titles{i_model} = 'UfRfRr'; 
%The model has 3 parameters: Unrecoverable Fatigue Rate, Recoverable Fatigue Rate, Recovery Rate
model_params(i_model) = 3; 

%% IV. Model fitting

%Run for each participant
for i_pp = 1:n_pps 
    id = char(ids(i_pp)); %get the id
    fprintf('%d/%d\n',i_pp, n_pps);
    fprintf('Subject ID: %s\n', id);
    
    %Load observed data
    E = population_effort(i_pp,:)'; %effort level on each trial
    fatigue_init = population_initFatigue(i_pp,1); %baseline fatigue rating
    fatigue_ratings = population_fatigue(i_pp,:)'; %task fatigue ratings 
            
    % Select the function for the model on which we want to run the fitting
    %procedure
    fit = @(params) fatigue_func(params, E, fatigue_ratings, fatigue_init);
    
    % IVa. Here we plug in different combinations of parameter values into the model function
    % to find parameter values that create estimates of fatigue that
    % most closely match the observed data (actual fatigue ratings).

    % Optimise parameter estimates through repeated sum-of-squares function minimization
    nRuns = 50; % We run fminsearch function 50 times
    n = 0; % Which run are we on
    fit_best = Inf; % Start with inf value of the best fit. We're trying to find the lowest possible value of this.
    
    % For each run...
    while n <= nRuns
        n = n + 1;
        
        % ...select new random starting points for the parameter values
        startp = rand(1,model_params(i_model)); 
        
        % ...constrain the parameters to be larger than 0 
        constrained_fit = @(params) fit(params) + (params(1)<0)*realmax + (params(2)<0)*realmax + (params(3)<0)*realmax; 

        % ...use fminsearch to find the best solution for the model, i.e.,
        % parameter values that result in the lowest value of the function (RSS)
        [parameter_values, fitk] = fminsearch(constrained_fit, startp, optimset('MaxFunEvals',100000,'MaxIter',100000));
        % ('MaxFunEvals' How many times the algorithm checks the value of
        % the function at different points;
        % 'MaxIter' How many adjustments the algorithm is allowed to make
        % in its search)
        
        % ...if value of the function on this try is lower than the best
        % result so far, update the best result
        if fitk < fit_best
            parameter_estimates_best{i_pp, i_model} = parameter_values;
            fit_best = fitk;
        end
    end
    
    % At this point you have optimal parameter estimates for this
    % participant. These are parameter values which result in the lowest RSS = best fit to observed ratings!

    % IVb. Fit the model with optimised parameters.
    % We use the best parameters from fminsearch in the model function to get the
    % estimates of fatigue
    [RSS_best(i_pp, i_model), fatigue_estimate_best{i_pp, i_model}, Rfat_estimate_best{i_pp, i_model}, Ufat_estimate_best{i_pp, i_model}] = fatigue_func(parameter_estimates_best{i_pp, i_model}, E, fatigue_ratings, fatigue_init);     

    % IVc. Calculate the model metrics for this model for this participant
    % We've got RSS as a measure of fit, but it doesn't take into account complexity of
    % the model. To avoid overfitting we look at AIC and BIC. Both metrics
    % balance the quality of fit with model simplicity - number of parameters. BIC
    % penalises complexity more.

    %Define k, n, and RSS
    k = model_params(i_model); %number of parameters
    n = length(fatigue_ratings(~isnan(fatigue_ratings))); %number of data points
    RSS = RSS_best(i_pp, i_model); %RSS

    %--Calculate AIC AND BIC--%
    
    %AIC formula:
    %AIC = 2k + n * ln(RSS/n)
    AIC(i_pp, i_model) = ############### ;

    %BIC formula:
    %BIC = n * ln(RSS/n) + k * ln(n)
    BIC(i_pp, i_model) = ############### ;
    
    %---%
        
    fprintf('Fitting of %s model completed.\nRSS = %f\nAIC = %f\nBIC = %f\n', char(model_titles(i_model)), RSS_best(i_pp, i_model), AIC(i_pp, i_model), BIC(i_pp, i_model));
end

%% V. Alternative model. Fatigue Model with Recoverable Fatigue Only

% You'll find the function for the recoverable model in the script fatigue_estimate_RfRr.m 

%--Check it out now.--%

% We repeat the fitting procedure for this model. 

%--Set up variables the same way as in full model--%

%Create a function handle for the model
fatigue_func = ############### ; %Model function handle
% Give the model id and a title
i_model = # ; %ID
model_titles{i_model} = ####### ; %Title
% How many parameters are in the model?
model_params(i_model) = # ;

%---%

% Fit the model
for i_pp = 1:n_pps %for each participant
    id = char(ids(i_pp)); %get the id
    fprintf('%d/%d\n',i_pp, n_pps);
    fprintf('Subject ID: %s\n', id);
    
    % Load observed data for this participant
    E = population_effort(i_pp,:)'; %effort level on each trial
    fatigue_init = population_initFatigue(i_pp,1); %baseline rating
    fatigue_ratings = population_fatigue(i_pp,:)'; %task ratings 
    
    % Select the model function 
    fit = @(params) fatigue_func(params, E, fatigue_ratings, fatigue_init);

    % Optimise parameter estimates through repeated sum-of-squares function minimization
    nRuns = 50;
    n = 0;
    fit_best = Inf;
    
    while n <= nRuns
        n = n + 1;
        startp = rand(1,model_params(i_model));
        
        % Constrain parameters (less parameters in this model!) 
        constrained_fit = ########################### ;
        
        % Use fminsearch to find the best solution for the model 
        [parameter_values, fitk] = ######################### ;

        % Update best
        if fitk < fit_best
            parameter_estimates_best{i_pp, i_model} = parameter_values;
            fit_best = fitk;
        end
    end
    
    % Now fit the model with optimised parameters:
    [RSS_best(i_pp, i_model), fatigue_estimate_best{i_pp, i_model}, Rfat_estimate_best{i_pp, i_model}, Ufat_estimate_best{i_pp, i_model}] = fatigue_func(parameter_estimates_best{i_pp, i_model}, E, fatigue_ratings, fatigue_init);     

    % Calculate the model metrics
    % Define k, n, and RSS
    k = model_params(i_model); %number of parameters
    n = length(fatigue_ratings(~isnan(fatigue_ratings))); %number of data points
    RSS = RSS_best(i_pp, i_model); %RSS

    % Calculate AIC AND BIC
    AIC(i_pp, i_model) = 2 * k + n * log(RSS/n);
    BIC(i_pp, i_model) = n * log(RSS/n) + k * log(n);
            
    fprintf('Fitting of %s model completed.\nRSS = %f\nAIC = %f\nBIC = %f\n', char(model_titles(i_model)), RSS_best(i_pp, i_model), AIC(i_pp, i_model), BIC(i_pp, i_model));
end

%% VI. And another alternative model: Fatigue Model with Unrecoverable Fatigue Only (DIY model function).

%--Duplicate the script fatigue_estimate_UfRfRr.m and change its name to
% fatigue_estimate_Uf.m--%

% Change the script to create a function where fatigue estimate consists
% only of unrecoverable fatigue (i.e., fatigue never decreases with rest):
% Change the function name to fatigue_estimate_Uf
% Change the number of parameters 
% Set recoverable fatigue on each trial to 0
% Make fatigue estimate on each trial be equal to estimate of unrecoverable
% fatigue only

%...unless we're running out of time, then use solution_Uf.m and change the name of the
%file to fatigue_estimate_Uf.m :D 

%---%

% We repeat the same fitting procedure for this model:
fatigue_func = @(params, E, ratings, baseline) fatigue_estimate_Uf(params, E, ratings, baseline); %Function handle
i_model = 3; %ID
model_titles{i_model} = 'Uf'; %Title
model_params(i_model) = 1; %Number of parameters

% Fit the model
for i_pp = 1:n_pps %for each participant
    id = char(ids(i_pp)); %get the id
    fprintf('%d/%d\n',i_pp, n_pps);
    fprintf('Subject ID: %s\n', id);
    
    % Load observed data for this participant
    E = population_effort(i_pp,:)'; %effort level on each trial
    fatigue_init = population_initFatigue(i_pp,1); %baseline rating
    fatigue_ratings = population_fatigue(i_pp,:)'; %task ratings 
    
    % Select the model function 
    fit = @(params) fatigue_func(params, E, fatigue_ratings, fatigue_init);

    % Optimise parameter estimates through repeated sum-of-squares function minimization
    nRuns = 50;
    n = 0;
    fit_best = Inf;
    
    while n <= nRuns
        n = n + 1;
        startp = rand(1,model_params(i_model));
        constrained_fit = @(params) fit(params) + (params(1)<0)*realmax; 

        % Use fminsearch to find the best solution for the model 
        [parameter_values, fitk] = fminsearch(constrained_fit, startp, optimset('MaxFunEvals',100000,'MaxIter',100000));

        % Update best
        if fitk < fit_best
            parameter_estimates_best{i_pp, i_model} = parameter_values;
            fit_best = fitk;
        end
    end
    
    % Now fit the model with optimised parameters:
    [RSS_best(i_pp, i_model), fatigue_estimate_best{i_pp, i_model}, Rfat_estimate_best{i_pp, i_model}, Ufat_estimate_best{i_pp, i_model}] = fatigue_func(parameter_estimates_best{i_pp, i_model}, E, fatigue_ratings, fatigue_init);     

    % Calculate the model metrics
    % Define k, n, and RSS
    k = model_params(i_model); %number of parameters
    n = length(fatigue_ratings(~isnan(fatigue_ratings))); %number of data points
    RSS = RSS_best(i_pp, i_model); %RSS

    % Calculate AIC AND BIC
    AIC(i_pp, i_model) = 2 * k + n * log(RSS/n);
    BIC(i_pp, i_model) = n * log(RSS/n) + k * log(n);
        
    fprintf('Fitting of %s model completed.\nRSS = %f\nAIC = %f\nBIC = %f\n', char(model_titles(i_model)), RSS_best(i_pp, i_model), AIC(i_pp, i_model), BIC(i_pp, i_model));
end

%% VII. Determine the winning (lowest AIC/BIC) model for each participant. 
n_models = i_model;

%% AIC
for i_pp = 1:n_pps %For each participant
    
    id = char(ids(i_pp)); %get the id
    fprintf('%d/%d Subject ID: %s\n',i_pp, n_pps, id);
    
    %...find the lowest AIC value 
    lowest_AIC = min(AIC(i_pp, 1:n_models));    
    %...find the id of the model with lowest AIC
    lowestAICid = find(AIC(i_pp, 1:n_models)==lowest_AIC);
    
    best_AIC_value(i_pp) = lowest_AIC;
    best_AIC_model(i_pp)= lowestAICid;
    best_AIC_model_title(i_pp) = model_titles(best_AIC_model(i_pp));
    best_AIC_params{i_pp} = parameter_estimates_best{i_pp, best_AIC_model(i_pp)};
    best_AIC_fatigue_estimates{i_pp, best_AIC_model(i_pp)} = fatigue_estimate_best{i_pp, best_AIC_model(i_pp)};

    fprintf('AIC model win: %s (AIC = %f)\n\n', char(model_titles(best_AIC_model(i_pp))), best_AIC_value(i_pp));
    
end

%% BIC
for i_pp = 1:n_pps %For each participant
    
    id = char(ids(i_pp)); %get the id
    fprintf('%d/%d Subject ID: %s\n',i_pp, n_pps, id);
    
    %--find the lowest BIC value--%
    lowest_BIC = ###############;
    %---%
    lowestBICid = find(BIC(i_pp, 1:n_models)==lowest_BIC); %model ID
    
    best_BIC_value(i_pp) = lowest_BIC;
    best_BIC_model(i_pp)= lowestBICid;
    best_BIC_model_title(i_pp) = model_titles(best_BIC_model(i_pp));
    best_BIC_params{i_pp} = parameter_estimates_best{i_pp, best_BIC_model(i_pp)};
    best_BIC_fatigue_estimates{i_pp, best_BIC_model(i_pp)} = fatigue_estimate_best{i_pp, best_BIC_model(i_pp)};
        
    fprintf('BIC model win: %s (BIC = %f)\n\n', char(model_titles(best_BIC_model(i_pp))), best_BIC_value(i_pp));

end

%% VIII. Determine the winning model across participants.

%For each model calculate AIC and BIC sum score for each model across
%participants & and proportion of wins between participants.

%Initialise variables to save results
AIC_sum = zeros(1, n_models); 
AIC_win_share = zeros(1, n_models);
best_AIC_sum = 0; 
best_AIC_sum_model_title = NaN; 
best_AIC_share = 0; 
best_AIC_share_model_title = NaN; 
BIC_sum = zeros(1, n_models);
BIC_win_share = zeros(1, n_models);
best_BIC_sum = 0;
best_BIC_sum_model_title = NaN;
best_BIC_share = 0;
best_BIC_share_model_title = NaN;

%% AIC
%For each model:
for i_model = 1:n_models
    AIC_sum(i_model) = sum(AIC(:, i_model), "omitnan"); %...find AIC sum score
    AIC_win_share(i_model) = mean(best_AIC_model==i_model); %...and proportion of AIC wins across participants
end

[best_AIC_sum, best_AIC_sum_model] = min(AIC_sum); %AIC overall best sum score
best_AIC_sum_model_title = model_titles(best_AIC_sum_model); %The name of the model with the best AIC sum score

[best_AIC_share, best_AIC_share_model] = max(AIC_win_share); %Highest proportion of AIC wins
best_AIC_share_model_title = model_titles(best_AIC_share_model); %The name of the model with the highest proportion of AIC wins

fprintf('\nAIC MODEL COMPARISON RESULTS\n')
fprintf('Win by sum score: %s (%f)\n', best_AIC_sum_model_title{1}, best_AIC_sum);
fprintf('Highest win share: %s (%f)\n', best_AIC_share_model_title{1}, best_AIC_share);

%% BIC
%--repeat for BIC--%

%For each model:
for i_model = 1:n_models
    BIC_sum(i_model) = ############### ; %...find BIC sum score
    BIC_win_share(i_model) = ############# ; %...and proportion of BIC wins across participants
end

[best_BIC_sum, best_BIC_sum_model] = ######### ; %BIC overall best sum score
best_BIC_sum_model_title = ############# ; %The name of the model with the best BIC sum score

[best_BIC_share, best_BIC_share_model] = ########## ; %Highest proportion of BIC wins
best_BIC_share_model_title = ############# ; %The name of the model with the highest proportion of BIC wins

%--%

fprintf('\nBIC MODEL COMPARISON RESULTS\n')
fprintf('Win by sum score: %s (%f)\n', best_BIC_sum_model_title{1}, best_BIC_sum);
fprintf('Highest win share: %s (%f)\n', best_BIC_share_model_title{1}, best_BIC_share);


%% IX. Visualise the results.

%% Plot sum of AIC
figure
bar(AIC_sum)
xticklabels(model_titles)
ylabel("AIC")
title("AIC sum")

%% Plot sum of BIC
figure
bar(#########)
xticklabels(model_titles)
ylabel("BIC")
title("BIC sum")

%% Plot proportion of AIC wins
figure
bar(AIC_win_share)
xticklabels(model_titles)
ylabel("proportion of participants")
title("AIC win share")

%% Plot proportion of BIC wins
figure
bar(#########)
xticklabels(model_titles)
ylabel("proporion of participants")
title("BIC win share")

%% Plot fatigue from winning model against observed fatigue ratings
plo = tiledlayout(3,5);

title(plo,'Fatigue: observed & estimates')
xlabel(plo,'Trial')
ylabel(plo,'Fatigue (z-scored)')

for i_pp = 1:n_pps
    id = ids{i_pp};
    y1_estimates = fatigue_estimate_best{i_pp, best_BIC_sum_model}';
    y2_observed = population_fatigue(i_pp,:);
    x_trials = 1:n_trials;
    
    nexttile
    plot(x_trials, y1_estimates,'b', x_trials, y2_observed, 'c*')
    title(['Ptcpt ', id]);

end

%% X. BONUS
% What alternative models can we try?