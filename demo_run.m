clear;
clc;
exp_n = 'FKCC';
data_name = 'ds_person_9120n_5625d_19c_8g_uni_CDKM200';
dir_name = [pwd, filesep, exp_n, filesep, data_name];
load(data_name);
y = Y;
nCluster = length(unique(y));
nSmp = length(y);
nBase = 20;

% **************************************************************************
% Parameter Configuration
% **************************************************************************

nRepeat = 10;
nMeasure = 17;

seed = 2024;
rng(seed, 'twister')

% Generate 50 random seeds
random_seeds = randi([0, 1000000], 1, nRepeat * nRepeat);

% Store the original state of the random number generator
original_rng_state = rng;

FKCC_result = zeros(nRepeat, nMeasure);


t1_s = tic;
t1 = toc(t1_s);
t2_s = tic;
for iRepeat = 1:nRepeat
    idx = (iRepeat - 1) * nBase + 1 : iRepeat * nBase;
    BPi = BPs(:, idx);
    
    % Restore the original state of the random number generator
    rng(original_rng_state);
    % Set the seed for the current iteration
    rng(random_seeds((iRepeat-1) * nRepeat+1));
    Hc = compute_Hc(BPi);
    t = 20;
    [bcs, baseClsSegs] = getAllSegs(Hc);
    clsSim = full(simxjac(baseClsSegs)); 
    clsSimRW = computePTS_II(clsSim, t);
    Hc_new = Hc * clsSimRW;
    [label_0, C] = kmeanspp(Hc_new', nCluster);
    entropy_type = 'Gini1';
    [label, iter_num, objHistory] = FKCC_v1(Hc_new, label_0', entropy_type, g');
    res_17 = my_eval_y_fair_mismatch(y, g, label);
    FKCC_result(iRepeat, :) = res_17;
end
t2 = toc(t2_s);
ts = [t1, t2];
FKCC_result_time = t1 + t2/nRepeat;
FKCC_result_mean = mean(FKCC_result, 1);
FKCC_result_std =  std(FKCC_result, 1);
