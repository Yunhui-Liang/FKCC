function [Y_label, iter_num, objHistory] = FKCC_v1(X, label, entropy_type, gIndex)
Z = ind2vec(gIndex)';
[nSmp, nGroup] = size(Z);
nFea = size(X, 2);
Z = full(Z > 0);
Y = ind2vec(label')';
Y = ensure_nonzero_A(Z, Y);
Y = full(Y > 0);
nCluster = size(Y, 2);

last = 0;
iter_num = 0;

% store once
e = sum(X.^2, 2);
b = 1./sum(Z, 1); % 1* 2
ZTY = Z'*Y;
ZTY = full(ZTY); % 2 * 3

sYY = sum(Y, 1);
p = bsxfun(@times, ZTY, b');
h = compute_entropy_fair(p, entropy_type);

ZoYEs = zeros(nGroup, nCluster);
ZoYXs = cell(nGroup, 1);
ZoYXXYoZs = zeros(nGroup, nCluster);
for iGroup = 1:nGroup
    ZoYX = zeros(nFea, nCluster);
    for iCluster = 1:nCluster
        zoy = Z(:, iGroup) .* Y(:, iCluster);
        ZoYEs(iGroup, iCluster) = sum(zoy.*e);
        zoyX = zoy' * X;
        zoyX = zoyX';
        ZoYX(:, iCluster) = zoyX;
        ZoYXXYoZs(iGroup, iCluster) = sum(zoyX.^2);
    end
    ZoYXs{iGroup} = ZoYX;
end

obj_c = sum(h .* (ZoYEs - ZoYXXYoZs./max(ZTY, eps)), 1);
obj = sum(obj_c);
objHistory = obj;
iter = 0;
maxIter = 100;
while any(label ~= last)
    last = label;
    for iSmp = 1:nSmp
        j = label(iSmp);
        r = gIndex(iSmp);
        
        if ZTY(r, j) == 1
            continue;
        end
        
        %*********************************************************************
        % The following matlab code is O(dk)
        %*********************************************************************
        
        xi = X(iSmp, :);
        zy_1 = ZTY(r, :) + 1;
        pzy_1 = zy_1 * b(r);
        hbzy_1 = compute_entropy_fair(pzy_1, entropy_type);
        yze_1 = ZoYEs(r, :) + e(iSmp);
        yozXXyoz_1 = ZoYXXYoZs(r, :) + 2 * xi * ZoYXs{r} + e(iSmp);
        
        obj_1 = hbzy_1 .* (yze_1 - yozXXyoz_1./max(zy_1, eps));
        
        delta = obj_1 - obj_c; %% todo
        
        zy_0 = ZTY(r, j) - 1;
        pzy_0 = zy_0 * b(r);
        hbzy_0 = compute_entropy_fair(pzy_0, entropy_type);
        yze_0 = ZoYEs(r, j) - e(iSmp);
        yozXXyoz_0 = ZoYXXYoZs(r, j) - 2 * xi * ZoYXs{r}(:, j) + e(iSmp);
        obj_j_0 = hbzy_0 * (yze_0 - yozXXyoz_0/max(zy_0, eps));
        
        delta(j) = obj_c(j) - obj_j_0;
        
        [~, j_star] = min(delta);
        if j_star ~= j % sample i is moved from cluster j to cluster j_star
            ZTY(r, j_star) = ZTY(r, j_star) + 1;
            ZTY(r, j) = ZTY(r, j) - 1;
            Y(iSmp, [j, j_star]) = [false, true];
            ZoYEs(r, j) = yze_0;
            ZoYEs(r, j_star) = yze_1(j_star);
            ZoYXs{r}(:, j) = ZoYXs{r}(:, j) - xi';
            ZoYXs{r}(:, j_star) = ZoYXs{r}(:, j_star) + xi';
            ZoYXXYoZs(r, j) = yozXXyoz_0;
            ZoYXXYoZs(r, j_star) = yozXXyoz_1(j_star);
            sYY(j) = sYY(j) - 1;
            sYY(j_star) = sYY(j_star) + 1;
            obj_c([j, j_star]) = [obj_j_0, obj_1(j_star)];
            label(iSmp) = j_star;
            obj = sum(obj_c);
            objHistory = [objHistory; obj]; %#ok
        end
    end
    iter = iter + 1;
    if iter > maxIter
        break;
    end
end
Y_label = label;
end