function H = compute_entropy_fair(p_k, entropy_type)
% 检查输入是否有效
if ~isnumeric(p_k) || length(p_k) < 1
    error('p_k must be a numeric vector with at least one element.');
end
if ~ischar(entropy_type) || isempty(entropy_type)
    error('entropy_type must be a string.');
end

% 计算熵度
switch entropy_type
    case 'Gini1'
        H = p_k.^2;
    otherwise
        error('Unknown entropy type: %s', entropy_type);
end

end