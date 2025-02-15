function Y = ensure_nonzero_A(G, Y)
    % Inputs:
    % G: n-by-r binary matrix, where each column represents a group
    % Y: n-by-r binary matrix, where each column represents a cluster
    % Output:
    % Y: Updated Y matrix such that A = G' * Y has no zero entries

    % Compute A = G' * Y
    A = G' * Y;

    % Iterate until all entries in A are non-zero
    while any(A(:) == 0)
        % Find the zero entries in A
        [zero_rows, zero_cols] = find(A == 0);

        % Process each zero entry
        for k = 1:length(zero_rows)
            r = zero_rows(k); % Group index
            q = zero_cols(k); % Cluster index

            % Find the largest a_{rq'} in the same group r
            [max_val, max_cluster] = max(A(r, :));

            % Ensure the largest cluster is not the same as the empty cluster
            if max_cluster == q
                continue; % Skip if the largest cluster is the empty one
            end

            % Find samples in group r and cluster max_cluster
            group_samples = find(G(:, r) == 1); % Indices of samples in group r
            cluster_samples = find(Y(:, max_cluster) == 1); % Indices of samples in cluster max_cluster
            candidates = intersect(group_samples, cluster_samples); % Samples in group r and cluster max_cluster

            % Move the first candidate to cluster q
            if ~isempty(candidates)
                sample_to_move = candidates(1); % Move the first candidate
                Y(sample_to_move, max_cluster) = 0; % Remove from cluster max_cluster
                Y(sample_to_move, q) = 1; % Add to cluster q

                % Recompute A after moving the sample
                A = G' * Y;
            end
        end
    end
end