function Dt = svt(A, tau)
[U, S, V]=svd(A, 'econ');
k = 0;
while k<size(S, 1)
    if S(k+1, k+1)<=tau
        break
    end
    k = k+1;
end
Dt = U(:, 1:k) * S(1:k, 1:k) * V(:, 1:k)';
end