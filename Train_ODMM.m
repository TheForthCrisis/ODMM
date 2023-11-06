function model = Train_ODMM(Xlist, y, lambda, theta, mu, tau, rho, varargin)
%% Initialization
qp_options = optimoptions('quadprog', 'Display', 'none');
[xdim, ydim, m]=size(Xlist);  % Size of matrix, number of samples
S0 = rand(xdim, ydim);  % initial value of S
Lam0 = rand(xdim, ydim);  % initial value of Lam

parser=inputParser;
addRequired(parser, 'Xlist', @(x) length(size(Xlist))==3);
addRequired(parser, 'y', @(x) size(y,2)==size(Xlist, 3));
addRequired(parser, 'lambda', @(x) x>0);
addRequired(parser, 'theta', @(x) x>0 && x<1);  % hyperparameter
addRequired(parser, 'mu', @(x) x>0 && x<=1);  % hyperparameter
addRequired(parser, 'rho', @(x) x>=0);  % hyperparameter
addRequired(parser, 'tau', @(x) x>=0);  % hyperparameter
addOptional(parser, 'eta', 0.999, @(x) x>0 && x<1);
addParameter(parser, 'S', S0);
addParameter(parser, 'Lam', Lam0);
addParameter(parser, 'iter', 100, @(x) x>0);
addParameter(parser, 'InfoOutput', 0)  % 0: no info; 1: iteration; 2: residual curve
addParameter(parser, 'Tolorance', 1e-6, @(x) x>0);
parse(parser, Xlist, y, lambda, theta, mu, tau, rho, varargin{:});

eta = parser.Results.eta;
S_old = parser.Results.S;
S_hat = parser.Results.S;
Lam_old = parser.Results.Lam;
Lam_hat = parser.Results.Lam;
iter = parser.Results.iter;
Tolorance = parser.Results.Tolorance;
info_flag = parser.Results.InfoOutput;

%% Solving
t = 1; t_new = 1;
residual_list = zeros(1, iter);
restart_flag = false(1, iter);

% Calculate auxiliary variables
X_vec = reshape(Xlist, [xdim*ydim, m])';
Xy_vec = X_vec.*y';
Q = Xy_vec*Xy_vec';

H = [Q + 0.5*m*(1+rho)*(1-theta)^2/lambda * eye(m, m), -Q;
    -Q, Q + 0.5*m*(1+rho)*(1-theta)^2/lambda/mu * eye(m, m)];

for k=1:iter
    % Calculate auxiliary variables
    temp = Lam_hat + rho * S_hat;
    u = y.*(X_vec*temp(:))';
    q = [(1+rho)*(theta-1) * ones(1, m) + u, (1+rho)*(theta+1) * ones(1, m) - u];
    % Solve W
    alpha=quadprog(H, q, [], [], [], [], zeros(1, 2*m), [], [], qp_options);
    
    v = y'.*(alpha(1:m) - alpha(m+1:2*m));
    W = reshape((v'*X_vec)', [xdim, ydim]);
    W = (W + Lam_hat + rho * S_hat) / (rho + 1);

    xi = 0.5*m*(1-theta)^2/lambda * alpha(1:m);
    epsilon = 0.5*m*(1-theta)^2/lambda/mu * alpha(m+1:2*m);
    % Solve b
    b = 0;
    flag_xi = xi > 0;
    flag_eps = epsilon > 0;
    if sum(flag_xi) > 0
        b = b + sum(flag_xi.*(y'.*(1-theta-xi) - X_vec*W(:)))/sum(flag_xi);
    end
    if sum(flag_eps) > 0
        b = b + sum(flag_eps.*(y'.*(1+theta+epsilon) - X_vec*W(:)))/sum(flag_eps);
    end

    % Solve S
    S = svt(rho * W - Lam_hat, tau) / rho;
    % Update multiplier
    Lam = Lam_hat - rho * (W - S);
    % Update residual
    residual_list(k) = sum((Lam - Lam_hat).^2, 'all') / rho + rho * sum((S - S_hat).^2, 'all');
    % Accelerate or restart
    if k == 1 || residual_list(k) < eta * residual_list(k-1)
        t_new = (1+sqrt(1+4*t^2))/2;
        S_hat = S + (t - 1)/t_new * (S - S_old);
        Lam_hat = Lam + (t - 1)/t_new * (Lam - Lam_old);
    else
        t_new = 1;
        S_hat = S_old;
        Lam_hat = Lam_old;
        residual_list(k) = residual_list(k-1) / eta;
    end
    
    t = t_new;
    S_old = S;
    Lam_old = Lam;
    
    % Iteration
    if info_flag >= 1
        fprintf('epoch -- %d/%d', k, iter)
        if restart_flag(k)
            fprintf(' -- restart\n');
        else
            fprintf('\n');
        end
    end
    
    if k>1 && abs(residual_list(k)-residual_list(k-1)) < Tolorance * residual_list(k)
        break
    end
end

%% Outputs
model.W = W;
model.b = b;

model.v = v;
model.Lam = Lam;
model.Lam0 = Lam0;
model.S = S;
model.S0 = S0;

model.lambda = lambda;
model.theta = theta;
model.mu = mu;
model.tau = tau;
model.rho = rho;
model.eta = eta;

model.residual_list = residual_list;

if info_flag >= 1
    fprintf('Number of Acceleration Step = %d\n', k - sum(restart_flag));
end
if info_flag >= 2
    figure()
    semilogy(1:iter, residual_list, 'r-o')
    ylabel('Residual'), xlabel('Iteration')
end