clearvars; % Clears variables better than 'clear lines'
close all; 

% --- Setup ---
N = 10;
global G masses
G = 1;
masses = ones(N,1)';

% Random positions and velocities [-1, 1]
pos_column = 2*rand(2*N,1) - 1; 
vel_column = 2*rand(2*N,1) - 1;
STATE0 = [pos_column; vel_column]; 

% --- Time Settings for Animation ---
t_start = 0;
t_end = 5;   % Longer duration to see the chaos
dt = 0.0001;   % Time step specifically for the animation frames
Timespan = t_start:dt:t_end; % Vector: [0, 0.01, 0.02 ... 5.00]

% --- Solver ---
opts = odeset('RelTol', 1e-9, 'AbsTol', 1e-9);
% Passing the full 'Timespan' vector forces ode113 to output solution at these exact times
[T, sol] = ode113(@state_deriv, Timespan, STATE0, opts); 

% --- Animation Setup ---
figure('Color', 'w'); % White background
axis_limit = 2;       % Zoom level
axis([-axis_limit axis_limit -axis_limit axis_limit]);
axis equal;
grid on;
box on;
hold on;

% Extract data for easier indexing
% X and Y are matrices where rows = time, cols = bodies
X_sol = sol(:, 1:N);
Y_sol = sol(:, N+1:2*N);

% Initialize Plot Handles (Create the objects once)
% colors = lines(N); % Use a colormap for distinct colors
colors = hsv(N); % HSV is often distinct for N-body
head_handles = gobjects(N, 1);
tail_handles = gobjects(N, 1);

for i = 1:N
    % Initialize tails (empty to start)
    tail_handles(i) = plot(NaN, NaN, '-', 'Color', [colors(i,:) 0.5], 'LineWidth', 1);
    % Initialize heads (dots)
    head_handles(i) = plot(NaN, NaN, 'o', 'MarkerFaceColor', colors(i,:), ...
                           'MarkerEdgeColor', 'none', 'MarkerSize', 8);
end

title('N-Body Gravitational Simulation');
xlabel('X Position');
ylabel('Y Position');

% --- Animation Loop ---
disp('Starting animation...');
% We loop through the length of T (the number of time steps)
for k = 1:length(T)
    
    for i = 1:N
        % Update Head Position
        set(head_handles(i), 'XData', X_sol(k,i), 'YData', Y_sol(k,i));
        
        % Update Tail (History)
        % Let's keep the last 50 frames for the tail so it doesn't get too messy
        tail_start = max(1, k - 50); 
        set(tail_handles(i), 'XData', X_sol(tail_start:k, i), ...
                             'YData', Y_sol(tail_start:k, i));
    end
    
    drawnow limitrate; % updates the screen efficiently
    
    % Optional: Stop if the window is closed to prevent errors
    if ~isvalid(head_handles(1)) 
        break; 
    end
end
disp('Animation Complete.');


% --- Functions ---
function dYdt = state_deriv(~, Y) 
    N_pos = length(Y) / 2;
    dxdt = Y(N_pos + 1 : end); 
    dvdt = get_accel(Y(1 : N_pos));
    dYdt = [dxdt; dvdt]; 
end

function acc_col = get_accel(pos_col)
    global G masses
    N = length(masses);
    acc_col = zeros(2*N,1);
    
    % Re-shaping for easier vectorization could be done here, 
    % but your double loop is fine for N=5
    for i = 1:N
        p1 = [pos_col(i); pos_col(i+N)];
        accumulated_acc = [0; 0];
        
        for j = 1:N
            if i ~= j
                p2 = [pos_col(j); pos_col(j+N)];
                r_vec = p2 - p1;
                dist = norm(r_vec);
                
                % Add a "softening" parameter if bodies get too close 
                % to avoid infinity (optional, but good for stability)
                softening = 1e-2; 
                
                % F = G * m * r_vec / |r|^3
                a = G * masses(j) * r_vec / (dist^2 + softening^2)^(3/2);
                accumulated_acc = accumulated_acc + a;
            end
        end
        acc_col(i) = accumulated_acc(1);
        acc_col(i+N) = accumulated_acc(2);
    end
end