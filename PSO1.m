function [Cost, FN] = PSO1(NP, Max_FEs, EFs, Run, Input, Target, Cost, FN)

E = EFs;
nVar = size(Input, 2);   % Number of Decision Variables
VarSize = [1 nVar];      % Size of Decision Variables Matrix

% PSO Parameters
VarMin = -3;             % Lower Bound of Variables
VarMax = 3;              % Upper Bound of Variables
lu_v = 3 * [-ones(1, nVar); ones(1, nVar)];

nPop = NP;               % Population Size (Swarm Size)
c1 = 1.5;                % Personal Learning Coefficient
c2 = 2;                  % Global Learning Coefficient
competition_factor = 0.12;  % Competition factor

empty_particle.Position = [];
empty_particle.Cost = [];
empty_particle.Velocity = [];
empty_particle.Best.Position = [];
empty_particle.Best.Cost = [];

particle = repmat(empty_particle, nPop, 1);

GlobalBest.Cost = 0;

for i = 1:NP
    particle(i).Velocity = unifrnd(VarMin, VarMax, VarSize);
end

for i = 1:nVar
    particle(1).Velocity(1, i) = 3;
    particle(1).Position(1, i) = 1;
end

particle(1).Cost = Fit(Input, Target, particle(1).Position);

for i = 2:nPop
    % Initialize Velocity
    SS = 1 ./ (1 + exp(-particle(i).Velocity));
    R = rand(1, nVar);
    particle(i).Position = R < SS;
    particle(i).Cost = Fit(Input, Target, particle(i).Position);

    % Update Personal Best
    particle(i).Best.Position = particle(i).Position;
    particle(i).Best.Cost = particle(i).Cost;

    % Update Global Best
    if particle(i).Best.Cost > GlobalBest.Cost
        GlobalBest = particle(i).Best;
    end
end

% Update Personal Best
particle(1).Best.Position = particle(1).Position;
particle(1).Best.Cost = particle(1).Cost;

GlobalBest = particle(1).Best;

%% PSO Main Loop

while EFs <= Max_FEs
    EFs = EFs + NP;

    for i = 1:nPop
        % Update Velocity
        if EFs <= 0.5 * Max_FEs
            % Use global topology in the first half
            competition = competition_factor * (GlobalBest.Position - particle(i).Position);
        else
            % Switch to ring topology in the second half
            neighbor_index = mod(i, nPop) + 1;
            competition = competition_factor * (particle(neighbor_index).Position - particle(i).Position);
        end

        particle(i).Velocity = 1 * particle(i).Velocity ...
            + c1 * rand(VarSize) .* (particle(i).Best.Position - particle(i).Position) ...
            + c2 * rand(VarSize) .* (GlobalBest.Position - particle(i).Position) ...
            + competition;

        V_u = lu_v(2, :);
        particle(i).Velocity = (particle(i).Velocity > V_u) .* V_u + (particle(i).Velocity <= V_u) .* particle(i).Velocity;
        particle(i).Velocity = (particle(i).Velocity < -V_u) .* (-V_u) + (particle(i).Velocity >= -V_u) .* particle(i).Velocity;

        SS = 1 ./ (1 + exp(-particle(i).Velocity));
        R = rand(1, nVar);
        particle(i).Position = R < SS;

        % Evaluation
        particle(i).Cost = Fit(Input, Target, particle(i).Position);

        % Update Personal Best
        if particle(i).Cost > particle(i).Best.Cost
            particle(i).Best.Position = particle(i).Position;
            particle(i).Best.Cost = particle(i).Cost;

            % Update Global Best
            if particle(i).Best.Cost > GlobalBest.Cost
                GlobalBest = particle(i).Best;
            end
        end
    end

    Cost(E:E + NP, Run) = GlobalBest.Cost;

    disp(['SFE-CSO :   ' 'Function Evaluation: ' num2str(EFs) '   Accuracy = ' num2str(GlobalBest.Cost) '   Number of Selected Features = ' num2str(sum(GlobalBest.Position)) '   Run: ' num2str(Run)]);

    E = E + NP;
end

FN(Run, 1) = sum(GlobalBest.Position);

end
