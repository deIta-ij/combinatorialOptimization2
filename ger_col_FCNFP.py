import os
from reader import read_instance
from docplex.mp.model import Model

class FCNFP_CPLEX:
    def __init__(self, nodes, arcs, supplies, costs, fixed_costs, capacities): # inicializador
        self.nodes = nodes
        self.arcs = arcs
        self.supplies = supplies
        self.costs = costs
        self.fixed_costs = fixed_costs
        self.capacities = capacities
        
        self.generated_columns = []
        self.duals_capacity = {} # pi_ij
        self.dual_convexity = 0.0 # alpha

    # subproblema: fluxo de custo mínimo
    def solve_subproblem(self):
        # custo "ajustado"
        # (c - muA) - coeficiente do x* no cálculo do custo reduzido
        adjusted_costs = {}
        for (i, j) in self.arcs:
            pi = self.duals_capacity.get((i, j), 0.0)
            adjusted_costs[(i, j)] = self.costs[(i, j)] - pi

        mdl = Model(name='Subproblem')
        
        # variável
        x = mdl.continuous_var_dict(self.arcs, lb=0, ub=self.capacities, name='x')
        
        # função objetivo
        mdl.minimize(mdl.sum(adjusted_costs[a] * x[a] for a in self.arcs))
        
        # restrição: conservação de fluxo
        for n in self.nodes:
            # pra se o nó não tiver b definido, assumir 0
            demand = self.supplies.get(n, 0)
            
            flow_in = mdl.sum(x[(i, j)] for (i, j) in self.arcs if j == n)
            flow_out = mdl.sum(x[(i, j)] for (i, j) in self.arcs if i == n)
            
            mdl.add_constraint(flow_out - flow_in == demand, f'Const_{n}')
            
        # solver
        mdl.context.solver.log_output = False
        sol = mdl.solve()
        
        if not sol:
            # poderia falhar com uma demanda impossível
            return None, None

        obj_val = sol.objective_value
        reduced_cost = obj_val - self.dual_convexity
        
        # salvando os resultados
        results = {(i, j): sol.get_value(x[(i, j)]) for (i, j) in self.arcs}
        
        return results, reduced_cost

    # problema mestre restrito
    def solve_master(self):
        mdl = Model(name='Master')
        K = range(len(self.generated_columns))
        
        # variáveis
        lambdas = mdl.continuous_var_dict(K, lb=0, name='lambda')
        y = mdl.continuous_var_dict(self.arcs, lb=0, ub=1, name='y')
        
        # função objetivo
        col_costs = []
        for col in self.generated_columns:
            c_k = sum(self.costs[a] * col[a] for a in self.arcs)
            col_costs.append(c_k)
            
        flow = mdl.sum(col_costs[k] * lambdas[k] for k in K)
        fixed = mdl.sum(self.fixed_costs[a] * y[a] for a in self.arcs)
        
        mdl.minimize(flow + fixed)
        
        # restrições
        convexity = mdl.add_constraint(mdl.sum(lambdas[k] for k in K) == 1, 'Convexity')
        
        capacity = {}
        for (i, j) in self.arcs:
            flow_sum = mdl.sum(lambdas[k] * self.generated_columns[k][(i, j)] for k in K)
            # fluxo <= capacidade * y
            ct = mdl.add_constraint(flow_sum - self.capacities[(i, j)] * y[(i, j)] <= 0, f'Capacity_{i}_{j}')
            capacity[(i, j)] = ct

        # solver
        mdl.context.solver.log_output = False
        sol = mdl.solve()
        
        if not sol:
            # pode indicar uma coluna inicial inválida
            print("Problema Mestre inviável.")
            return None

        # duais
        self.dual_convexity = convexity.dual_value
        self.duals_capacity = {}
        for (i, j) in self.arcs:
            self.duals_capacity[(i, j)] = capacity[(i, j)].dual_value
            
        return sol.objective_value

    def run(self, max_iter=2000, tolerance=1e-5):
        print(f"    GERAÇÃO DE COLUNAS")
        
        self.duals_capacity = {(i,j): 0.0 for (i,j) in self.arcs}
        self.dual_convexity = 0.0
        
        # geração de coluna inicial
        init_pattern, _ = self.solve_subproblem()
        if init_pattern is None:
            # isso não deve ocorrer já que as instâncias 
            # têm redes q suportam a demanda e não são desconexas
            print("Erro: Subproblema inviável na inicialização.")
            return
        self.generated_columns.append(init_pattern)
        print(f"Coluna inicial gerada.")
        
        final_obj = 0
        for iter in range(1, max_iter + 1):
            # chama o mestre
            obj_val = self.solve_master()
            if obj_val is None: 
                # aqui seria o caso de adicionar variáveis
                # "artificiais"
                break
            
            # chama o subproblema
            new_column, reduced_cost = self.solve_subproblem()
            if new_column is None:
                print("Erro no subproblema.")
                break

            active_flows = [f"{k}: {v:.1f}" for k, v in new_column.items() if v > 1e-5]
            print(f"Coluna gerada: {active_flows}")
            print(f"Iter {iter}: Obj Mestre = {obj_val:.4f}, Custo Reduzido = {reduced_cost:.6f}")
            
            if reduced_cost >= -tolerance:
                print("Algoritmo convergiu.")
                final_obj = obj_val
                break
            
            self.generated_columns.append(new_column)
            final_obj = obj_val
        
        if final_obj > 0:
            print("\n   SOLUÇÃO")
            print(f"Custo Ótimo: {final_obj:.4f}")
        else:
             print("\nFALHOU.")
        
        return final_obj

# EXECUÇÃO

instance_file = "instances/instancia_frcf_5.txt"
params, raw_balances, raw_arcs = read_instance(instance_file)

nodes = list(raw_balances.keys())
for i, j, _, _, _ in raw_arcs:
    if i not in nodes: nodes.append(i)
    if j not in nodes: nodes.append(j)
    
supplies = raw_balances
arcs = []
costs = {}
fixed_costs = {}
capacities = {}

for (i, j, c, f, w) in raw_arcs:
    arcs.append((i, j))
    costs[(i, j)] = c
    fixed_costs[(i, j)] = f
    capacities[(i, j)] = w
cp = FCNFP_CPLEX(nodes, arcs, supplies, costs, fixed_costs, capacities)
cp.run()