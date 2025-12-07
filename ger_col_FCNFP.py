import os
from reader import read_instance
from docplex.mp.model import Model

class FCNFP_GC:
    def __init__(self, nodes, arcs, supplies, costs, fixed_costs, capacities): # inicializador
        self.nodes = nodes
        self.arcs = arcs
        self.supplies = supplies
        self.costs = costs
        self.fixed_costs = fixed_costs
        self.capacities = capacities
        
        self.generated_columns = []
        self.duals_capacity = {} # pi
        self.dual_convexity = 0.0 # alpha

    # subproblema: fluxo de custo mínimo
    def solve_subproblem(self):
        # custo ajustado
        # (c - pi) - coeficiente do x no cálculo do custo reduzido
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
            
        # pega a solução primal
        y_vals = {a: sol.get_value(y[a]) for a in self.arcs if sol.get_value(y[a]) > 1e-5}
        lambda_vals = {k: sol.get_value(lambdas[k]) for k in K if sol.get_value(lambdas[k]) > 1e-5}
            
        return sol.objective_value, y_vals, lambda_vals

    def run(self, tolerance=1e-5):
        print(f"GERAÇÃO DE COLUNAS")
        
        self.duals_capacity = {(i,j): 0.0 for (i,j) in self.arcs}
        self.dual_convexity = 0.0
        
        init_pattern, _ = self.solve_subproblem() # primeira coluna
        if init_pattern is None:
            print("Erro: Subproblema inviável na inicialização.")
            return
        self.generated_columns.append(init_pattern)
        print(f"Coluna inicial gerada.")
        
        final_obj = 0.0
        final_y = {}
        final_lambdas = {}
        iter_count = 0

        while True:
            iter_count += 1
            
            # resolve o mestre e pega a solução
            obj_val, y_sol, lambda_sol = self.solve_master()
            
            if obj_val is None: 
                break
            
            # chama o subproblema
            new_column, reduced_cost = self.solve_subproblem()
            if new_column is None:
                print("Erro no subproblema.")
                break

            print(f"Iter: {iter_count}, Obj Mestre = {obj_val:.4f}, Custo Reduzido = {reduced_cost:.6f}")
            
            # filtra os fluxos positivos para visualizar a coluna gerada
            fluxos_ativos = {arc: flow for arc, flow in new_column.items() if flow > 1e-5}
            print(f"Coluna gerada: {fluxos_ativos}")

            # parada em caso de custo reduzido positivo
            if reduced_cost >= -tolerance:
                print(f"PARADA: solução ótima.")
                final_obj = obj_val
                final_y = y_sol
                final_lambdas = lambda_sol
                break
            
            # parada em caso de ciclagem
            if new_column in self.generated_columns:
                print("PARADA: nenhuma coluna nova gerada.")
                final_obj = obj_val
                final_y = y_sol
                final_lambdas = lambda_sol
                break

            self.generated_columns.append(new_column)
            final_obj = obj_val
            final_y = y_sol
            final_lambdas = lambda_sol
        
        if final_obj > 0:
            print("\nSOLUÇÃO\n")
            print(f"Custo Ótimo: {final_obj:.4f}")
            print(f"Iterações Totais: {iter_count}\n")
            
            #print("Arcos abertos (y):")
            #if not final_y:
            #    print("   Nenhum arco aberto (y=0).")
            #for a, val in final_y.items():
            #    print(f"Arco {a}: y = {val:.4f}")
            
            #print("Fluxo nos Arcos (x_total):")
            # fluxo total na rede: soma(lambda_k * fluxo_k) (x original)
            #aggregated_flow = {(i,j): 0.0 for (i,j) in self.arcs}
            
            #for k, l_val in final_lambdas.items():
            #    col = self.generated_columns[k]
            #    for arc in self.arcs:
            #        aggregated_flow[arc] += l_val * col[arc]
            
            #for arc, flow in aggregated_flow.items():
            #    if flow > 1e-5:
            #        print(f"Arco {arc}: fluxo = {flow:.4f}")
            
        else:
             print("\nMÉTODO FALHOU.")
        
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

for (i, j, c, f, w) in raw_arcs: # transforma em dicionário indexado por arco
    arcs.append((i, j))
    costs[(i, j)] = c
    fixed_costs[(i, j)] = f
    capacities[(i, j)] = w

cp = FCNFP_GC(nodes, arcs, supplies, costs, fixed_costs, capacities)
cp.run()