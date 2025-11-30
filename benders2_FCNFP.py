from docplex.mp.model import Model
from cplex.callbacks import LazyConstraintCallback
from cplex import SparsePair # para criar o corte no formato raw
from reader import read_instance
import time

# benders via callback
# vamos tentar

class BendersCallback(LazyConstraintCallback):
    def set_models(self, worker_mdl, worker_x, worker_feas, cap_constrs, flow_constrs, feas_cap_constrs, feas_flow_constrs, master_y, master_eta, arcs_data, node_balances):
        self.worker_model = worker_mdl
        self.worker_feas = worker_feas
        self.worker_x = worker_x
        self.worker_constraints_cap = cap_constrs
        self.worker_constraints_flow = flow_constrs
        self.feas_constraints_cap = feas_cap_constrs
        self.feas_constraints_flow = feas_flow_constrs
        self.master_vars_y = master_y
        self.master_var_eta = master_eta
        self.arcs_data = arcs_data
        self.node_balances = node_balances
        self.arcs_map_w = {(i, j): w for (i, j, c, f, w) in arcs_data}
        self.num_cuts_optimality = 0
        self.num_cuts_feasibility = 0
        self.y_vars_list = list(self.master_vars_y.values())
        self.y_indices = [v.index for v in self.y_vars_list]
        self.eta_index = self.master_var_eta.index

    def __call__(self):
        eta_index = self.master_var_eta.index
        
        # pede os valores usando os índices já cacheados
        y_values = self.get_values(self.y_indices) 
        eta_val = self.get_values(self.eta_index)

        # index -> valor
        y_val_map = {self.y_vars_list[i].name: y_values[i] for i in range(len(self.y_vars_list))}

        # atualiza o worker com os y recuperados do mestre
        for (i, j), var_y in self.master_vars_y.items():
            val_y_binary = 1 if y_val_map[var_y.name] > 0.5 else 0
            w_ij = self.arcs_map_w[(i, j)]
            # atualiza RHS: x <= w * y
            self.worker_constraints_cap[(i, j)].rhs = w_ij * val_y_binary

        # resolve o worker
        self.worker_model.solve()
        status = self.worker_model.solve_details.status_code

        # se ótimo, verifica corte de otimidade
        if status == 1 or status == 'optimal':
            obj_worker = self.worker_model.objective_value
            
            if obj_worker > eta_val + 1e-4: # se passar do valor de eta
                # custo = sum(demanda * pi) + sum(capacidade * y * mu)
                term_constant = 0 # sum(b_i * pi_i)
                for node_id, balance in self.node_balances.items():
                    pi = self.worker_constraints_flow[node_id].dual_value
                    term_constant += balance * pi
                 
                # precisa construir os vetores de índices e coeficientes para o SparsePair
                cut_indices = [eta_index]
                cut_coeffs = [1.0] # coeficiente de eta é 1
                rhs_val = term_constant # termo constante vai para o RHS
                
                # eta - sum(mu * w * y) >= constant
                # eta + sum(-mu*w)*y >= constant
                
                for (i, j), ct in self.worker_constraints_cap.items():
                    mu = ct.dual_value
                    w_ij = self.arcs_map_w[(i, j)]
                    
                    # se for significativo
                    if abs(mu * w_ij) > 1e-3:
                        y_idx = self.master_vars_y[(i, j)].index
                        coeff = -1.0 * mu * w_ij
                        cut_indices.append(y_idx)
                        cut_coeffs.append(coeff)
                
                # adiciona corte "raw" (baixo nível)
                # self.add no CPLEX raw espera: (SparsePair(ind, val), sense, rhs)
                self.add(constraint=SparsePair(ind=cut_indices, val=cut_coeffs), sense="G", rhs=rhs_val)
                self.num_cuts_optimality += 1

        # caso inviável, corte de viabilidade
        elif status in [3, 'infeasible', 'infeasible_or_unbounded']:
            # atualizar o worker de viabilidade com os y do mestre
            for (i, j), var_y in self.master_vars_y.items():
                val_y_binary = 1 if y_val_map[var_y.name] > 0.5 else 0
                w_ij = self.arcs_map_w[(i, j)]
                self.feas_constraints_cap[(i, j)].rhs = w_ij * val_y_binary

            # resolve o worker
            self.worker_feas.solve()

            if self.worker_feas.objective_value < 1e-4:
                return
            
            # pega os duais
            term_constant = 0
            # acessa duais
            for node_id, balance in self.node_balances.items():
                pi_bar = self.feas_constraints_flow[node_id].dual_value
                term_constant += balance * pi_bar
            
            cut_indices = []
            cut_coeffs = []
            
            # acessa duais de capacidade
            for (i, j), var_y in self.master_vars_y.items():
                mu_bar = self.feas_constraints_cap[(i, j)].dual_value 
                w_ij = self.arcs_map_w[(i, j)]
                
                if abs(mu_bar * w_ij) > 1e-3:
                    y_idx = var_y.index
                    # sum(mu * w * y) >= constant
                    coeff = -1.0 * mu_bar * w_ij
                    
                    cut_indices.append(y_idx)
                    cut_coeffs.append(coeff)
            
            # adiciona os cortes
            if cut_indices:
                self.add(constraint=SparsePair(ind=cut_indices, val=cut_coeffs), sense="G", rhs=term_constant)
                self.num_cuts_feasibility += 1

        # se tudo der errado, restrições combinatórias de viabilidade:
        #elif status in [3, 'infeasible', 'infeasible_or_unbounded']:
            # corte: sum(1 - y) >= 1 para os y que eram 0
            # -sum(y_fechados) >= 1 - |conjunto| -> ajustando algebra
            # sum(y_fechados) >= 1
            #cut_indices = []
            #cut_coeffs = []
            #for idx, val in enumerate(y_values):
                #if val < 0.5: # se o arco estava fechado
                    #cut_indices.append(y_indices[idx])
                    #cut_coeffs.append(1.0)
            #if cut_indices:
                #self.add(constraint=SparsePair(ind=cut_indices, val=cut_coeffs), sense="G", rhs=1.0)

#solver
def solve_benders(params, node_balances, arcs_data, nodes):    
    # mestre
    mdl_master = Model(name="Master")
    y = {(i, j): mdl_master.binary_var(name=f"y_{i}_{j}") for (i, j, c, f, w) in arcs_data}
    eta = mdl_master.continuous_var(name="eta", lb=0)
    mdl_master.minimize(mdl_master.sum(f * y[(i, j)] for (i, j, c, f, w) in arcs_data) + eta)
    
    # worker
    mdl_worker = Model(name="Worker")
    mdl_worker.parameters.preprocessing.presolve = 0 
    x = {(i, j): mdl_worker.continuous_var(name=f"x_{i}_{j}", lb=0) for (i, j, c, f, w) in arcs_data}
    mdl_worker.minimize(mdl_worker.sum(c * x[(i, j)] for (i, j, c, f, w) in arcs_data))
    
    # restrições do worker
    ct_flow = {}
    for k in nodes:
        flow_out = mdl_worker.sum(x[(i, j)] for (i, j, c, f, w) in arcs_data if i == k)
        flow_in  = mdl_worker.sum(x[(i, j)] for (i, j, c, f, w) in arcs_data if j == k)
        ct_flow[k] = mdl_worker.add_constraint(flow_out - flow_in == node_balances[k], f"flow_{k}")

    ct_cap = {}
    for (i, j, c, f, w) in arcs_data:
        ct_cap[(i, j)] = mdl_worker.add_constraint(x[(i, j)] <= 0, f"cap_{i}_{j}")

    mdl_feas = mdl_worker.clone() # cópia pro worker de viabildsas
    mdl_feas.name = "Worker_Feasibility"
    
    # variáveis de folga nas restrições de fluxo
    # flow_out - flow_in + slack = balance
    feas_ct_cap = {}
    for (i, j, c, f, w) in arcs_data:
        # busca a restrição de capacidade no clone apenas uma vez aqui fora
        feas_ct_cap[(i, j)] = mdl_feas.get_constraint_by_name(f"cap_{i}_{j}")

    feas_slacks = {}
    feas_ct_flow = {} # restrições de fluxo modificadas
    
    for node in nodes:
        # variável de folga positiva (art_plus) e negativa (art_minus)
        s_pos = mdl_feas.continuous_var(lb=0, name=f"slack_pos_{node}")
        s_neg = mdl_feas.continuous_var(lb=0, name=f"slack_neg_{node}")
        feas_slacks[node] = (s_pos, s_neg)
        # pega a restrição antiga
        ct_old = mdl_feas.get_constraint_by_name(f"flow_{node}")
        # salva os dados dela
        lhs_expr = ct_old.left_expr
        rhs_val = ct_old.right_expr
        # remove a antiga
        mdl_feas.remove_constraint(ct_old)
        # add a nova e guarda
        new_ct = mdl_feas.add_constraint(lhs_expr + s_pos - s_neg == rhs_val, f"flow_{node}")
        feas_ct_flow[node] = new_ct

    # minimizar a soma das folgas
    mdl_feas.minimize(mdl_feas.sum(s[0] + s[1] for s in feas_slacks.values()))

    # CALLBACK
    cb = mdl_master.register_callback(BendersCallback)
    cb.set_models(mdl_worker, x, mdl_feas, ct_cap, ct_flow, feas_ct_cap, feas_ct_flow, y, eta, arcs_data, node_balances)
    
    # desativa o presolve do mestre para lazy constraints funcionarem bem
    mdl_master.parameters.preprocessing.presolve = 0
    start_time = time.time()
    sol = mdl_master.solve(log_output=True)
    end_time = time.time()
    
    if sol:
        print(f"OBJ: {sol.objective_value}")
        print(f"Tempo:    {end_time - start_time:.4f} s")
        print(f"Cortes de Otimidade:   {cb.num_cuts_optimality}")
        print(f"Cortes de Viabilidade: {cb.num_cuts_feasibility}")
    else:
        print("Sem solução.")

filename = "ndrc\instancia_frcf_2.txt"
print(f"Instância: {filename}")
params, node_balances, arcs_data = read_instance(filename)
nodes = list(node_balances.keys())
solve_benders(params, node_balances, arcs_data, nodes)