from docplex.mp.model import Model
from cplex.callbacks import LazyConstraintCallback, UserCutCallback
from cplex import SparsePair
from reader import read_instance
import time

# DECOMPOSIÇÃO DE BENDERS

# classe auxiliar para o benders

class BendersCutGenerator:
    def __init__(self, worker_mdl, worker_feas, cap_constrs, flow_constrs, 
                 feas_cap_constrs, feas_flow_constrs, master_y, master_eta, 
                 arcs_data, node_balances):
        self.worker_model = worker_mdl
        self.worker_feas = worker_feas
        self.worker_constraints_cap = cap_constrs
        self.worker_constraints_flow = flow_constrs
        self.feas_constraints_cap = feas_cap_constrs
        self.feas_constraints_flow = feas_flow_constrs
        self.master_vars_y = master_y
        self.master_var_eta = master_eta
        self.arcs_data = arcs_data
        self.node_balances = node_balances
        self.arcs_map_w = {(i, j): w for (i, j, c, f, w) in arcs_data}
        
        self.y_vars_list = list(self.master_vars_y.values())
        self.y_indices = [v.index for v in self.y_vars_list]
        self.y_names = [v.name for v in self.y_vars_list]
        self.eta_index = self.master_var_eta.index

        self.num_cuts_optimality = 0
        self.num_cuts_feasibility = 0

    def generate(self, callback_context, is_integer_check=True):
        # recupera valores do master
        y_values = callback_context.get_values(self.y_indices)
        eta_val = callback_context.get_values(self.eta_index)
        
        y_val_map = {name: val for name, val in zip(self.y_names, y_values)}

        # worker de otimalidade
        for (i, j), var_y in self.master_vars_y.items():
            val_y = y_val_map[var_y.name]
            
            if is_integer_check:
                val_y_effective = 1.0 if val_y > 0.5 else 0.0
            else:
                val_y_effective = val_y 

            w_ij = self.arcs_map_w[(i, j)]
            self.worker_constraints_cap[(i, j)].rhs = w_ij * val_y_effective

        # Tenta resolver o Worker Original
        sol_opt = self.worker_model.solve()
        
        # se sol_opt é None, o docplex indica que falhou ou é inviável
        # assume inviabilidade para tentar o corte de viabilidade
        if sol_opt is None:
            status_code = 3 # código CPLEX para infeasible
        else:
            status_code = self.worker_model.solve_details.status_code

        # Corte de Otimalidade
        if status_code == 1 or str(status_code) == 'optimal':
            obj_worker = self.worker_model.objective_value
            
            if obj_worker > eta_val + 1e-4:
                term_constant = 0
                for node_id, balance in self.node_balances.items():
                    pi = self.worker_constraints_flow[node_id].dual_value
                    term_constant += balance * pi
                 
                cut_indices = [self.eta_index]
                cut_coeffs = [1.0]
                rhs_val = term_constant
                
                for (i, j), ct in self.worker_constraints_cap.items():
                    mu = ct.dual_value
                    w_ij = self.arcs_map_w[(i, j)]
                    
                    if abs(mu * w_ij) > 1e-3:
                        y_idx = self.master_vars_y[(i, j)].index
                        coeff = -1.0 * mu * w_ij
                        cut_indices.append(y_idx)
                        cut_coeffs.append(coeff)
                
                cut_obj = SparsePair(ind=cut_indices, val=cut_coeffs)
                
                # constraint vs cut
                if is_integer_check:
                    callback_context.add(constraint=cut_obj, sense="G", rhs=rhs_val)
                else:
                    callback_context.add(cut=cut_obj, sense="G", rhs=rhs_val)
                
                self.num_cuts_optimality += 1
                return 

        # Corte de Viabilidade
        # se for None (tratado acima) ou status de inviabilidade
        if sol_opt is None or status_code in [3, 'infeasible', 'infeasible_or_unbounded']:
            
            # atualiza worker de viabilidade
            for (i, j), var_y in self.master_vars_y.items():
                val_y = y_val_map[var_y.name]
                if is_integer_check:
                    val_y_effective = 1.0 if val_y > 0.5 else 0.0
                else:
                    val_y_effective = val_y

                w_ij = self.arcs_map_w[(i, j)]
                self.feas_constraints_cap[(i, j)].rhs = w_ij * val_y_effective

            # resolve
            sol_feas = self.worker_feas.solve()
            
            # se aqui der none, tem que ver isso aí, as slacks deveriam garantir viabilidade
            if sol_feas is None:
                # print("Worker de Viabilidade falhou (Verifique o Worker).")
                return

            # Se a soma das violações for positiva
            if self.worker_feas.objective_value > 1e-4:
                term_constant = 0
                for node_id, balance in self.node_balances.items():
                    pi_bar = self.feas_constraints_flow[node_id].dual_value
                    term_constant += balance * pi_bar
                
                cut_indices = []
                cut_coeffs = []
                
                for (i, j), var_y in self.master_vars_y.items():
                    mu_bar = self.feas_constraints_cap[(i, j)].dual_value 
                    w_ij = self.arcs_map_w[(i, j)]
                    
                    if abs(mu_bar * w_ij) > 1e-3:
                        y_idx = var_y.index
                        coeff = -1.0 * mu_bar * w_ij
                        cut_indices.append(y_idx)
                        cut_coeffs.append(coeff)
                
                if cut_indices:
                    cut_obj = SparsePair(ind=cut_indices, val=cut_coeffs)
                    if is_integer_check:
                        callback_context.add(constraint=cut_obj, sense="G", rhs=term_constant)
                    else:
                        callback_context.add(cut=cut_obj, sense="G", rhs=term_constant)
                        
                    self.num_cuts_feasibility += 1

# CLASSES DE CALLBACK

class BendersLazyCallback(LazyConstraintCallback):
    def set_generator(self, generator):
        self.generator = generator
        
    def __call__(self):
        self.generator.generate(self, is_integer_check=True)

class BendersUserCallback(UserCutCallback):
    def set_generator(self, generator):
        self.generator = generator

    def __call__(self):
        # controle de profundidade para não pesar demais
        depth = self.get_current_node_depth()
        # user cuts apenas nos primeiros 10 níveis da árvore
        if depth < 1: 
            self.generator.generate(self, is_integer_check=False)

# LOOP PRINCIPAL

def solve_benders(params, node_balances, arcs_data, nodes):    
    # problema mestre
    mdl_master = Model(name="Master")
    y = {(i, j): mdl_master.binary_var(name=f"y_{i}_{j}") for (i, j, c, f, w) in arcs_data}
    eta = mdl_master.continuous_var(name="eta", lb=0)
    mdl_master.minimize(mdl_master.sum(f * y[(i, j)] for (i, j, c, f, w) in arcs_data) + eta)
    
    # subproblema
    mdl_worker = Model(name="Worker")
    mdl_worker.parameters.preprocessing.presolve = 0 
    mdl_worker.parameters.threads = 1
    x = {(i, j): mdl_worker.continuous_var(name=f"x_{i}_{j}", lb=0) for (i, j, c, f, w) in arcs_data}
    mdl_worker.minimize(mdl_worker.sum(c * x[(i, j)] for (i, j, c, f, w) in arcs_data))
    
    ct_flow = {} # restrição de balanço de fluxo
    for k in nodes:
        flow_out = mdl_worker.sum(x[(i, j)] for (i, j, c, f, w) in arcs_data if i == k)
        flow_in  = mdl_worker.sum(x[(i, j)] for (i, j, c, f, w) in arcs_data if j == k)
        ct_flow[k] = mdl_worker.add_constraint(flow_out - flow_in == node_balances[k], f"flow_{k}")

    ct_cap = {} # restrição de capacidade do arco inicial (começa <= 0)
    for (i, j, c, f, w) in arcs_data:
        ct_cap[(i, j)] = mdl_worker.add_constraint(x[(i, j)] <= 0, f"cap_{i}_{j}")

    # worker de viabilidade - slack variables (clone do subproblema)
    mdl_feas = mdl_worker.clone()
    mdl_feas.name = "Worker_Feasibility"
    mdl_feas.parameters.threads = 1
    
    feas_ct_cap = {}
    for (i, j, c, f, w) in arcs_data:
        feas_ct_cap[(i, j)] = mdl_feas.get_constraint_by_name(f"cap_{i}_{j}")

    feas_slacks = {}
    feas_ct_flow = {} 
    
    for node in nodes:
        s_pos = mdl_feas.continuous_var(lb=0, name=f"slack_pos_{node}")
        s_neg = mdl_feas.continuous_var(lb=0, name=f"slack_neg_{node}")
        feas_slacks[node] = (s_pos, s_neg)
        ct_old = mdl_feas.get_constraint_by_name(f"flow_{node}")
        lhs_expr = ct_old.left_expr
        rhs_val = ct_old.right_expr
        mdl_feas.remove_constraint(ct_old)
        new_ct = mdl_feas.add_constraint(lhs_expr + s_pos - s_neg == rhs_val, f"flow_{node}")
        feas_ct_flow[node] = new_ct

    mdl_feas.minimize(mdl_feas.sum(s[0] + s[1] for s in feas_slacks.values())) # minimiza as slacks

    # configuração dos callbacks
    cut_generator = BendersCutGenerator(
        mdl_worker, mdl_feas, ct_cap, ct_flow, feas_ct_cap, feas_ct_flow, 
        y, eta, arcs_data, node_balances
    )

    cb_lazy = mdl_master.register_callback(BendersLazyCallback)
    cb_lazy.set_generator(cut_generator)

    cb_user = mdl_master.register_callback(BendersUserCallback)
    cb_user.set_generator(cut_generator)
    
    # parâmetros CPLEX
    mdl_master.parameters.preprocessing.presolve = 0
    mdl_master.parameters.mip.limits.cutpasses = 10
    mdl_master.parameters.mip.cuts.cliques = -1
    mdl_master.parameters.mip.cuts.covers = -1
    mdl_master.parameters.mip.cuts.flowcovers = -1
    mdl_master.parameters.mip.cuts.gomory = -1
    mdl_master.parameters.mip.cuts.gubcovers = -1
    mdl_master.parameters.mip.cuts.implied = -1
    mdl_master.parameters.mip.cuts.liftproj = -1
    mdl_master.parameters.mip.cuts.localimplied = -1
    mdl_master.parameters.mip.cuts.mcfcut = -1
    mdl_master.parameters.mip.cuts.mircut = -1
    mdl_master.parameters.mip.cuts.pathcut = -1
    mdl_master.parameters.mip.cuts.zerohalfcut = -1
    #mdl_master.parameters.mip.strategy.heuristicfreq = -1
    mdl_master.parameters.timelimit = 1200
    
    start_time = time.time()
    sol = mdl_master.solve(log_output=True)
    end_time = time.time()
    
    if sol:
        print(f"\nCusto Total:....................... {sol.objective_value}")
        print(f"Tempo:............................. {end_time - start_time:.4f} s")
        print(f"Cortes de Otimalidade:............. {cut_generator.num_cuts_optimality}")
        print(f"Cortes de Viabilidade:............. {cut_generator.num_cuts_feasibility}")
    else:
        print("Sem solução.")

# EXECUÇÃO

filename = "instances/instancia_frcf_5.txt"
print(f"Instância: {filename}")
params, node_balances, arcs_data = read_instance(filename)
nodes = list(node_balances.keys())
solve_benders(params, node_balances, arcs_data, nodes)