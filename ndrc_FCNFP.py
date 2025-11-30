import cplex
import numpy as np
import time
import sys
from itertools import chain, combinations

# LEITOR DE INSTÂNCIAS

def read_instance(filename):
    params = {}
    node_balances = {}
    arcs_data = []

    try:
        with open(filename, 'r') as f:
            section = None
            for line in f:
                line = line.strip()
                if not line or line.startswith('#'):
                    continue

                if line == "NODE_BALANCES":
                    section = "nodes"
                    continue
                elif line == "ARCS":
                    section = "arcs"
                    continue
                elif line == "END_NODE_BALANCES" or line == "END_ARCS":
                    section = None
                    continue

                if section == "nodes":
                    parts = line.split()
                    node_id = int(parts[0])
                    b_i = int(parts[1])
                    node_balances[node_id] = b_i
                elif section == "arcs":
                    parts = line.split()
                    i = int(parts[0])
                    j = int(parts[1])
                    c_ij = int(parts[2])
                    f_ij = int(parts[3])
                    w_ij = int(parts[4])
                    arcs_data.append((i, j, c_ij, f_ij, w_ij))
                elif section is None and ':' in line:
                    key, value = line.split(':', 1)
                    params[key.strip()] = int(value.strip())

    except FileNotFoundError:
        print(f"Erro: File '{filename}' not found.")
        sys.exit(1)
        
    return params, node_balances, arcs_data

# HEURÍSTICA PRIMAL

def initial_heuristic(num_nodes, b_list, arcs_data):
    print("  Calculando UB inicial (Heurística MCF).")
    try:
        model = cplex.Cplex()
        model.set_log_stream(None)
        model.set_results_stream(None)
        model.objective.set_sense(model.objective.sense.minimize)

        x_names = [f"x_{i}_{j}" for (i, j, c, f, w) in arcs_data]
        x_costs = [c for (i, j, c, f, w) in arcs_data]
        x_caps = [w for (i, j, c, f, w) in arcs_data]
        model.variables.add(obj=x_costs, ub=x_caps, names=x_names)

        constraints = []
        for node_id in range(num_nodes):
            flow_balance = cplex.SparsePair()
            for k, (i, j, c, f, w) in enumerate(arcs_data):
                if i == node_id:
                    flow_balance.ind.append(x_names[k])
                    flow_balance.val.append(1.0)
                if j == node_id:
                    flow_balance.ind.append(x_names[k])
                    flow_balance.val.append(-1.0)
            constraints.append(flow_balance)
        model.linear_constraints.add(lin_expr=constraints, senses=['E'] * num_nodes, rhs=b_list)

        model.solve()

        if model.solution.get_status() == model.solution.status.optimal:
            solution_x = model.solution.get_values()
            variable_cost = model.solution.get_objective_value()
            fixed_cost = 0.0
            for k, (i, j, c, f, w) in enumerate(arcs_data):
                if solution_x[k] > 1e-6:
                    fixed_cost += f
            return variable_cost + fixed_cost
        else:
            return float('inf')
    except cplex.exceptions.CplexError as e:
        return float('inf')
    
# HEURÍSTICA LAGRANGEANA
        
def lagrangean_heuristic(num_nodes, b_list, arcs_data, y_bar):
    open_arcs = []
    open_arcs_names = []
    
    for (i, j, c, f, w) in arcs_data:
         if y_bar.get(f"y_{i}_{j}", 0) > 0.5:
            open_arcs.append((i, j, c, f, w))
            open_arcs_names.append(f"x_{i}_{j}")
    
    if not open_arcs:
        if all(b == 0 for b in b_list): return 0.0
        return float('inf')

    try:
        model = cplex.Cplex()
        model.set_log_stream(None)
        model.set_results_stream(None)
        model.objective.set_sense(model.objective.sense.minimize)

        x_costs = [c for (i, j, c, f, w) in open_arcs]
        x_caps = [w for (i, j, c, f, w) in open_arcs]
        model.variables.add(obj=x_costs, ub=x_caps, names=open_arcs_names)

        constraints = []
        for node_id in range(num_nodes):
            flow_balance = cplex.SparsePair()
            for k, (i, j, c, f, w) in enumerate(open_arcs):
                if i == node_id:
                    flow_balance.ind.append(open_arcs_names[k])
                    flow_balance.val.append(1.0)
                if j == node_id:
                    flow_balance.ind.append(open_arcs_names[k])
                    flow_balance.val.append(-1.0)
            constraints.append(flow_balance)
        model.linear_constraints.add(lin_expr=constraints, senses=['E'] * num_nodes, rhs=b_list)

        model.solve()

        if model.solution.get_status() == model.solution.status.optimal:
            variable_cost = model.solution.get_objective_value()
            fixed_cost = sum(f for (i, j, c, f, w) in open_arcs)
            return variable_cost + fixed_cost
        else:
            return float('inf')
    except cplex.exceptions.CplexError as e:
        return float('inf')

# SUBPROBLEMA LAGRANGEANO

def lagrangian_subproblem(num_nodes, arcs_data, b_list, 
                                lambda_multipliers, active_cuts_pool):

    try:
        model = cplex.Cplex()
        model.set_log_stream(None)
        model.set_results_stream(None)
        model.objective.set_sense(model.objective.sense.minimize)

        num_arcs = len(arcs_data)
        x_names = [f"x_{i}_{j}" for (i, j, _, _, _) in arcs_data]
        y_names = [f"y_{i}_{j}" for (i, j, _, _, _) in arcs_data]

        if len(lambda_multipliers) != num_arcs:
            raise ValueError("O vetor lambda deve ter dimensão igual ao número de arcos (num_arcs).")

        # objetivo
        obj_costs_x = []
        obj_costs_y = []

        for k, (i, j, c, f, w) in enumerate(arcs_data):
            lam = lambda_multipliers[k]
            obj_costs_x.append(c + lam)            # custo c_ij + lambda_ij para x_ij
            obj_costs_y.append(f - lam * w)        # custo f_ij - lambda_ij * w_ij para y_ij

        # ajuste dos custos com mu_k (cortes de conectividade)
        for (cut_id, mu_k, cut_S, cut_RHS) in active_cuts_pool:
            if mu_k > 1e-9:
                for k, (i, j, c, f, w) in enumerate(arcs_data):
                    if (i in cut_S) and (j not in cut_S):
                        obj_costs_y[k] -= mu_k * w

        # variáveis
        model.variables.add(
            obj=obj_costs_x,
            lb=[0.0] * num_arcs,
            ub=[a[4] for a in arcs_data],
            names=x_names
        )

        y_types = [model.variables.type.binary] * num_arcs
        model.variables.add(
            obj=obj_costs_y,
            lb=[0.0] * num_arcs,
            ub=[1.0] * num_arcs,
            types=y_types,
            names=y_names
        )

        # conservação de fluxo
        flow_exprs = []
        flow_rhs = []
        flow_sense = []

        for i in range(num_nodes):
            ind = []
            val = []
            # saída
            for (a, (u, v, _, _, _)) in enumerate(arcs_data):
                if u == i:
                    ind.append(x_names[a])
                    val.append(1.0)
            # entrada
            for (a, (u, v, _, _, _)) in enumerate(arcs_data):
                if v == i:
                    ind.append(x_names[a])
                    val.append(-1.0)
            flow_exprs.append(cplex.SparsePair(ind=ind, val=val))
            flow_rhs.append(b_list[i])
            flow_sense.append('E')  # igualdade

        model.linear_constraints.add(lin_expr=flow_exprs, senses=flow_sense, rhs=flow_rhs)

        model.solve()
        status = model.solution.get_status()
        status_str = model.solution.get_status_string()

        if status in (
            model.solution.status.optimal,
            model.solution.status.MIP_optimal,
            model.solution.status.optimal_tolerance,
            model.solution.status.feasible,
        ):
            lrp_obj_value = model.solution.get_objective_value()

            # constante da parte dual (somente mu)
            const_mu = sum(mu_k * cut_RHS for (_, mu_k, _, cut_RHS) in active_cuts_pool)

            # constante das restrições dualizadas: lambda_ij * 0
            z_lrp_total = lrp_obj_value + const_mu

            sol = model.solution.get_values()
            x_bar = {x_names[k]: sol[k] for k in range(num_arcs)}
            y_bar = {y_names[k]: sol[num_arcs + k] for k in range(num_arcs)}

            return z_lrp_total, x_bar, y_bar
        else:
            model.write("debug_lrp_model.lp")
            print("   Modelo gravado em debug_lrp_model.lp para inspeção.")
            return float('inf'), None, None

    except cplex.exceptions.CplexError as e:
        print(f"Erro (Subproblema LRP-CPLEX): {e}")
        return float('inf'), None, None

# PROBLEMA DE SEPARAÇÃO
# encontra o "Dicot Cut" mais violado.
# min sum(w_ij * y_bar_ij * u_ij) - sum(b_i * v_i)
# s.t. u_ij >= v_i - v_j (para u_ij capturar o corte)
# v_i, u_ij binários

from itertools import combinations

def find_violated_cuts(num_nodes, arcs_data, b_list, y_bar, tol=1e-6, max_cuts=100):
    # força bruta

    violated = []
    yvals = {(i, j): y_bar.get(f"y_{i}_{j}", 0.0) for (i, j, _, _, _) in arcs_data}
    nodes = list(range(num_nodes))
    total_sets = 2 ** num_nodes - 2  # exclui vazio e V

    checked = 0
    for r in range(1, num_nodes):
        for comb in combinations(nodes, r):
            checked += 1
            S = set(comb)

            RHS = sum(max(0, b_list[i]) for i in S)
            LHS = sum(w * yvals[(i, j)] for (i, j, _, _, w) in arcs_data
                      if (i in S and j not in S))
            viol = RHS - LHS

            if checked > 100:
                break

            if viol > tol:
                violated.append((frozenset(S), RHS, LHS, viol))
                if len(violated) >= max_cuts:
                    # print(f"Atingido limite de {max_cuts} cortes.")
                    violated.sort(key=lambda x: -x[3])
                    return violated

    violated.sort(key=lambda x: -x[3])

    #if violated:
    #    print(f"{len(violated)} cortes violados encontrados "
    #          f"(de {total_sets} subconjuntos testados).")
    #else:
    #    print(f"Nenhum corte violado encontrado "
    #          f"(avaliados {total_sets} subconjuntos).")

    return violated

# MÉTODO DO SUBGRADIENTE

def subgradient_method_ndrc(num_nodes, num_arcs, b_list, arcs_data, initial_upper_bound, 
                            max_iter=2000, no_improvement_limit=50):

    import sys
    import time

    start_time = time.time() 

    # multiplicadores para restrições de ativação de arco
    lambda_multipliers = np.zeros(num_arcs)

    # multiplicadores para os cortes
    cut_pool = {}           # cut_id -> (cut_S, cut_RHS)
    mu_multipliers = {}     # cut_id -> mu_k
    cut_counter = 0

    pi = 2.0
    best_primal_ub = initial_upper_bound
    best_lagrangian_lb = -float('inf')
    iter_without_lb_improvement = 0

    print(f"Iniciando Subgradiente NDRC com UB inicial = {best_primal_ub:.2f}  (máx {max_iter} iterações)")
    sys.stdout.flush()

    existing_cuts_S = set()

    for k in range(max_iter):
        active_cuts_for_lrp = []
        for cut_id, mu_k in mu_multipliers.items():
            if mu_k > 1e-9:
                (cut_S, cut_RHS) = cut_pool[cut_id]
                active_cuts_for_lrp.append((cut_id, mu_k, cut_S, cut_RHS))

        # subproblema lagrangeano
        z_lrp_total, x_bar, y_bar = lagrangian_subproblem(
            num_nodes, arcs_data, b_list,
            lambda_multipliers,
            active_cuts_for_lrp
        )

        if x_bar is None:
            print("Parando: LRP inviável.")
            break

        # limite inferior atualizado
        if z_lrp_total > best_lagrangian_lb:
            best_lagrangian_lb = z_lrp_total
            iter_without_lb_improvement = 0
        else:
            iter_without_lb_improvement += 1

        # heurística lagrangeana
        if k % 10 == 0:
            new_ub = lagrangean_heuristic(num_nodes, b_list, arcs_data, y_bar)
            if new_ub < best_primal_ub:
                print(f"Novo Limite Superior (UB) na iteração {k+1}: {new_ub:.2f} (era {best_primal_ub:.2f}).")
                best_primal_ub = new_ub
                pi = max(pi, 1.0)
                iter_without_lb_improvement = 0

        # subgradiente g_lambda: x - w*y
        g_lambda = np.zeros(num_arcs, dtype=float)
        for idx, (i, j, c, f, w) in enumerate(arcs_data):
            g_lambda[idx] = x_bar.get(f"x_{i}_{j}", 0.0) - w * y_bar.get(f"y_{i}_{j}", 0.0)

        # separação de cortes
        violated_cuts = find_violated_cuts(num_nodes, arcs_data, b_list, y_bar)
        new_cuts = []
        for (cut_S, RHS, LHS, viol) in violated_cuts:
            if cut_S not in existing_cuts_S and frozenset(set(range(num_nodes)) - set(cut_S)) not in existing_cuts_S:
                new_cuts.append((cut_S, RHS))
                existing_cuts_S.add(cut_S)
                cut_pool[cut_counter] = (cut_S, RHS)
                mu_multipliers[cut_counter] = 0.0
                #print(f"Adicionado corte id={cut_counter} | viol={viol:.3f} | S={set(cut_S)}")
                cut_counter += 1

        # subgradiente g_mu
        g_mu = {}
        for cut_id, (cut_S, cut_RHS) in cut_pool.items():
            cut_LHS_val = sum(w * y_bar.get(f"y_{i}_{j}", 0.0)
                              for (i, j, c, f, w) in arcs_data
                              if (i in cut_S) and (j not in cut_S))
            g_mu[cut_id] = cut_RHS - cut_LHS_val

        # modificação de g_mu
        g_mu_modified = {}
        norm_g_mu_sq = 0.0
        for cut_id, g_k in g_mu.items():
            is_violated = g_k > 1e-6
            is_active = mu_multipliers.get(cut_id, 0.0) > 1e-9
            if is_violated or is_active:
                g_mu_modified[cut_id] = g_k
                norm_g_mu_sq += g_k**2
            else:
                g_mu_modified[cut_id] = 0.0

        # cálculo do passo
        norm_g_lambda_sq = np.sum(g_lambda**2)
        norm_g_total_sq = norm_g_lambda_sq + norm_g_mu_sq
        if norm_g_total_sq < 1e-9:
            print(f"Subgradiente nulo na iteração {k+1}. Parando.")
            break

        if (best_primal_ub - z_lrp_total) < 1e-6:
            print(f"Gap (UB - Z(lambda)) muito pequeno na iteração {k+1}. Parando.")
            break

        T = pi * (best_primal_ub - z_lrp_total) / norm_g_total_sq

        # atualização dos multiplicadores
        lambda_multipliers = np.maximum(0.0, lambda_multipliers + T * g_lambda)
        for cut_id, g_k_mod in g_mu_modified.items():
            if g_k_mod != 0.0:
                mu_multipliers[cut_id] = max(0.0, mu_multipliers[cut_id] + T * g_k_mod)

        # redução de pi
        if iter_without_lb_improvement >= no_improvement_limit:
            pi /= 2.0
            iter_without_lb_improvement = 0

        # relatório
        gap = 0.0
        if best_primal_ub != float('inf') and abs(best_primal_ub) > 1e-6:
            gap = ((best_primal_ub - best_lagrangian_lb) / best_primal_ub) * 100

        if (k == 0) or ((k + 1) % 50 == 0) or ((k + 1) == max_iter):
            print(f"  | Iter {k+1:4d} | Pi: {pi:4.2f} | LB: {best_lagrangian_lb:10.2f} | "
                  f"UB: {best_primal_ub:10.2f} | Gap: {gap:6.2f}% | Cortes Ativos: {len(active_cuts_for_lrp):3d} |")
            sys.stdout.flush()

        if abs(gap) < 0.01:
            print(f"Gap muito pequeno ({gap:.4f}%). Parando.")
            break

    total_iter = k + 1
    elapsed_time = time.time() - start_time

    print(f"\nMétodo NDRC concluído em {total_iter} iterações.")
    print(f"Tempo total de execução: {elapsed_time:.4f} segundos.")
    sys.stdout.flush()

    return best_primal_ub, best_lagrangian_lb

# SOLVER PRINCIPAL

def solver(file):
    start_total_time = time.time()
    params, node_balances_dict, arcs_data = read_instance(file)
    num_arcs = len(arcs_data)
    num_nodes = len(node_balances_dict)
    
    b_list = [node_balances_dict[i] for i in range(num_nodes)]
        
    print(f"Instância carregada: {num_nodes} nós, {num_arcs} arcos.\n")
   
    print("   > HEURÍSTICA INICIAL (UB) <")
    initial_ub = initial_heuristic(num_nodes, b_list, arcs_data)
    print(f"Valor da solução heurística (UB Inicial): {initial_ub:.2f}\n")

    if initial_ub == float('inf'):
        print("Heurística inicial infactível. Parando.")
        return

    print("   > MÉTODO DO SUBGRADIENTE (NDRC) <")

    final_ub, final_lb = subgradient_method_ndrc(
        num_nodes, num_arcs, b_list, arcs_data, 
        initial_upper_bound=initial_ub
    )
    
    print("\n   > RELATÓRIO FINAL DE EXECUÇÃO <")
    print(f"Limite Inferior (Lagrangeano NDRC).: {final_lb:.2f}")
    print(f"Limite Superior Final (Primal).....: {final_ub:.2f}")

    gap = 0.0
    if final_ub != float('inf') and abs(final_ub) > 1e-6:
        gap = ((final_ub - final_lb) / final_ub) * 100
    
    print(f"Gap de Otimização Final............: {gap:.2f}%\n")
    total_time = time.time() - start_total_time
    print(f"Tempo Total de Execução do Solver..: {total_time:.4f} segundos")


# EXECUÇÃO

file = 'instancia_frcf_1.txt'
solver(file)