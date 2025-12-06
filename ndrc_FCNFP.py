import sys
import cplex
import numpy as np
import time
from reader import read_instance
from itertools import chain, combinations

# HEURÍSTICA PRIMAL

def initial_heuristic(num_nodes, b_list, arcs_data): # MFC
    try:
        model = cplex.Cplex()
        model.set_log_stream(None)
        model.set_results_stream(None)
        model.objective.set_sense(model.objective.sense.minimize)

        x_names = [f"x_{i}_{j}" for (i, j, c, f, w) in arcs_data]
        x_costs = [c + (f / w) if w > 0 else c for (i, j, c, f, w) in arcs_data]
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
            sol_x = model.solution.get_values()
            total_cost = 0.0
            for k, (i, j, c, f, w) in enumerate(arcs_data):
                flow_val = sol_x[k]
                total_cost += flow_val * c
                if flow_val > 1e-6:
                    total_cost += f
            
            return total_cost
        else:
            return float('inf')
    except cplex.exceptions.CplexError as e:
        return float('inf')
    
# HEURÍSTICA LAGRANGEANA
        
def lagrangean_heuristic(num_nodes, b_list, arcs_data, y_bar, penalty_factor=1.0): # Slope Scaling
    try:
        model = cplex.Cplex()
        model.set_log_stream(None)
        model.set_results_stream(None)
        model.objective.set_sense(model.objective.sense.minimize)

        x_names = [f"x_{i}_{j}" for (i, j, c, f, w) in arcs_data]
        x_caps = [w for (i, j, c, f, w) in arcs_data]

        guided_costs = []
        for (i, j, c, f, w) in arcs_data:
            val_y = y_bar.get(f"y_{i}_{j}", 0.0)
            
            if val_y > 0.5:
                guided_costs.append(c)
            else:
                # penalidade multiplicada pelo fator penalty_factor
                base_penalty = c + (f / w) if w > 1e-6 else c
                guided_costs.append(base_penalty * penalty_factor)

        model.variables.add(obj=guided_costs, ub=x_caps, names=x_names)

        # restrições de fluxo
        constraints = []
        for node_id in range(num_nodes):
            flow_balance = cplex.SparsePair()
            for k, (i, j, c, f, w) in enumerate(arcs_data):
                if i == node_id:
                    flow_balance.ind.append(x_names[k]); flow_balance.val.append(1.0)
                if j == node_id:
                    flow_balance.ind.append(x_names[k]); flow_balance.val.append(-1.0)
            constraints.append(flow_balance)
        model.linear_constraints.add(lin_expr=constraints, senses=['E'] * num_nodes, rhs=b_list)

        model.solve()

        if model.solution.get_status() == model.solution.status.optimal:
            sol_x = model.solution.get_values()
            
            # calcula os custos reais
            real_total_cost = 0.0
            for k, (i, j, c, f, w) in enumerate(arcs_data):
                flow_val = sol_x[k]
                real_total_cost += flow_val * c
                if flow_val > 1e-6:
                    real_total_cost += f
            return real_total_cost
        else:
            return float('inf')

    except cplex.exceptions.CplexError:
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
            obj_costs_x.append(c + lam) # custo c_ij + lambda_ij para x_ij
            obj_costs_y.append(f - lam * w) # custo f_ij - lambda_ij * w_ij para y_ij

        # ajuste dos custos com mu_k (cortes de conectividade)
        const_mu = 0.0

        for (cut_id, mu_k, cut_S, cut_RHS) in active_cuts_pool:
            if mu_k > 1e-9:
                const_mu += mu_k * cut_RHS
                for k, (i, j, c, f, w) in enumerate(arcs_data):
                    if (i not in cut_S) and (j in cut_S):
                        obj_costs_y[k] -= mu_k * w

        # variáveis
        model.variables.add(obj=obj_costs_x, lb=[0.0] * num_arcs, ub=[a[4] for a in arcs_data], names=x_names)

        y_types = [model.variables.type.binary] * num_arcs
        model.variables.add(obj=obj_costs_y, lb=[0.0] * num_arcs, ub=[1.0] * num_arcs, types=y_types, names=y_names)

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
            # print("lrp_obj_value=", lrp_obj_value, " const_mu=", const_mu, " z_lrp_total=", z_lrp_total)

            return z_lrp_total, x_bar, y_bar
        else:
            model.write("debug_lrp_model.lp")
            print("   Modelo gravado em debug_lrp_model.lp para inspeção.")
            return float('inf'), None, None

    except cplex.exceptions.CplexError as e:
        print(f"Erro (Subproblema LRP-CPLEX): {e}")
        return float('inf'), None, None

# PROBLEMA DE SEPARAÇÃO (HEURÍSTICA POR COMPONENTES CONEXOS)

def find_violated_cuts(num_nodes, arcs_data, b_list, y_bar, tol=1e-6):
    # constroi o grafo G' com base nos arcos abertos em y_bar
    # lista de adjacência para o grafo não direcionado
    adj_list = {i: [] for i in range(num_nodes)}
    
    for (i, j, c, f, w) in arcs_data:
        # verifica se o arco (i,j) está aberto em y_bar
        if y_bar.get(f"y_{i}_{j}", 0.0) > 0.5:
            adj_list[i].append(j)
            adj_list[j].append(i) # grafo não direcionado para componentes

    violated = []
    visited = set()

    # encontra todos os componentes conexos
    for node in range(num_nodes):
        if node not in visited:
            # se novo componente encontrado, inicia a busca 
            current_component_S = set()
            queue = [node] # usando BFS
            
            while queue:
                current_node = queue.pop(0)
                if current_node not in visited:
                    visited.add(current_node)
                    current_component_S.add(current_node)
                    
                    for neighbor in adj_list[current_node]:
                        if neighbor not in visited:
                            queue.append(neighbor)
            
            # para cada componente S, verifica se ele viola um dicot cut
            if (not current_component_S) or (len(current_component_S) == num_nodes):
                continue

            # calcula RHS: demanda total dentro do componente S
            RHS_NET_DEMAND = sum(b_list[i] for i in current_component_S)

            if RHS_NET_DEMAND <= tol:
                continue

            # calcula LHS: capacidade de ENTRADA em S em y_bar
            LHS_INFLOW_CAP = 0.0
            for (i, j, c, f, w) in arcs_data:
                if (i not in current_component_S) and (j in current_component_S):
                    LHS_INFLOW_CAP += w * y_bar.get(f"y_{i}_{j}", 0.0)
            
            viol = RHS_NET_DEMAND - LHS_INFLOW_CAP

            if viol > tol:
                violated.append((frozenset(current_component_S), RHS_NET_DEMAND, LHS_INFLOW_CAP, viol))
                #print(f"Corte violado: S={set(current_component_S)}, Viol={viol:.2f} (RHS={RHS:.2f}, LHS={LHS:.2f})")

    # ordena pelos cortes mais violados primeiro
    violated.sort(key=lambda x: -x[3], reverse=True)
    
    # limita o número de cortes para não sobrecarregar o LRP
    max_cuts_to_add = 50 
    return violated[:max_cuts_to_add]

# LOOP PRINCIPAL

def subgradient_method_ndrc(num_nodes, num_arcs, b_list, arcs_data, initial_upper_bound, 
                            max_iter=2000, no_improvement_limit=100):

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
        existing_demand_sets_this_iter = set()
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

        active_arcs_count = sum(1 for val in y_bar.values() if val > 0.5)
        #print(f"Iter {k} o subproblema abriu {active_arcs_count} arcos.")

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
        if k % 5 == 0:
            # diferentes pesos para as penalidades
            factors_to_test = [1.2, 2.5, 5.0] 
            
            # a escolha do peso depende do k (iter)
            factor = factors_to_test[(k // 5) % len(factors_to_test)]
            
            new_ub = lagrangean_heuristic(num_nodes, b_list, arcs_data, y_bar, penalty_factor=factor)
            
            if new_ub < best_primal_ub:
                gap_now = ((new_ub - best_lagrangian_lb) / new_ub) * 100
                print(f"UB caiu para {new_ub:.2f} (Fator: {factor}) | Gap: {gap_now:.2f}%")
                best_primal_ub = new_ub
                
                # se achar uma solução melhor, dá um boost no pi para explorar a região
                pi = max(pi, 1.5) 
                iter_without_lb_improvement = 0

        # subgradiente g_lambda: x - w*y
        g_lambda = np.zeros(num_arcs, dtype=float)
        for idx, (i, j, c, f, w) in enumerate(arcs_data):
            g_lambda[idx] = x_bar.get(f"x_{i}_{j}", 0.0) - w * y_bar.get(f"y_{i}_{j}", 0.0)

        # separação de cortes
        violated_cuts = find_violated_cuts(num_nodes, arcs_data, b_list, y_bar)
        # zera o rastreador de demanda desta iteração
        existing_demand_sets_this_iter = set()
        
        # itera sobre os cortes encontrados
        for (cut_S, RHS, LHS, viol) in violated_cuts:
            
            # verifica se o corte tem demanda e pula se não tiver
            cut_demand_set = frozenset(i for i in cut_S if b_list[i] > 0)
            if not cut_demand_set:
                continue

            # verifica se a demanda já foi adicionada nessa iteração e pula se sim
            if cut_demand_set in existing_demand_sets_this_iter:
                continue

            # verifica se já adicionou o corte S ou seu complemento no processo total
            # impede duplicatas no pool
            complement_S = frozenset(set(range(num_nodes)) - set(cut_S))
            if cut_S in existing_cuts_S or complement_S in existing_cuts_S:
                continue

            # se passou por todos os filtros, o corte é novo e válido
            
            # adiciona ao rastreador global de S
            existing_cuts_S.add(cut_S) 
            # adiciona ao rastreador desta iteração
            existing_demand_sets_this_iter.add(cut_demand_set)
            # adiciona ao pool de cortes para o LRP
            cut_pool[cut_counter] = (cut_S, RHS)
            mu_multipliers[cut_counter] = 0.0
            cut_counter += 1

        # subgradiente g_mu
        g_mu = {}
        for cut_id, (cut_S, cut_RHS) in cut_pool.items():
            cut_LHS_val = sum(w * y_bar.get(f"y_{i}_{j}", 0.0)
                              for (i, j, c, f, w) in arcs_data
                              if (i not in cut_S) and (j in cut_S))
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

        numerator = (best_primal_ub - z_lrp_total)
        if numerator <= 1e-12:
            # sem passo (ou passo muito pequeno) evita passo negativo
            T = 0.0
        else:
            T = pi * numerator / norm_g_total_sq


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
    if cut_pool:
        total_cuts_in_pool = len(cut_pool)
        active_cuts_count = sum(1 for mu_k in mu_multipliers.values() if mu_k > 1e-9)
        print(f"Total de cortes encontrados.............: {total_cuts_in_pool}")
        print(f"Total de cortes ativos no final (mu > 0): {active_cuts_count}")
    else:
        print("Nenhum corte de conectividade foi adicionado ao pool.")
    print(f"Tempo total de execução: {elapsed_time:.4f} segundos.")
    sys.stdout.flush()

    return best_primal_ub, best_lagrangian_lb

# SOLVER

def solver(file):
    start_total_time = time.time()
    params, node_balances_dict, arcs_data = read_instance(file)
    num_arcs = len(arcs_data)
    num_nodes = len(node_balances_dict)
    
    b_list = [node_balances_dict[i] for i in range(num_nodes)]
        
    print(f"Instância carregada: {num_nodes} nós, {num_arcs} arcos.\n")
   
    print("INICIALIZANDO... ")
    initial_ub = initial_heuristic(num_nodes, b_list, arcs_data)
    print(f"Valor da solução heurística (UB Inicial): {initial_ub:.2f}\n")

    if initial_ub == float('inf'):
        print("Heurística inicial infactível. Parando.")
        return

    final_ub, final_lb = subgradient_method_ndrc(
        num_nodes, num_arcs, b_list, arcs_data, 
        initial_upper_bound=initial_ub
    )
    
    print(f"Limite Inferior (Lagrangeano NDRC)......: {final_lb:.2f}")
    print(f"Limite Superior Final (Primal)..........: {final_ub:.2f}")

    gap = 0.0
    if final_ub != float('inf') and abs(final_ub) > 1e-6:
        gap = ((final_ub - final_lb) / final_ub) * 100
    
    print(f"Gap de Otimização Final.................: {gap:.4f}%")
    total_time = time.time() - start_total_time
    print(f"Tempo Total de Execução do Solver.......: {total_time:.4f} segundos")


# EXECUÇÃO

file = 'instances/instancia_frcf_4.txt'
solver(file)