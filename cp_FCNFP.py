from docplex.cp.model import CpoModel
from reader import read_instance

# VERSÃO 1
# com restrição lógica SE - ENTÃO

def solve_cp_v1(params, b, arcs_data):
    nodes = list(range(params['NUM_NODES']))
    
    arcs = set()
    c, f, w = {}, {}, {}

    for (i, j, c_ij, f_ij, w_ij) in arcs_data:
        arc = (i, j)
        arcs.add(arc)
        c[arc] = c_ij
        f[arc] = f_ij
        w[arc] = w_ij

    mdl = CpoModel(name='FCNFP_CP_V1')

    # y_ij: variável binária de ativação
    y = {a: mdl.binary_var(name=f'y_{a[0]}_{a[1]}') for a in arcs}
    
    # x_ij: variável INTEIRA de fluxo
    x = {a: mdl.integer_var(min=0, max=w[a], name=f'x_{a[0]}_{a[1]}') for a in arcs}

    # restrições de balanço
    for i in nodes:
        fluxo_saida = mdl.sum(x[i, j] for j in nodes if (i, j) in arcs)
        fluxo_entrada = mdl.sum(x[j, i] for j in nodes if (j, i) in arcs)
        mdl.add(fluxo_saida - fluxo_entrada == b[i])

    # restrições de capacidade e ativação
    # "SE o arco estiver desativado (y[a] == 0), ENTÃO o fluxo deve ser zero (x[a] == 0)"
    # a restrição x[a] <= w[a] quando y[a] == 1 já é garantida
    # pelo domínio da variável x[a].
    for a in arcs:
        mdl.add(mdl.if_then(y[a] == 0, x[a] == 0))

    custo_variavel = mdl.sum(c[a] * x[a] for a in arcs)
    custo_fixo = mdl.sum(f[a] * y[a] for a in arcs)
    
    mdl.add(mdl.minimize(custo_variavel + custo_fixo))
    msol = mdl.solve(log_output=True, TimeLimit=1200) 

    # resultados
    if msol:
        print(f"Status: {msol.get_solve_status()}")
        print(f"Custo Total Mínimo: {msol.get_objective_value():.0f}")
        
        print("\nArcos utilizados e fluxo:")
        total_fluxo = 0
        for a in arcs:
            y_val = msol.get_value(y[a])
            if y_val > 0.5:
                x_val = msol.get_value(x[a])
                total_fluxo += x_val
                print(f"  Arc {a}: Ativado (y=1). Fluxo (x) = {x_val} / Capacidade = {w[a]}")
        
        print(f"\nVolume total de fluxo enviado: {total_fluxo}")
    
    else:
        print("\nNenhuma solução encontrada.")

# VERSÃO 2
# aplicação direta do modelo

def solve_cp_v2(params, b, arcs_data):
    nodes = list(range(params['NUM_NODES']))
    
    arcs = set()
    c, f, w = {}, {}, {}

    for (i, j, c_ij, f_ij, w_ij) in arcs_data:
        arc = (i, j)
        arcs.add(arc)
        c[arc] = c_ij
        f[arc] = f_ij
        w[arc] = w_ij

    mdl = CpoModel(name='FCNFP_CP_V2')

    # y_ij: variável binária de ativação
    y = {a: mdl.binary_var(name=f'y_{a[0]}_{a[1]}') for a in arcs}
    
    # x_ij: variável INTEIRA de fluxo
    x = {a: mdl.integer_var(name=f'x_{a[0]}_{a[1]}') for a in arcs}

    # restrições de balanço
    for i in nodes:
        fluxo_saida = mdl.sum(x[i, j] for j in nodes if (i, j) in arcs)
        fluxo_entrada = mdl.sum(x[j, i] for j in nodes if (j, i) in arcs)
        
        mdl.add(fluxo_saida - fluxo_entrada == b[i])

    # restrições de capacidade e ativação
    for a in arcs:
        mdl.add(x[a] <= w[a] * y[a])

    custo_variavel = mdl.sum(c[a] * x[a] for a in arcs)
    custo_fixo = mdl.sum(f[a] * y[a] for a in arcs)
    
    mdl.add(mdl.minimize(custo_variavel + custo_fixo))
    msol = mdl.solve(log_output=True, TimeLimit=1200) 

    # resultados
    if msol:
        print(f"Status: {msol.get_solve_status()}")
        print(f"Custo Total Mínimo (Função Objetivo): {msol.get_objective_value():.0f}")
        
        print("\nArcos utilizados e Fluxo:")
        total_fluxo = 0
        for a in arcs:
            y_val = msol.get_value(y[a])
            if y_val > 0.5:
                x_val = msol.get_value(x[a])
                total_fluxo += x_val
                print(f"  Arc {a}: Ativado (y=1). Fluxo (x) = {x_val} / Capacidade = {w[a]}")
        
        print(f"\nVolume total de fluxo enviado: {total_fluxo}")
    
    else:
        print("\nNenhuma solução encontrada.")

# EXECUÇÃO

instance_file = "instances\instancia_frcf_1.txt" 
params, balances, arc_data = read_instance(instance_file)
print(f"Instância carregada: {params['NUM_NODES']} nós, {params['NUM_ARCS']} arcos.")
solve_cp_v1(params, balances, arc_data)