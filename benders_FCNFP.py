import sys
from docplex.mp.model import Model
from collections import defaultdict

# benders automático

# leitura de instância
def read_instance(filename):
    params = {}
    node_balances = {}
    arcs_data = {} 

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
                    arcs_data[(i, j)] = (c_ij, f_ij, w_ij)
                elif section is None and ':' in line:
                    key, value = line.split(':', 1)
                    params[key.strip()] = int(value.strip())
    except FileNotFoundError:
        print(f"Erro: File '{filename}' not found.")
        return {}, {}, {} 
    return params, node_balances, arcs_data

def main(params, NODE_BALANCES, ARC_DATA):       
    mdl = Model(name='FCNFP')
    all_arcs = ARC_DATA.keys()
    all_nodes = NODE_BALANCES.keys()
    NUM_NODES = params.get('NUM_NODES', len(all_nodes))

    # variáveis
    x = mdl.continuous_var_dict(all_arcs, lb=0)
    y = mdl.binary_var_dict(all_arcs)

    # annotations
    for arc, var_x in x.items():
        var_x.set_benders_annotation(1) # variáveis do worker
    for arc, var_y in y.items():
        var_y.set_benders_annotation(0) # variáveis do problema mestre
        
    # dicionários
    C = {arc: ARC_DATA[arc][0] for arc in all_arcs} # custo variável
    F = {arc: ARC_DATA[arc][1] for arc in all_arcs} # custo fixo
    W = {arc: ARC_DATA[arc][2] for arc in all_arcs} # capacidade

    # objetivo
    objective = mdl.sum(C[arc] * x[arc] + F[arc] * y[arc] for arc in all_arcs)
    mdl.minimize(objective)

    # restrições de fluxo
    arc_out = defaultdict(list)
    arc_in = defaultdict(list)

    for i, j in all_arcs:
        arc_out[i].append((i, j))
        arc_in[j].append((i, j))

    for node in all_nodes:
        flow_out = mdl.sum(x[arc] for arc in arc_out[node])
        flow_in = mdl.sum(x[arc] for arc in arc_in[node])
        mdl.add_constraint(flow_out - flow_in == NODE_BALANCES[node], ctname=f'balance_{node}')

    # restrições de capacidade
    for arc in all_arcs:
        mdl.add_constraint(x[arc] <= W[arc] * y[arc], ctname=f'link_cap_{arc[0]}_{arc[1]}')

    # config benders
    # evita que ele tente fazer outra decomposição
    # usa as annotations
    mdl.parameters.benders.strategy.set(3)

    mdl.parameters.timelimit = 1200

    # tentando desativar todos os outros cortes
    mdl.parameters.mip.cuts.cliques = -1
    mdl.parameters.mip.cuts.covers = -1
    mdl.parameters.mip.cuts.flowcovers = -1
    mdl.parameters.mip.cuts.gubcovers = -1
    mdl.parameters.mip.cuts.implied = -1
    mdl.parameters.mip.cuts.liftproj = -1
    mdl.parameters.mip.cuts.localimplied = -1
    mdl.parameters.mip.cuts.mcfcut = -1
    mdl.parameters.mip.cuts.mircut = -1
    mdl.parameters.mip.cuts.pathcut = -1
    mdl.parameters.mip.cuts.zerohalfcut = -1 
    mdl.parameters.mip.cuts.gomory = -1
    mdl.parameters.mip.strategy.heuristicfreq = -1
    
    print(f"Modelo configurado com {NUM_NODES} nós e {len(all_arcs)} arcos.")
    print("Estratégia Benders: Workers.")

    solution = mdl.solve(log_output=True)

    # resultados
    if solution:
        print("\nRESULTADOS")
        print(f"Tempo de Solução (segundos):.............. {mdl.get_solve_details().time:.4f}")
        print(f"Custo Total:.............................. {solution.get_objective_value():.2f}")
        
        #print("\nDetalhes dos Arcos Utilizados (y=1):")
        #for arc in all_arcs:
        #    flow_val = solution.get_value(x[arc])
        #    design_val = solution.get_value(y[arc])
        #    
        #    if design_val > 0.5:
        #        print(f"  Arco {arc[0]}->{arc[1]} | Custo Fixo: {F[arc]} | Fluxo: {flow_val:.2f} / {W[arc]}")
    else:
        print("\nO CPLEX não encontrou solução ótima.")

# executar
instance_file = "instances/instancia_frcf_5.txt"
print(f"Lendo a instância: {instance_file}")
params, NODE_BALANCES, ARC_DATA = read_instance(instance_file)
main(params, NODE_BALANCES, ARC_DATA)