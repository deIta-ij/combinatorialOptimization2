import sys

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
        print(f"Erro: Arquivo '{filename}' não encontrado.")
        sys.exit(1)
        
    return params, node_balances, arcs_data