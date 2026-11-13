import sumolib


def build_neighbor_map(net_file):
    net = sumolib.net.readNet(net_file)

    tls_list = list(net.getTrafficLights())
    neighbors = {tls.getID(): set() for tls in tls_list}

    # Map each edge ID to the traffic lights that control it
    edge_to_tls = {}
    for tls in tls_list:
        tid = tls.getID()
        for edge in tls.getEdges():
            if edge is None:
                continue
            eid = edge.getID()
            if eid.startswith(":"):   # skip internal junction edges
                continue
            edge_to_tls.setdefault(eid, set()).add(tid)

    # For each TLS, look at its controlled edges and find nearby TLSs
    for tls in tls_list:
        tid = tls.getID()

        for edge in tls.getEdges():
            if edge is None:
                continue
            if edge.getID().startswith(":"):
                continue

            # check connected incoming/outgoing edges through the edge's nodes
            connected_edges = []

            from_node = edge.getFromNode()
            to_node = edge.getToNode()

            connected_edges.extend(from_node.getIncoming())
            connected_edges.extend(from_node.getOutgoing())
            connected_edges.extend(to_node.getIncoming())
            connected_edges.extend(to_node.getOutgoing())

            for other_edge in connected_edges:
                if other_edge is None:
                    continue
                other_id = other_edge.getID()
                if other_id.startswith(":"):
                    continue

                for other_tls in edge_to_tls.get(other_id, set()):
                    if other_tls != tid:
                        neighbors[tid].add(other_tls)

    return {k: sorted(v) for k, v in neighbors.items()}