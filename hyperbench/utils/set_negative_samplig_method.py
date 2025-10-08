from negative_sampling.hypergraph_negative_sampling_algorithm import SizedHypergraphNegativeSampler, MotifHypergraphNegativeSampler, CliqueHypergraphNegativeSampler, HypergraphNegativeSampler

def setNegativeSamplingAlgorithm(ns_algorithm: str, num_node: int):
    ns_method : HypergraphNegativeSampler
    match(ns_algorithm):
        case 'SizedHypergraphNegativeSampler':
            ns_method = SizedHypergraphNegativeSampler(num_node)
        case 'MotifHypergraphNegativeSampler': 
            ns_method = MotifHypergraphNegativeSampler(num_node)
        case 'CliqueHypergraphNegativeSampler':
            ns_method = CliqueHypergraphNegativeSampler(num_node)
    
    return ns_method