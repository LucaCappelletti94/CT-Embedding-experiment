"""Submodule handling graph construction and preprocessing."""

from typing import List, Tuple, Set
from ensmallen.datasets.monarchinitiative import Monarch
from ensmallen import Graph


HPO_NODE_TYPE = "biolink:PhenotypicFeature"
PATIENT_NODE_TYPE = "biolink:Case"
EDGE_TYPE = "biolink:has_phenotype"


def load_kg(directed: bool) -> Graph:
    """Load and preprocess the knowledge graph from the Monarch Initiative dataset.

    Returns:
        Graph: The preprocessed knowledge graph.
    """
    monarch: Graph = Monarch(directed=directed)
    # We preprocess the PT-HPO mapping file, if needed
    pt2hpo = parse_pt_hpo(
        path="pt_hpo_terms.tsv",
        node_list="pt_hpo_nodes.tsv",
        edge_list="pt_hpo_edges.tsv",
        monarch=monarch,
        directed=directed,
    )

    # We combine the two graphs
    combined = monarch | pt2hpo

    if not directed:
        # We retrieve the number of patient nodes
        num_patients = pt2hpo.get_node_type_names_counts_hashmap()[PATIENT_NODE_TYPE]
        combined = combined.remove_components(top_k_components=1)
        num_patients_after = combined.get_node_type_names_counts_hashmap().get(
            PATIENT_NODE_TYPE, 0
        )
        print(f"Number of patient nodes before removing components: {num_patients}")
        print(
            f"Number of patient nodes after removing components: {num_patients_after}"
        )

    return combined


def parse_pt_hpo(
    path: str,
    node_list: str,
    edge_list: str,
    monarch: Graph,
    directed: bool = True,
) -> Graph:
    """Parses the patient to HPO term mapping file.

    Args:
        path (str): Path to the mapping file.
        node_list (str): Path to the output node list file.
        edge_list (str): Path to the output edge list file.
        monarch (Graph): The Monarch knowledge graph.
        directed (bool): Whether the graph is directed.

    Returns:
        None
    """
    edges: List[Tuple[int, str]] = []
    patient_nodes: Set[int] = set()
    hpo_nodes: Set[str] = set()

    with open(path, "r", encoding="utf-8") as file:
        for line in file:
            parts = line.strip().split("\t")
            if len(parts) < 2:
                continue
            patient_id = int(parts[0])
            hpo_terms = parts[1].split(" ")
            patient_nodes.add(patient_id)
            for hpo_term in hpo_terms:
                if hpo_term == "":
                    continue
                # This term has been determined to not be
                # associated to the patient
                if hpo_term.startswith("!"):
                    continue
                assert monarch.has_node_name(
                    hpo_term
                ), f"HPO term {hpo_term} not in graph"
                hpo_nodes.add(hpo_term)
                edges.append((patient_id, hpo_term))

    # First we printout the node list,
    # with node,node_type as a header
    with open(node_list, "w", encoding="utf-8") as node_file:
        node_file.write("node\tnode_type\n")
        for patient in sorted(patient_nodes):
            node_file.write(f"Patient:{patient}\t{PATIENT_NODE_TYPE}\n")
        for hpo in sorted(hpo_nodes):
            node_file.write(f"{hpo}\t{HPO_NODE_TYPE}\n")

    # Then we printout the edge list,
    # with source,target,edge_type as a header
    with open(edge_list, "w", encoding="utf-8") as edge_file:
        edge_file.write("source\ttarget\tedge_type\n")
        for source, target in edges:
            edge_file.write(f"Patient:{source}\t{target}\t{EDGE_TYPE}\n")

    return Graph.from_csv(
        directed=directed,
        node_path=node_list,
        node_list_separator="\t",
        node_list_header=True,
        nodes_column="node",
        node_list_node_types_column="node_type",
        edge_path=edge_list,
        edge_list_separator="\t",
        edge_list_header=True,
        sources_column="source",
        destinations_column="target",
        edge_list_edge_types_column="edge_type",
        verbose=True,
        name="PT2HPO",
    )


if __name__ == "__main__":
    kg = load_kg(directed=False)
    report = str(kg)

    # We save the report to a file
    with open("kg_report.html", "w", encoding="utf-8") as report_file:
        report_file.write(report)
