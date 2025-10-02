"""Submodule handling graph construction and preprocessing."""

import os
from typing import List, Tuple, Set, Optional
from ensmallen.datasets.monarchinitiative import Monarch
from ensmallen import Graph
import pandas as pd
import numpy as np
from tqdm.auto import tqdm


HPO_NODE_TYPE = "biolink:PhenotypicFeature"
PATIENT_NODE_TYPE = "biolink:Case"
EDGE_TYPE = "biolink:has_phenotype"


def load_features(
    path: str, header_path: str, imputed_path: Optional[str], graph: Graph
) -> pd.DataFrame:
    """Load patient nodes features from TSV file."""
    if imputed_path is not None and os.path.exists(imputed_path):
        print("Loading imputed features from file...")
        return pd.read_csv(imputed_path, sep="\t", index_col=None, low_memory=False)

    df = pd.read_csv(
        path,
        sep="\t",
        index_col=None,
        header=None,
        true_values=["Y"],
        false_values=["N"],
        na_values=[".", ""],
        low_memory=False,
    )

    # We convert all boolean columns to floats so to uniformely
    # handle the NaN values in float64 dtype instead of object dtype
    for col in df.columns:
        if df[col].dtype == object and set(df[col].dropna().unique()).issubset(
            {True, False}
        ):
            df[col] = df[col].astype(float)
            continue

    header = pd.read_csv(header_path, sep="\t", index_col=0, header=None)
    df.columns = header.index.tolist()

    if imputed_path is not None:
        imputed_df = impute_features(df, graph)

        # We store the imputed features to a file
        imputed_df.to_csv(imputed_path, sep="\t", index=False)

        return imputed_df

    return df


def load_kg(directed: bool) -> Graph:
    """Load and preprocess the knowledge graph from the Monarch Initiative dataset.

    Returns:
        Graph: The preprocessed knowledge graph.
    """
    monarch: Graph = Monarch(directed=directed, version="2025-09-17")
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
    if not os.path.exists(node_list) or not os.path.exists(edge_list):
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


def impute_feature_column(feature: np.ndarray, graph: Graph) -> np.ndarray:
    """Impute missing features using the graph structure.

    Args:
        feature (np.ndarray): Array containing a single feature column with NaN values.
        graph (Graph): The knowledge graph.

    Returns:
        pd.DataFrame: DataFrame with imputed features.
    """
    # If the feature does not have NaN values or it is not a float array,
    # we return it as is
    if not np.issubdtype(feature.dtype, np.floating) or not np.any(np.isnan(feature)):
        return feature

    # First, we need to create a mapping between the features and the graph nodes
    node_features = np.full((graph.get_number_of_nodes(),), np.nan, dtype=np.float64)
    patient_node_type_id = graph.get_node_type_id_from_node_type_name(PATIENT_NODE_TYPE)
    patient_node_ids = graph.get_node_ids_from_node_type_id(patient_node_type_id)
    patient_node_names = graph.get_node_names_from_node_type_id(patient_node_type_id)
    reverse_map = np.full((feature.shape[0],), -1, dtype=np.int64)
    for patient_node_id, patient_name in zip(patient_node_ids, patient_node_names):
        # We extract the patient ID from the node name
        patient_id = int(patient_name.split(":")[1])
        node_features[patient_node_id] = feature[patient_id]
        reverse_map[patient_id] = patient_node_id

    # Next we execute the graph diffusion
    unchanged = False
    while not unchanged:
        unchanged = True
        new_node_features = node_features.copy()
        for node_id in range(graph.get_number_of_nodes()):
            if np.isnan(node_features[node_id]):
                neighbors = graph.get_neighbour_node_ids_from_node_id(node_id)
                neighbor_features = node_features[neighbors]
                neighbor_features = neighbor_features[~np.isnan(neighbor_features)]
                if len(neighbor_features) > 0:
                    new_value = np.mean(neighbor_features)
                    new_node_features[node_id] = new_value
                    unchanged = False
        node_features = new_node_features

    # Finally, we need to execute the counter-mapping
    imputed_feature = node_features[reverse_map]

    return imputed_feature


def impute_features(df: pd.DataFrame, graph: Graph) -> pd.DataFrame:
    """Impute missing features using the graph structure.

    Args:
        df (pd.DataFrame): DataFrame containing patient features.
        graph (Graph): The knowledge graph.

    Returns:
        pd.DataFrame: DataFrame with imputed features.
    """

    for col in tqdm(df.columns, desc="Imputing features"):
        if df[col].dtype != float or not np.any(np.isnan(df[col])):
            continue
        feature = df[col].to_numpy(dtype=np.float64)
        imputed_feature = impute_feature_column(feature, graph)
        df[col] = imputed_feature

    return df


if __name__ == "__main__":
    kg = load_kg(directed=False)
    # We save the report to a file
    # with open("kg_report.html", "w", encoding="utf-8") as report_file:
    #     report_file.write(str(kg))

    imputed_features = load_features(
        path="THORACIC_DATA.DAT",
        header_path="data_header.tsv",
        imputed_path="imputed_features.tsv",
        graph=kg,
    )
    not_imputed_features = load_features(
        path="THORACIC_DATA.DAT",
        header_path="data_header.tsv",
        imputed_path=None,
        graph=kg,
    )

    total_number_of_values = imputed_features.size
    print(f"Total number of values: {total_number_of_values}")
    number_of_original_nans = not_imputed_features.isna().sum().sum()
    number_of_imputed_nans = imputed_features.isna().sum().sum()
    number_of_imputable_features = sum(
        1
        for col in imputed_features.columns
        if (
            imputed_features[col].dtype == float
            or (
                imputed_features[col].dtype == object
                and set(imputed_features[col].dropna().unique()).issubset({True, False})
            )
        )
    )
    number_of_features = imputed_features.shape[1]
    print(f"Number of imputable (numeric or boolean) features: {number_of_imputable_features} out of {number_of_features}")
    print(
        f"Number of NaN values (original): {number_of_original_nans}, ({round(100 * number_of_original_nans / total_number_of_values, 2)}%)"
    )
    print(
        f"Number of NaN values (imputed): {number_of_imputed_nans}, ({round(100 * number_of_imputed_nans / total_number_of_values, 2)}%)"
    )
