"""
Plotting utilities for the Career Path Simulator.

This module contains functions for generating various visualizations
of the simulation results, such as income curves, state distributions,
and transition graphs. It also includes utility functions for displaying
metrics and sample data paths.
"""
import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import networkx as nx

from core.config import N_YEARS # N_YEARS is used in get_sample_paths_df

def plot_expected_income(all_incomes_list: list[np.ndarray], trajectory_names_list: list[str], ax=None, title_suffix: str = ""):
    """
    Plots the expected income per year for one or more trajectories.

    Args:
        all_incomes_list: A list of numpy arrays, where each array contains income paths for a trajectory.
        trajectory_names_list: A list of names for the trajectories.
        ax: Optional matplotlib Axes object to plot on. If None, a new figure and axes are created.
        title_suffix: Optional suffix to add to the plot title.
    """
    create_new_fig = ax is None
    if create_new_fig:
        fig, ax = plt.subplots(figsize=(10, 6))

    for idx, all_incomes in enumerate(all_incomes_list):
        expected_income_per_year = np.mean(all_incomes, axis=0)
        ax.plot(range(len(expected_income_per_year)), expected_income_per_year, marker='o', linestyle='-', label=trajectory_names_list[idx])

    ax.set_xlabel("Ano")
    ax.set_ylabel("Renda Média Esperada (R$)")
    ax.set_title(f"Curva de Renda Esperada{title_suffix}")
    ax.grid(True)
    ax.legend()

    if create_new_fig:
        st.pyplot(fig)
    return ax # Return ax for potential further manipulation if needed

def plot_final_state_distribution(all_paths_list: list[np.ndarray], trajectory_names_list: list[str], current_states_df: pd.DataFrame, title_suffix: str = ""):
    """
    Plots the distribution of final states for one or more trajectories.

    Args:
        all_paths_list: A list of numpy arrays, where each array contains state paths for a trajectory.
        trajectory_names_list: A list of names for the trajectories.
        current_states_df: DataFrame with current state information (Nome, Renda, etc.).
        title_suffix: Optional suffix to add to the plot title.
    """
    if not all_paths_list:
        st.warning("Nenhum dado de caminho fornecido para plotar a distribuição de estados finais.")
        return

    if len(all_paths_list) == 1:
        all_paths = all_paths_list[0]
        trajectory_name = trajectory_names_list[0]
        final_states = all_paths[:, -1]
        state_counts = pd.Series(final_states).value_counts(normalize=True).sort_index()

        # Filter for valid indices present in current_states_df
        valid_indices = [idx for idx in state_counts.index if idx in current_states_df.index]
        state_labels = current_states_df.loc[valid_indices, "Nome"]
        state_counts_filtered = state_counts.loc[valid_indices]

        if state_counts_filtered.empty:
            st.warning(f"Nenhum estado final válido para exibir para: {trajectory_name}{title_suffix}")
            return

        fig, ax_new = plt.subplots(figsize=(12, 7))
        state_counts_filtered.plot(kind='bar', ax=ax_new)
        ax_new.set_xticklabels(state_labels, rotation=45, ha="right")
        ax_new.set_xlabel("Estado Final")
        ax_new.set_ylabel("Proporção de Agentes")
        ax_new.set_title(f"Distribuição Final de Estados: {trajectory_name}{title_suffix}")
        plt.tight_layout()
        st.pyplot(fig)
    else:
        counts_list = []
        for ap, tn in zip(all_paths_list, trajectory_names_list):
            if ap.shape[1] > 0: # Ensure there are paths
                 counts_list.append(pd.Series(ap[:, -1]).value_counts(normalize=True).rename(tn))
            else:
                 counts_list.append(pd.Series(dtype=float).rename(tn)) # Add empty series if no paths

        if not counts_list:
            st.warning("Nenhum dado para comparação de estados finais.")
            return

        df_compare_final = pd.concat(counts_list, axis=1).fillna(0)

        # Ensure all states from current_states_df are present for consistent indexing
        df_compare_final = df_compare_final.reindex(current_states_df.index, fill_value=0)
        df_compare_final = df_compare_final[(df_compare_final.T != 0).any()] # Remove rows that are all zero

        if df_compare_final.empty:
            st.warning("Nenhum dado para comparação de estados finais (após filtro).")
            return

        df_compare_final.index = current_states_df.loc[df_compare_final.index, "Nome"]

        fig_comp_dist, ax_comp_dist = plt.subplots(figsize=(12,8))
        df_compare_final.plot(kind='bar', ax=ax_comp_dist, width=0.8)
        ax_comp_dist.set_xlabel("Estado Final")
        ax_comp_dist.set_ylabel("Proporção de Agentes")
        ax_comp_dist.set_title(f"Comparativo: Distribuição Final de Estados{title_suffix}")
        plt.xticks(rotation=60, ha="right")
        plt.tight_layout()
        st.pyplot(fig_comp_dist)

def plot_final_income_distribution_hist(all_incomes_list: list[np.ndarray], trajectory_names_list: list[str], title_suffix: str = ""):
    """
    Plots the histogram and KDE of final year incomes for one or more trajectories.

    Args:
        all_incomes_list: A list of numpy arrays, each containing income paths.
        trajectory_names_list: A list of names for the trajectories.
        title_suffix: Optional suffix for the plot title.
    """
    if not all_incomes_list:
        st.warning("Nenhum dado de renda para plotar distribuição.")
        return

    fig, ax = plt.subplots(figsize=(10, 6))
    for i, all_incomes in enumerate(all_incomes_list):
        if all_incomes.shape[1] > 0: # Ensure there are income paths
            renda_final = all_incomes[:, -1]
            pd.Series(renda_final).plot(kind='kde', ax=ax, linestyle='--', label=f"{trajectory_names_list[i]} (KDE)")
            ax.hist(renda_final, bins=30, edgecolor='black', alpha=0.5, density=True, label=f"{trajectory_names_list[i]} (Hist)")
        else:
            st.caption(f"Sem dados de renda para {trajectory_names_list[i]}.")

    ax.set_xlabel("Renda no Último Ano (R$)")
    ax.set_ylabel("Densidade")
    ax.set_title(f"Distribuição de Renda Final{title_suffix}")
    ax.grid(axis='y', alpha=0.75)
    ax.legend()
    st.pyplot(fig)

    for i, all_incomes in enumerate(all_incomes_list):
        if all_incomes.shape[1] > 0:
            media_renda_final = np.mean(all_incomes[:, -1])
            mediana_renda_final = np.median(all_incomes[:, -1])
            st.caption(f"{trajectory_names_list[i].split(' ')[0][:10]}... - Média: R$ {media_renda_final:,.0f}, Mediana: R$ {mediana_renda_final:,.0f}".replace(",", "."))

def get_sample_paths_df(all_paths: np.ndarray, current_states_df: pd.DataFrame) -> pd.DataFrame:
    """
    Prepares a DataFrame of sample individual career paths for display.
    Uses N_YEARS from core.config for column naming.

    Args:
        all_paths: Numpy array of shape (n_simulations, n_years + 1) with state IDs.
        current_states_df: DataFrame with state information (Nome).

    Returns:
        A pandas DataFrame with formatted sample paths.
    """
    if all_paths.shape[0] == 0:
        return pd.DataFrame() # Return empty DataFrame if no paths

    sample_df = pd.DataFrame(all_paths[:20, :]).applymap(
        lambda x: f"{x}: {current_states_df.loc[x, 'Nome']}" if x in current_states_df.index else f"{x}: Estado Desconhecido"
    )
    sample_df.columns = [f"Ano {i}" for i in range(N_YEARS + 1)] # N_YEARS from core.config
    return sample_df

def plot_transition_graph_mpl(P_matrix: np.ndarray, trajectory_name: str, current_states_df: pd.DataFrame, n_total_states_run: int):
    """
    Plots a transition graph using Matplotlib and NetworkX.
    Shows transitions with probability > 0.05.

    Args:
        P_matrix: The transition probability matrix (numpy array).
        trajectory_name: Name of the trajectory for the title.
        current_states_df: DataFrame with state information (Nome).
        n_total_states_run: Total number of states used in P_matrix.
    """
    G = nx.DiGraph()
    node_labels = {}
    edge_labels = {}
    nodes_in_graph = set()

    for i in range(n_total_states_run):
        # Condition to include a node:
        # 1. It has any outgoing transition > 0.05
        # 2. It has any incoming transition > 0.05
        # 3. It's an "island" state with self-loop > 0.05 but no other significant transitions
        is_significant_island = (
            P_matrix[i, i] > 0.05 and
            not np.any(P_matrix[i, np.arange(n_total_states_run) != i] > 0.05) and # No other outgoing
            not np.any(P_matrix[np.arange(n_total_states_run) != i, i] > 0.05)    # No other incoming
        )
        if np.any(P_matrix[i, :] > 0.05) or np.any(P_matrix[:, i] > 0.05) or is_significant_island:
            if i in current_states_df.index: # Ensure state ID is valid
                nodes_in_graph.add(i)

    if not nodes_in_graph:
        st.warning(f"Nenhuma transição significativa (> 5%) para exibir no grafo de '{trajectory_name}'.")
        return

    for i in nodes_in_graph:
        G.add_node(i)
        node_labels[i] = f"{i}: {current_states_df.loc[i, 'Nome'][:20]}..."
        for j in nodes_in_graph: # Iterate only over nodes that will be in the graph
            if P_matrix[i, j] > 0.05:
                G.add_edge(i, j, weight=P_matrix[i, j])
                edge_labels[(i,j)] = f"{P_matrix[i, j]:.2f}"

    if not G.nodes() or not G.edges(): # Check if graph is empty after filtering edges
        st.warning(f"Nenhuma aresta de transição significativa (> 5%) para exibir no grafo de '{trajectory_name}'.")
        return

    fig, ax = plt.subplots(figsize=(18, 18))
    try:
        pos = nx.kamada_kawai_layout(G)
    except Exception: # Fallback layout
        pos = nx.spring_layout(G, k=1.5/np.sqrt(len(G.nodes())) if len(G.nodes()) > 0 else 1, iterations=30)

    nx.draw_networkx_nodes(G, pos, ax=ax, node_size=3500, node_color="skyblue", alpha=0.9)
    nx.draw_networkx_edges(G, pos, ax=ax, edgelist=G.edges(), arrowstyle='-|>', arrowsize=20,
                           edge_color="gray", alpha=0.7, node_size=3500)
    nx.draw_networkx_labels(G, pos, labels=node_labels, ax=ax, font_size=9)
    nx.draw_networkx_edge_labels(G, pos, edge_labels=edge_labels, ax=ax,
                                 font_color='darkred', font_size=8)

    ax.set_title(f"Grafo de Transição (Prob > 0.05): {trajectory_name}", fontsize=15)
    plt.axis('off')
    st.pyplot(fig)

def display_renda_boa_metrics(
    all_paths_data: np.ndarray,
    all_incomes_data: np.ndarray,
    trajectory_name_data: str,
    current_states_df_data: pd.DataFrame,
    renda_boa_thresh_data: float,
    n_simulations_data: int,
    title_prefix: str = ""
):
    """
    Displays metrics related to achieving a 'good income' and specific success states.

    Args:
        all_paths_data: Numpy array of state paths.
        all_incomes_data: Numpy array of income paths.
        trajectory_name_data: Name of the trajectory.
        current_states_df_data: DataFrame with state information.
        renda_boa_thresh_data: Threshold for 'good income'.
        n_simulations_data: Total number of simulations run.
        title_prefix: Optional prefix for the subheader.
    """
    st.subheader(f"{title_prefix}Análise de 'Renda Boa' para: {trajectory_name_data}")

    if all_incomes_data.shape[0] == 0 or all_paths_data.shape[0] == 0:
        st.warning("Dados insuficientes para métricas de 'Renda Boa'.")
        return

    renda_final_agentes = all_incomes_data[:, -1]
    chance_renda_boa_limiar = np.sum(renda_final_agentes >= renda_boa_thresh_data) / n_simulations_data
    st.metric(label=f"Chance Renda Final ≥ R$ {renda_boa_thresh_data:,.0f}".replace(",", "."), value=f"{chance_renda_boa_limiar:.2%}")
    st.markdown("---")

    final_states = all_paths_data[:, -1]
    # Estado 15: "Sucesso elevado"
    if 15 in current_states_df_data.index:
        success_freq = np.sum(final_states == 15) / n_simulations_data
        st.metric(label=f"Chance 'Sucesso Elevado' (Renda R$ {current_states_df_data.loc[15, 'Renda']:,})".replace(",", "."), value=f"{success_freq:.2%}")
    else:
        st.caption("Estado 15 'Sucesso Elevado' não definido.")
    st.markdown("---")

    # Estados 10 (Grande empresa), 11 (Empresa global), 14 (Serviço público federal)
    estados_alta_renda_idx = [idx for idx in [10, 11, 14] if idx in current_states_df_data.index]
    if estados_alta_renda_idx:
        nomes_estados_alta_renda = current_states_df_data.loc[estados_alta_renda_idx, "Nome"].tolist()
        chance_estados_alta_renda = np.sum(np.isin(final_states, estados_alta_renda_idx)) / n_simulations_data
        st.write(f"**Chance Outros Estados de Alta Renda:** ({'; '.join(nomes_estados_alta_renda)})")
        st.metric(label="Probabilidade Combinada", value=f"{chance_estados_alta_renda:.2%}")
    else:
        st.caption("Estados de alta renda (10, 11, 14) não definidos ou não encontrados.")
