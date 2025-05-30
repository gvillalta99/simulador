import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import networkx as nx

# --- Configurações e Dados Iniciais ---
N_YEARS = 10

initial_states_data = {
    0: {"Nome": "Técnico trabalhando", "Categoria": "Entrada", "Renda": 1500},
    1: {"Nome": "Faculdade de computação (sem trabalho)", "Categoria": "Entrada", "Renda": 0},
    2: {"Nome": "Faculdade comp. (trabalhando na área)", "Categoria": "Entrada", "Renda": 3500},
    3: {"Nome": "Faculdade comp. (fora da área)", "Categoria": "Entrada", "Renda": 2200},
    4: {"Nome": "Outra faculdade (trabalhando)", "Categoria": "Entrada", "Renda": 2000},
    5: {"Nome": "Empreendendo (baixo capital)", "Categoria": "Entrada", "Renda": 1200},
    6: {"Nome": "Não estuda nem trabalha", "Categoria": "Entrada", "Renda": 0},
    7: {"Nome": "Desempregado", "Categoria": "Entrada", "Renda": 0},
    8: {"Nome": "Pequena empresa na área", "Categoria": "Progresso", "Renda": 2800},
    9: {"Nome": "Startup na área", "Categoria": "Progresso", "Renda": 4000},
    10: {"Nome": "Grande empresa na área", "Categoria": "Progresso", "Renda": 6500},
    11: {"Nome": "Empresa global na área", "Categoria": "Progresso", "Renda": 9000},
    12: {"Nome": "Serviço público municipal", "Categoria": "Progresso", "Renda": 2500},
    13: {"Nome": "Serviço público estadual", "Categoria": "Progresso", "Renda": 3500},
    14: {"Nome": "Serviço público federal", "Categoria": "Progresso", "Renda": 5000},
    15: {"Nome": "Sucesso elevado", "Categoria": "Sucesso extremo", "Renda": 20000},
    16: {"Nome": "Trabalhar em outra área (não TI)", "Categoria": "Progresso", "Renda": 2300}
}
N_INITIAL_STATES = len(initial_states_data)

trajectories_options = {
    "Técnico e não faz faculdade": 0,
    "Faculdade de computação + trabalha na área": 2,
    "Faculdade de computação + não trabalha na área": 1,
    "Empreender (baixo capital)": 5,
    "Faculdade outra área + trabalha": 4,
    "Não estuda nem trabalha": 6
}

if 'editable_salaries' not in st.session_state:
    st.session_state.editable_salaries = {
        state_id: data["Renda"] for state_id, data in initial_states_data.items()
    }

# --- Funções do Modelo de Markov ---
@st.cache_data
def get_base_transition_matrix(trajectory_name, n_total_states):
    P = np.zeros((n_total_states, n_total_states))

    # Definições base de transição (probabilidades ilustrativas)
    if trajectory_name == "Técnico e não faz faculdade":
        P[0, 0] = 0.5; P[0, 8] = 0.1; P[0, 10] = 0.05; P[0, 7] = 0.1; P[0, 6] = 0.05; P[0, 5] = 0.05; P[0, 12] = 0.05; P[0, 16] = 0.1
    elif trajectory_name == "Faculdade de computação + trabalha na área":
        P[2, 2] = 0.5; P[2, 10] = 0.2; P[2, 9] = 0.1; P[2, 11] = 0.05; P[2, 7] = 0.05
        P[1, 1] = 0.3; P[1, 2] = 0.4; P[1, 3] = 0.2
    elif trajectory_name == "Faculdade de computação + não trabalha na área":
        P[1, 1] = 0.4; P[1, 2] = 0.2; P[1, 3] = 0.2; P[1, 16]= 0.1; P[1, 7] = 0.1
        P[3, 3] = 0.4; P[3, 16] = 0.2; P[3, 2] = 0.1; P[3, 7] = 0.1
    elif trajectory_name == "Empreender (baixo capital)":
        P[5, 5] = 0.4; P[5, 8] = 0.2; P[5, 9] = 0.1; P[5, 7] = 0.2; P[5, 6] = 0.05
    elif trajectory_name == "Faculdade outra área + trabalha":
        P[4, 4] = 0.5; P[4, 16] = 0.3; P[4, 7] = 0.1; P[4, 12]= 0.05; P[4, 13]= 0.05
    
    P[6, 6] = 0.6; P[6, 7] = 0.1; P[6, 0] = 0.05; P[6, 5] = 0.05; P[6, 16] = 0.1
    P[7, 7] = 0.5; P[7, 0] = 0.1; P[7, 6] = 0.1; P[7, 5] = 0.1; P[7, 16] = 0.2
    P[16, 16] = 0.6; P[16, 7] = 0.15; P[16, 6] = 0.05; P[16, 0] = 0.05; P[16, 5] = 0.05; P[16, 12] = 0.05

    # Probabilidades de alcançar "Sucesso Elevado" (Estado 15)
    # Usar P[estado, 15] = max(P[estado, 15], chance_sucesso) para não sobrescrever outras transições para 15 já definidas
    # e para garantir que a probabilidade de sucesso seja pelo menos o valor especificado.
    P[5, 15] = max(P[5, 15] if 5 < n_total_states and 15 < n_total_states else 0, 0.05)
    P[8, 15] = max(P[8, 15] if 8 < n_total_states and 15 < n_total_states else 0, 0.02)
    P[9, 15] = max(P[9, 15] if 9 < n_total_states and 15 < n_total_states else 0, 0.03)
    P[10, 15] = max(P[10, 15] if 10 < n_total_states and 15 < n_total_states else 0, 0.01)
    P[11, 15] = max(P[11, 15] if 11 < n_total_states and 15 < n_total_states else 0, 0.02)
    P[12, 15] = max(P[12, 15] if 12 < n_total_states and 15 < n_total_states else 0, 0.001)
    P[13, 15] = max(P[13, 15] if 13 < n_total_states and 15 < n_total_states else 0, 0.002)
    P[14, 15] = max(P[14, 15] if 14 < n_total_states and 15 < n_total_states else 0, 0.005)
    P[16, 15] = max(P[16, 15] if 16 < n_total_states and 15 < n_total_states else 0, 0.005)


    # Normalização Robusta
    for i in range(n_total_states):
        sum_off_diagonal = np.sum(P[i, [j for j in range(n_total_states) if j != i]])
        
        if sum_off_diagonal >= 1.0: # Se a soma das transições para outros estados já é >= 1
            if sum_off_diagonal > 0 : # Evitar divisão por zero se por acaso sum_off_diagonal for 0 mas >= 1.0 (não deve acontecer)
                for j in range(n_total_states):
                    if i != j:
                        P[i, j] = P[i, j] / sum_off_diagonal # Normaliza as transições para outros estados
            P[i, i] = 0.0 # A probabilidade de ficar no mesmo estado se torna 0
        else: # Se há espaço, a probabilidade de ficar é 1 - soma das outras
            P[i, i] = 1.0 - sum_off_diagonal
        
        # Verificação final de segurança para ponto flutuante (raramente necessário com a lógica acima)
        final_row_sum = np.sum(P[i, :])
        if not np.isclose(final_row_sum, 1.0):
            if final_row_sum > 0:
                P[i, :] = P[i, :] / final_row_sum
            else: # Caso extremo: linha toda zero (não deveria acontecer)
                P[i,i] = 1.0
    return P

@st.cache_data
def run_simulation(initial_state_idx, base_P, current_states_df, n_simulations_run, n_years_run, n_total_states_run):
    all_paths = np.zeros((n_simulations_run, n_years_run + 1), dtype=int)
    all_incomes = np.zeros((n_simulations_run, n_years_run + 1))

    for sim in range(n_simulations_run):
        current_state = initial_state_idx
        all_paths[sim, 0] = current_state
        all_incomes[sim, 0] = current_states_df.loc[current_state, "Renda"]

        for year in range(n_years_run):
            probabilities = base_P[current_state, :]
            # Adicionada verificação mais explícita para soma de probabilidades
            if not np.isclose(np.sum(probabilities), 1.0):
                # st.warning(f"Probabilidades para estado {current_state} não somam 1: {np.sum(probabilities)}. Normalizando...")
                if np.sum(probabilities) <= 0 : # Se a soma for zero ou negativa, não há para onde ir. Fica no estado.
                    probabilities = np.zeros(n_total_states_run)
                    probabilities[current_state] = 1.0
                else:
                    probabilities = probabilities / np.sum(probabilities)
                
            next_state = np.random.choice(n_total_states_run, p=probabilities)
            current_state = next_state
            all_paths[sim, year + 1] = current_state
            all_incomes[sim, year + 1] = current_states_df.loc[current_state, "Renda"]
            
    return all_paths, all_incomes

# --- Funções de Visualização (plot_expected_income, etc. permanecem as mesmas) ---
def plot_expected_income(all_incomes_list, trajectory_names_list, ax=None, title_suffix=""):
    # Modificado para aceitar uma lista de incomes e nomes
    create_new_fig = ax is None
    if create_new_fig:
        fig, ax = plt.subplots(figsize=(10, 6))

    for idx, all_incomes in enumerate(all_incomes_list):
        trajectory_name = trajectory_names_list[idx]
        expected_income_per_year = np.mean(all_incomes, axis=0)
        ax.plot(range(len(expected_income_per_year)), expected_income_per_year, marker='o', linestyle='-', label=trajectory_name)
    
    ax.set_xlabel("Ano")
    ax.set_ylabel("Renda Média Esperada (R$)")
    ax.set_title(f"Curva de Renda Esperada{title_suffix}")
    ax.grid(True); ax.legend()
    
    if create_new_fig:
        st.pyplot(fig)
    return ax


def plot_final_state_distribution(all_paths_list, trajectory_names_list, current_states_df, title_suffix=""):
    # Modificado para plotar comparação se mais de uma trajetória for fornecida
    if len(all_paths_list) == 1: # Plot individual
        all_paths = all_paths_list[0]
        trajectory_name = trajectory_names_list[0]
        final_states = all_paths[:, -1]
        state_counts = pd.Series(final_states).value_counts(normalize=True).sort_index()
        valid_indices = [idx for idx in state_counts.index if idx in current_states_df.index]
        state_labels = current_states_df.loc[valid_indices, "Nome"]
        state_counts_filtered = state_counts.loc[valid_indices]

        fig, ax_new = plt.subplots(figsize=(12, 7))
        state_counts_filtered.plot(kind='bar', ax=ax_new)
        ax_new.set_xticklabels(state_labels, rotation=45, ha="right")
        ax_new.set_xlabel("Estado Final"); ax_new.set_ylabel("Proporção de Agentes")
        ax_new.set_title(f"Distribuição Final de Estados: {trajectory_name}{title_suffix}")
        plt.tight_layout(); st.pyplot(fig)
    else: # Plot de comparação
        counts_list = []
        for i, all_paths in enumerate(all_paths_list):
            final_states = all_paths[:, -1]
            counts = pd.Series(final_states).value_counts(normalize=True).rename(trajectory_names_list[i])
            counts_list.append(counts)
        
        df_compare_final = pd.concat(counts_list, axis=1).fillna(0)
        all_possible_states_idx_comp = current_states_df.index
        df_compare_final = df_compare_final.reindex(all_possible_states_idx_comp, fill_value=0)
        df_compare_final = df_compare_final[(df_compare_final.T != 0).any()] # Remover onde todos são 0
        if df_compare_final.empty:
            st.warning("Nenhum agente terminou em estados rastreáveis para comparação.")
            return
        df_compare_final.index = current_states_df.loc[df_compare_final.index, "Nome"]

        fig_comp_dist, ax_comp_dist = plt.subplots(figsize=(12,8))
        df_compare_final.plot(kind='bar', ax=ax_comp_dist, width=0.8)
        ax_comp_dist.set_xlabel("Estado Final"); ax_comp_dist.set_ylabel("Proporção de Agentes")
        ax_comp_dist.set_title(f"Comparativo: Distribuição Final de Estados{title_suffix}")
        plt.xticks(rotation=60, ha="right"); plt.tight_layout(); st.pyplot(fig_comp_dist)


def plot_final_income_distribution_hist(all_incomes_list, trajectory_names_list, title_suffix=""):
    fig, ax = plt.subplots(figsize=(10, 6))
    for i, all_incomes in enumerate(all_incomes_list):
        trajectory_name = trajectory_names_list[i]
        renda_final = all_incomes[:, -1]
        ax.hist(renda_final, bins=30, edgecolor='black', alpha=0.6, density=True, label=f"{trajectory_name} (Hist)")
        
        # Adicionar KDE sobreposto para suavizar
        pd.Series(renda_final).plot(kind='kde', ax=ax, linestyle='--', label=f"{trajectory_name} (KDE)")

        media_renda_final = np.mean(renda_final)
        mediana_renda_final = np.median(renda_final)
        # ax.axvline(media_renda_final, color=plt.cm.get_cmap('tab10')(i), linestyle='dashed', linewidth=1.5, label=f'Média {trajectory_name_short}: R$ {media_renda_final:,.0f}'.replace(",", "."))
        # ax.axvline(mediana_renda_final, color=plt.cm.get_cmap('tab10')(i), linestyle='dotted', linewidth=1.5, label=f'Mediana {trajectory_name_short}: R$ {mediana_renda_final:,.0f}'.replace(",", "."))

    ax.set_xlabel("Renda no Último Ano (R$)")
    ax.set_ylabel("Densidade de Probabilidade")
    ax.set_title(f"Distribuição de Renda Final{title_suffix}")
    ax.grid(axis='y', alpha=0.75)
    ax.legend()
    st.pyplot(fig)
    # Display mean/median separately for clarity if many lines
    for i, all_incomes in enumerate(all_incomes_list):
        trajectory_name_short = trajectory_names_list[i].split(" ")[0][:10]
        media_renda_final = np.mean(all_incomes[:, -1])
        mediana_renda_final = np.median(all_incomes[:, -1])
        st.caption(f"{trajectory_name_short} - Média: R$ {media_renda_final:,.0f}, Mediana: R$ {mediana_renda_final:,.0f}".replace(",", "."))

def display_sample_paths(all_paths, current_states_df, n_years_run): # Mesma de antes
    st.subheader("Exemplos de Caminhos Individuais (Primeiros 20)")
    sample_df = pd.DataFrame(all_paths[:20, :]).applymap(lambda x: f"{x}: {current_states_df.loc[x, 'Nome']}")
    sample_df.columns = [f"Ano {i}" for i in range(n_years_run + 1)]
    st.dataframe(sample_df)

def plot_transition_graph_mpl(P_matrix, trajectory_name, current_states_df, n_total_states_run): # Mesma de antes
    st.subheader(f"Grafo de Transição (Probabilidades > 0.05): {trajectory_name}")
    G = nx.DiGraph(); node_labels = {}; edge_labels = {}
    nodes_in_graph = set()

    for i in range(n_total_states_run):
        is_significant_self_loop_island = (P_matrix[i,i] > 0.05 and not np.any(P_matrix[i, np.arange(n_total_states_run) != i] > 0.05) and not np.any(P_matrix[np.arange(n_total_states_run) != i, i] > 0.05))
        if np.any(P_matrix[i, :] > 0.05) or np.any(P_matrix[:, i] > 0.05) or is_significant_self_loop_island:
             nodes_in_graph.add(i)
    
    if not nodes_in_graph:
        st.warning("Nenhuma transição significativa (>5%) para exibir no grafo para esta trajetória.")
        return

    for i in nodes_in_graph: # Apenas nós que estão no grafo
        if i not in current_states_df.index: continue # Segurança
        G.add_node(i); node_labels[i] = f"{i}: {current_states_df.loc[i, 'Nome'][:20]}..."
        for j in nodes_in_graph: # Apenas nós que estão no grafo
            if P_matrix[i, j] > 0.05:
                G.add_edge(i, j, weight=P_matrix[i, j])
                edge_labels[(i,j)] = f"{P_matrix[i, j]:.2f}"
    
    if not G.nodes() or not G.edges(): # Adicionado cheque de arestas
        st.warning("Nenhuma transição significativa (>5%) para exibir no grafo para esta trajetória (após filtro de arestas).")
        return

    fig, ax = plt.subplots(figsize=(18, 18))
    try: pos = nx.kamada_kawai_layout(G)
    except Exception: pos = nx.spring_layout(G, k=1.5/np.sqrt(len(G.nodes())) if len(G.nodes()) > 0 else 1, iterations=30)
    
    nx.draw_networkx_nodes(G, pos, ax=ax, node_size=3500, node_color="skyblue", alpha=0.9)
    nx.draw_networkx_edges(G, pos, ax=ax, edgelist=G.edges(), arrowstyle='-|>', arrowsize=20, edge_color="gray", alpha=0.7, node_size=3500)
    nx.draw_networkx_labels(G, pos, labels=node_labels, ax=ax, font_size=9)
    nx.draw_networkx_edge_labels(G, pos, edge_labels=edge_labels, ax=ax, font_color='darkred', font_size=8)
    ax.set_title(f"Grafo de Transição (Matplotlib): {trajectory_name}", fontsize=15); plt.axis('off'); st.pyplot(fig)


def display_renda_boa_metrics(all_paths_data, all_incomes_data, trajectory_name_data, current_states_df_data, renda_boa_thresh_data, n_simulations_data, title_prefix=""):
    st.subheader(f"{title_prefix}Análise de 'Renda Boa' para: {trajectory_name_data}")
    
    renda_final_agentes = all_incomes_data[:, -1]
    chance_renda_boa_limiar = np.sum(renda_final_agentes >= renda_boa_thresh_data) / n_simulations_data
    st.metric(
        label=f"Chance de Renda Final ≥ R$ {renda_boa_thresh_data:,.0f}".replace(",", "."), 
        value=f"{chance_renda_boa_limiar:.2%}"
    )
    st.markdown("---")

    final_states = all_paths_data[:, -1]
    # Estado 15 = Sucesso Elevado
    if 15 in current_states_df_data.index:
        success_freq = np.sum(final_states == 15) / n_simulations_data
        renda_sucesso_elevado = current_states_df_data.loc[15, "Renda"]
        st.metric(
            label=f"Chance de Terminar em 'Sucesso Elevado' (Renda R$ {renda_sucesso_elevado:,})".replace(",", "."),
            value=f"{success_freq:.2%}"
        )
    st.markdown("---")

    estados_alta_renda_idx = [idx for idx in [10, 11, 14] if idx in current_states_df_data.index]
    if estados_alta_renda_idx:
        nomes_estados_alta_renda = current_states_df_data.loc[estados_alta_renda_idx, "Nome"].tolist()
        chance_estados_alta_renda = np.sum(np.isin(final_states, estados_alta_renda_idx)) / n_simulations_data
        st.write(f"**Chance de Terminar em Outros Estados de Alta Renda Selecionados:**")
        st.write(f"*Considerando: {'; '.join(nomes_estados_alta_renda)}*")
        st.metric(label="Probabilidade", value=f"{chance_estados_alta_renda:.2%}")


# --- Interface Streamlit ---
st.set_page_config(layout="wide")
st.title("🧠 Simulador de Caminhos Profissionais Flexível")
st.markdown("Configure salários, escolha trajetórias e compare os resultados simulados!")

# --- Sidebar para Controles ---
st.sidebar.header("Configurações da Simulação")
selected_trajectory_name_1 = st.sidebar.selectbox(
    "Escolha a trajetória inicial (Trajetória 1):",
    list(trajectories_options.keys()),
    key="traj1"
)
renda_boa_threshold = st.sidebar.number_input(
    "Defina o limiar para 'Renda Boa' (R$):", 
    min_value=0, value=8000, step=500, format="%d"
)
n_simul_input = st.sidebar.number_input(
    "Número de Simulações (Agentes):",
    min_value=100, max_value=10000, value=1000, step=100,
    help="Quanto maior, mais preciso o resultado, porém mais lento."
)

st.sidebar.markdown("---")
with st.sidebar.expander("🔧 Configurar Salários dos Estados", expanded=False):
    temp_salaries = {} # Usar um dict temporário para coletar inputs antes de aplicar ao session_state
    for state_id, default_data in initial_states_data.items():
        # Usar st.session_state.editable_salaries.get() para o valor, para persistir entre edições parciais
        temp_salaries[state_id] = st.number_input(
            f"Salário Estado {state_id}: {default_data['Nome']}",
            value=st.session_state.editable_salaries.get(state_id, default_data["Renda"]),
            min_value=0,
            step=100,
            key=f"salary_input_{state_id}" # Chave única
        )
    if st.button("Aplicar Salários Personalizados", key="apply_salaries_button"):
        st.session_state.editable_salaries = temp_salaries.copy()
        st.success("Salários personalizados aplicados!")
        st.experimental_rerun() # Rerun para garantir que df_states seja atualizado em toda a UI

current_dynamic_df_states = pd.DataFrame.from_dict({
    s_id: {
        "Nome": initial_states_data[s_id]["Nome"],
        "Categoria": initial_states_data[s_id]["Categoria"],
        "Renda": st.session_state.editable_salaries.get(s_id, initial_states_data[s_id]["Renda"])
    } for s_id in initial_states_data
}, orient='index')
CURRENT_N_STATES = len(current_dynamic_df_states)


st.sidebar.header("Comparação de Trajetórias")
compare_mode = st.sidebar.checkbox("Ativar modo de comparação")
selected_trajectory_name_2 = None
if compare_mode:
    available_for_compare = [t for t in trajectories_options.keys() if t != selected_trajectory_name_1]
    if available_for_compare:
        selected_trajectory_name_2 = st.sidebar.selectbox(
            "Escolha a segunda trajetória para comparar (Trajetória 2):",
            available_for_compare, index=0, key="traj2"
        )
    else:
        st.sidebar.warning("Apenas uma trajetória disponível para seleção, não é possível comparar.")
        compare_mode = False

# --- Execução e Exibição ---
if st.sidebar.button("🚀 Rodar Simulação", key="run_sim_button"):
    df_states_for_sim = current_dynamic_df_states.copy()
    
    # Simulação para Trajetória 1
    initial_state_idx_1 = trajectories_options[selected_trajectory_name_1]
    P_base_1 = get_base_transition_matrix(selected_trajectory_name_1, CURRENT_N_STATES)
    
    st.header(f"Resultados para Trajetória 1: {selected_trajectory_name_1}")
    with st.spinner(f"Rodando {n_simul_input} simulações para {selected_trajectory_name_1}..."):
        all_paths_1, all_incomes_1 = run_simulation(
            initial_state_idx_1, P_base_1, df_states_for_sim, n_simul_input, N_YEARS, CURRENT_N_STATES
        )

    tab_titles_1 = ["📈 Renda Média", "📊 Dist. Estados", "SAL Dist. Renda", "👣 Caminhos", "🕸️ Grafo", "🎯 Renda Boa"]
    tabs_1 = st.tabs(tab_titles_1)

    with tabs_1[0]: plot_expected_income([all_incomes_1], [selected_trajectory_name_1])
    with tabs_1[1]: plot_final_state_distribution([all_paths_1], [selected_trajectory_name_1], df_states_for_sim)
    with tabs_1[2]: plot_final_income_distribution_hist([all_incomes_1], [selected_trajectory_name_1])
    with tabs_1[3]: display_sample_paths(all_paths_1, df_states_for_sim, N_YEARS)
    with tabs_1[4]: plot_transition_graph_mpl(P_base_1, selected_trajectory_name_1, df_states_for_sim, CURRENT_N_STATES)
    with tabs_1[5]: display_renda_boa_metrics(all_paths_1, all_incomes_1, selected_trajectory_name_1, df_states_for_sim, renda_boa_threshold, n_simul_input)

    # Simulação e Abas para Comparação (se ativado)
    if compare_mode and selected_trajectory_name_2:
        initial_state_idx_2 = trajectories_options[selected_trajectory_name_2]
        P_base_2 = get_base_transition_matrix(selected_trajectory_name_2, CURRENT_N_STATES)
        
        st.header(f"Comparação: {selected_trajectory_name_1} vs {selected_trajectory_name_2}")
        with st.spinner(f"Rodando {n_simul_input} simulações para {selected_trajectory_name_2}..."):
            all_paths_2, all_incomes_2 = run_simulation(
                initial_state_idx_2, P_base_2, df_states_for_sim, n_simul_input, N_YEARS, CURRENT_N_STATES
            )
        
        comp_tab_titles = ["📈 Rendas Médias", "📊 Estados Finais", "SAL Rendas Finais", "🎯 Renda Boa"]
        comp_tabs = st.tabs(comp_tab_titles)

        with comp_tabs[0]: # Curvas de Renda Média Comparadas
            plot_expected_income(
                [all_incomes_1, all_incomes_2], 
                [selected_trajectory_name_1, selected_trajectory_name_2],
                title_suffix=" (Comparativo)"
            )
        with comp_tabs[1]: # Distribuição de Estados Finais Comparada
            plot_final_state_distribution(
                [all_paths_1, all_paths_2],
                [selected_trajectory_name_1, selected_trajectory_name_2],
                df_states_for_sim,
                title_suffix=" (Comparativo)"
            )
        with comp_tabs[2]: # Distribuição de Renda Final (SAL) Comparada
             plot_final_income_distribution_hist(
                [all_incomes_1, all_incomes_2],
                [selected_trajectory_name_1, selected_trajectory_name_2],
                title_suffix=" (Comparativo)"
            )
        with comp_tabs[3]: # Análise de 'Renda Boa' Comparada
            col1, col2 = st.columns(2)
            with col1:
                display_renda_boa_metrics(all_paths_1, all_incomes_1, selected_trajectory_name_1, df_states_for_sim, renda_boa_threshold, n_simul_input, title_prefix="Traj. 1: ")
            with col2:
                display_renda_boa_metrics(all_paths_2, all_incomes_2, selected_trajectory_name_2, df_states_for_sim, renda_boa_threshold, n_simul_input, title_prefix="Traj. 2: ")
else:
    st.info("Escolha uma trajetória na barra lateral, ajuste as configurações e clique em 'Rodar Simulação'.")

st.sidebar.markdown("---")
st.sidebar.caption("Desenvolvido como exemplo educacional.")