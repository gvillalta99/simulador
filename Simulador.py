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
N_INITIAL_STATES = len(initial_states_data) # Deve ser 17

# Nomes dos estados para rótulos de matriz
STATE_LABELS = [f"{id}: {data['Nome'][:20]}..." for id, data in initial_states_data.items()]


trajectories_options = {
    "Técnico e não faz faculdade": 0,
    "Faculdade de computação + trabalha na área": 2,
    "Faculdade de computação + não trabalha na área": 1,
    "Empreender (baixo capital)": 5,
    "Faculdade outra área + trabalha": 4,
    "Não estuda nem trabalha": 6
}

# --- Inicialização do Session State ---
if 'editable_salaries' not in st.session_state:
    st.session_state.editable_salaries = {
        state_id: data["Renda"] for state_id, data in initial_states_data.items()
    }
if 'custom_transition_matrices' not in st.session_state:
    st.session_state.custom_transition_matrices = {} # Armazenará {trajectory_name: np.array}

# --- Funções do Modelo de Markov ---
def get_default_base_transition_matrix(trajectory_name, n_total_states):
    """Gera a matriz de transição PADRÃO baseada na lógica original."""
    P = np.zeros((n_total_states, n_total_states))

    # Definições base de transição (probabilidades ilustrativas)
    if trajectory_name == "Técnico e não faz faculdade":
        P[0, 0] = 0.5; P[0, 8] = 0.1; P[0, 10] = 0.05; P[0, 7] = 0.1; P[0, 6] = 0.05; P[0, 5] = 0.05; P[0, 12] = 0.05; P[0, 16] = 0.1
    elif trajectory_name == "Faculdade de computação + trabalha na área": # Estado inicial 2
        P[2, 2] = 0.5; P[2, 10] = 0.2; P[2, 9] = 0.1; P[2, 11] = 0.05; P[2, 7] = 0.05
        P[1, 1] = 0.3; P[1, 2] = 0.4; P[1, 3] = 0.2 # Estado 1 (só faculdade)
    elif trajectory_name == "Faculdade de computação + não trabalha na área": # Estado inicial 1
        P[1, 1] = 0.4; P[1, 2] = 0.2; P[1, 3] = 0.2; P[1, 16]= 0.1; P[1, 7] = 0.1
        P[3, 3] = 0.4; P[3, 16] = 0.2; P[3, 2] = 0.1; P[3, 7] = 0.1 # Estado 3 (faculdade + trabalho fora da área)
    elif trajectory_name == "Empreender (baixo capital)":
        P[5, 5] = 0.4; P[5, 8] = 0.2; P[5, 9] = 0.1; P[5, 7] = 0.2; P[5, 6] = 0.05
    elif trajectory_name == "Faculdade outra área + trabalha":
        P[4, 4] = 0.5; P[4, 16] = 0.3; P[4, 7] = 0.1; P[4, 12]= 0.05; P[4, 13]= 0.05
    
    P[6, 6] = 0.6; P[6, 7] = 0.1; P[6, 0] = 0.05; P[6, 5] = 0.05; P[6, 16] = 0.1
    P[7, 7] = 0.5; P[7, 0] = 0.1; P[7, 6] = 0.1; P[7, 5] = 0.1; P[7, 16] = 0.2
    P[16, 16] = 0.6; P[16, 7] = 0.15; P[16, 6] = 0.05; P[16, 0] = 0.05; P[16, 5] = 0.05; P[16, 12] = 0.05

    # Probabilidades de alcançar "Sucesso Elevado" (Estado 15)
    prob_sucesso = {5: 0.05, 8: 0.02, 9: 0.03, 10: 0.01, 11: 0.02, 12: 0.001, 13: 0.002, 14: 0.005, 16: 0.005}
    for estado_origem, chance in prob_sucesso.items():
        if estado_origem < n_total_states and 15 < n_total_states:
             P[estado_origem, 15] = max(P[estado_origem, 15], chance)
    
    # Normalização Robusta
    for i in range(n_total_states):
        # Probabilidade de ficar no estado i (P[i,i]) é o que sobra depois de transitar para outros j != i
        # Se P[i,i] não foi definida pela lógica da trajetória, ela será calculada aqui.
        # Se P[i,i] FOI definida, e a soma das outras é X, então P[i,i] deve ser ajustada.
        
        # Salva o P[i,i] definido pela trajetória, se houver, caso contrário considera 0 para cálculo inicial
        # A ideia é que P[i,i] definido pela trajetória tem precedência, e o resto é normalizado em volta.
        # Mas para simplificar, vamos calcular P[i,i] como o restante.
        
        current_P_ii = P[i,i] # Salva se foi definido pela lógica da trajetória
        P[i,i] = 0 # Zera P[i,i] temporariamente para calcular a soma das outras transições
        sum_off_diagonal = np.sum(P[i, :]) # Soma de todas as P[i,j] onde j != i

        if sum_off_diagonal >= 1.0:
            if sum_off_diagonal > 0:
                P[i, :] = P[i, :] / sum_off_diagonal # Normaliza P[i,j] para j!=i
            P[i,i] = 0.0 # P[i,i] se torna 0
        else:
            # Se P[i,i] foi explicitamente definida E é maior que o que sobrou, algo está errado.
            # Por agora, a lógica mais simples é: P[i,i] é o que falta para 1.
            P[i,i] = 1.0 - sum_off_diagonal
            
        # Verificação final (raramente necessária se a lógica acima for correta)
        final_row_sum = np.sum(P[i, :])
        if not np.isclose(final_row_sum, 1.0):
            if final_row_sum > 0: P[i, :] = P[i, :] / final_row_sum
            else: P[i,i] = 1.0
    return P

def get_matrix_for_simulation(trajectory_name, n_total_states):
    """Retorna a matriz customizada se existir, senão a padrão."""
    if trajectory_name in st.session_state.custom_transition_matrices:
        custom_P = st.session_state.custom_transition_matrices[trajectory_name]
        if custom_P.shape == (n_total_states, n_total_states):
            return custom_P.copy() 
    return get_default_base_transition_matrix(trajectory_name, n_total_states)

def normalize_matrix(matrix_df):
    """Normaliza um DataFrame (representando a matriz P) para que as linhas somem 1."""
    P_array = matrix_df.to_numpy(dtype=float)
    # Garantir que não haja valores negativos antes da normalização
    P_array[P_array < 0] = 0 
    
    for i in range(P_array.shape[0]):
        row_sum = np.sum(P_array[i, :])
        if row_sum > 0:
            P_array[i, :] = P_array[i, :] / row_sum
        else:
            # Linha toda zero ou negativa, define P[i,i] = 1 como fallback seguro
            P_array[i, :] = 0.0
            if i < P_array.shape[1]: # Evitar erro se i for maior que colunas (não deve acontecer)
                 P_array[i, i] = 1.0
    return pd.DataFrame(P_array, index=matrix_df.index, columns=matrix_df.columns)


# ... (run_simulation e funções de plotagem: plot_expected_income, etc. - sem grandes mudanças,
# mas precisam usar N_CURRENT_STATES e o df_states correto)
@st.cache_data # Cache para otimizar
def run_simulation(initial_state_idx, base_P, current_states_df, n_simulations_run, n_years_run, n_total_states_run):
    all_paths = np.zeros((n_simulations_run, n_years_run + 1), dtype=int)
    all_incomes = np.zeros((n_simulations_run, n_years_run + 1))

    for sim in range(n_simulations_run):
        current_state = initial_state_idx
        all_paths[sim, 0] = current_state
        all_incomes[sim, 0] = current_states_df.loc[current_state, "Renda"]

        for year in range(n_years_run):
            probabilities = base_P[current_state, :]
            if not np.isclose(np.sum(probabilities), 1.0):
                if np.sum(probabilities) <= 0 : 
                    probabilities = np.zeros(n_total_states_run)
                    probabilities[current_state] = 1.0
                else:
                    probabilities = probabilities / np.sum(probabilities)
                
            next_state = np.random.choice(n_total_states_run, p=probabilities)
            current_state = next_state
            all_paths[sim, year + 1] = current_state
            all_incomes[sim, year + 1] = current_states_df.loc[current_state, "Renda"]
            
    return all_paths, all_incomes

def plot_expected_income(all_incomes_list, trajectory_names_list, ax=None, title_suffix=""):
    create_new_fig = ax is None
    if create_new_fig: fig, ax = plt.subplots(figsize=(10, 6))
    for idx, all_incomes in enumerate(all_incomes_list):
        expected_income_per_year = np.mean(all_incomes, axis=0)
        ax.plot(range(len(expected_income_per_year)), expected_income_per_year, marker='o', linestyle='-', label=trajectory_names_list[idx])
    ax.set_xlabel("Ano"); ax.set_ylabel("Renda Média Esperada (R$)"); ax.set_title(f"Curva de Renda Esperada{title_suffix}"); ax.grid(True); ax.legend()
    if create_new_fig: st.pyplot(fig)
    return ax

def plot_final_state_distribution(all_paths_list, trajectory_names_list, current_states_df, title_suffix=""):
    if len(all_paths_list) == 1:
        all_paths = all_paths_list[0]; trajectory_name = trajectory_names_list[0]
        final_states = all_paths[:, -1]; state_counts = pd.Series(final_states).value_counts(normalize=True).sort_index()
        valid_indices = [idx for idx in state_counts.index if idx in current_states_df.index]
        state_labels = current_states_df.loc[valid_indices, "Nome"]; state_counts_filtered = state_counts.loc[valid_indices]
        fig, ax_new = plt.subplots(figsize=(12, 7)); state_counts_filtered.plot(kind='bar', ax=ax_new)
        ax_new.set_xticklabels(state_labels, rotation=45, ha="right"); ax_new.set_xlabel("Estado Final"); ax_new.set_ylabel("Proporção de Agentes")
        ax_new.set_title(f"Distribuição Final de Estados: {trajectory_name}{title_suffix}"); plt.tight_layout(); st.pyplot(fig)
    else:
        counts_list = [pd.Series(ap[:, -1]).value_counts(normalize=True).rename(tn) for i, (ap, tn) in enumerate(zip(all_paths_list, trajectory_names_list))]
        df_compare_final = pd.concat(counts_list, axis=1).fillna(0)
        df_compare_final = df_compare_final.reindex(current_states_df.index, fill_value=0)
        df_compare_final = df_compare_final[(df_compare_final.T != 0).any()]
        if df_compare_final.empty: st.warning("Nenhum dado para comparação de estados finais."); return
        df_compare_final.index = current_states_df.loc[df_compare_final.index, "Nome"]
        fig_comp_dist, ax_comp_dist = plt.subplots(figsize=(12,8)); df_compare_final.plot(kind='bar', ax=ax_comp_dist, width=0.8)
        ax_comp_dist.set_xlabel("Estado Final"); ax_comp_dist.set_ylabel("Proporção de Agentes"); ax_comp_dist.set_title(f"Comparativo: Distribuição Final de Estados{title_suffix}")
        plt.xticks(rotation=60, ha="right"); plt.tight_layout(); st.pyplot(fig_comp_dist)

def plot_final_income_distribution_hist(all_incomes_list, trajectory_names_list, title_suffix=""):
    fig, ax = plt.subplots(figsize=(10, 6))
    for i, all_incomes in enumerate(all_incomes_list):
        renda_final = all_incomes[:, -1]; pd.Series(renda_final).plot(kind='kde', ax=ax, linestyle='--', label=f"{trajectory_names_list[i]} (KDE)")
        ax.hist(renda_final, bins=30, edgecolor='black', alpha=0.5, density=True, label=f"{trajectory_names_list[i]} (Hist)") # Alpha reduzido
    ax.set_xlabel("Renda no Último Ano (R$)"); ax.set_ylabel("Densidade"); ax.set_title(f"Distribuição de Renda Final{title_suffix}"); ax.grid(axis='y', alpha=0.75); ax.legend()
    st.pyplot(fig)
    for i, all_incomes in enumerate(all_incomes_list):
        media_renda_final = np.mean(all_incomes[:, -1]); mediana_renda_final = np.median(all_incomes[:, -1])
        st.caption(f"{trajectory_names_list[i].split(' ')[0][:10]} - Média: R$ {media_renda_final:,.0f}, Mediana: R$ {mediana_renda_final:,.0f}".replace(",", "."))

def display_sample_paths(all_paths, current_states_df, n_years_run):
    st.subheader("Exemplos de Caminhos Individuais (Top 20)"); sample_df = pd.DataFrame(all_paths[:20, :]).applymap(lambda x: f"{x}: {current_states_df.loc[x, 'Nome']}")
    sample_df.columns = [f"Ano {i}" for i in range(n_years_run + 1)]; st.dataframe(sample_df)

def plot_transition_graph_mpl(P_matrix, trajectory_name, current_states_df, n_total_states_run):
    st.subheader(f"Grafo de Transição (Prob > 0.05): {trajectory_name}"); G = nx.DiGraph(); node_labels = {}; edge_labels = {}; nodes_in_graph = set()
    for i in range(n_total_states_run):
        is_sig_island = (P_matrix[i,i] > 0.05 and not np.any(P_matrix[i, np.arange(n_total_states_run) != i] > 0.05) and not np.any(P_matrix[np.arange(n_total_states_run) != i, i] > 0.05))
        if np.any(P_matrix[i, :] > 0.05) or np.any(P_matrix[:, i] > 0.05) or is_sig_island: nodes_in_graph.add(i)
    if not nodes_in_graph: st.warning("Nenhuma transição significativa no grafo."); return
    for i in nodes_in_graph:
        if i not in current_states_df.index: continue
        G.add_node(i); node_labels[i] = f"{i}: {current_states_df.loc[i, 'Nome'][:20]}..."
        for j in nodes_in_graph:
            if P_matrix[i, j] > 0.05: G.add_edge(i, j, weight=P_matrix[i, j]); edge_labels[(i,j)] = f"{P_matrix[i, j]:.2f}"
    if not G.nodes() or not G.edges(): st.warning("Nenhuma transição significativa no grafo (após filtro)."); return
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
    st.metric(label=f"Chance Renda Final ≥ R$ {renda_boa_thresh_data:,.0f}".replace(",", "."), value=f"{chance_renda_boa_limiar:.2%}")
    st.markdown("---")
    final_states = all_paths_data[:, -1]
    if 15 in current_states_df_data.index:
        success_freq = np.sum(final_states == 15) / n_simulations_data
        st.metric(label=f"Chance 'Sucesso Elevado' (Renda R$ {current_states_df_data.loc[15, 'Renda']:,})".replace(",", "."), value=f"{success_freq:.2%}")
    st.markdown("---")
    estados_alta_renda_idx = [idx for idx in [10, 11, 14] if idx in current_states_df_data.index]
    if estados_alta_renda_idx:
        nomes_estados_alta_renda = current_states_df_data.loc[estados_alta_renda_idx, "Nome"].tolist()
        chance_estados_alta_renda = np.sum(np.isin(final_states, estados_alta_renda_idx)) / n_simulations_data
        st.write(f"**Chance Outros Estados de Alta Renda:** ({'; '.join(nomes_estados_alta_renda)})")
        st.metric(label="Probabilidade", value=f"{chance_estados_alta_renda:.2%}")


# --- Interface Streamlit ---
st.set_page_config(layout="wide", page_title="Simulador de Carreiras")

# Navegação de Página
page = st.sidebar.radio("Navegar para:", ["🚀 Simulador", "⚙️ Configurações"], horizontal=True)

# Construir DataFrame de estados dinâmico com base nos salários da sessão
# Esta variável global será usada em ambas as páginas
CURRENT_DF_STATES = pd.DataFrame.from_dict({
    s_id: {
        "Nome": data["Nome"],
        "Categoria": data["Categoria"],
        "Renda": st.session_state.editable_salaries.get(s_id, data["Renda"])
    } for s_id, data in initial_states_data.items()
}, orient='index')
N_CURRENT_STATES = len(CURRENT_DF_STATES) # Número atual de estados (17)


# --- Página de Configurações ---
if page == "⚙️ Configurações":
    st.header("⚙️ Página de Configurações Avançadas")
    st.markdown("Ajuste os parâmetros base do simulador. As alterações são salvas automaticamente na sessão.")

    # 1. Configuração de Salários
    st.subheader("💰 Configurar Salários dos Estados")
    st.caption("Altere a renda mensal para cada estado. Clique fora do campo para salvar a alteração individual.")
    
    cols_salaries = st.columns(3) # Organizar em colunas para melhor visualização
    col_idx = 0
    for state_id_loop, data_loop in initial_states_data.items():
        with cols_salaries[col_idx % 3]:
            new_salary = st.number_input(
                label=f"Estado {state_id_loop}: {data_loop['Nome']}",
                value=int(st.session_state.editable_salaries.get(state_id_loop, data_loop["Renda"])), # Usar int para o input
                min_value=0,
                step=100,
                key=f"salary_config_input_{state_id_loop}",
                help=f"Renda mensal para o estado: {data_loop['Nome']}"
            )
            if new_salary != st.session_state.editable_salaries.get(state_id_loop, data_loop["Renda"]):
                 st.session_state.editable_salaries[state_id_loop] = new_salary
                 # st.rerun() # Rerun pode ser irritante aqui, o valor é atualizado na próxima interação
        col_idx += 1
    
    if st.button("Restaurar Salários Padrão", key="reset_salaries"):
        st.session_state.editable_salaries = {
            state_id: data["Renda"] for state_id, data in initial_states_data.items()
        }
        st.success("Salários padrão restaurados!")
        st.rerun()
    st.markdown("---")

    # 2. Configuração da Matriz de Transição
    st.subheader("📊 Configurar Matrizes de Probabilidade de Transição")
    st.caption("Selecione uma trajetória base para visualizar e editar sua matriz de transição.")
    
    config_traj_name = st.selectbox(
        "Selecione a Trajetória Base para Configurar sua Matriz:",
        list(trajectories_options.keys()),
        key="config_traj_select"
    )

    # Obter a matriz para edição: customizada ou padrão se não houver customização
    matrix_to_edit_np = get_matrix_for_simulation(config_traj_name, N_CURRENT_STATES)
    matrix_to_edit_df = pd.DataFrame(matrix_to_edit_np, index=STATE_LABELS, columns=STATE_LABELS)

    st.write(f"Editando Matriz para: **{config_traj_name}**")
    st.write("Cada linha representa o estado atual, e as colunas o próximo estado. A soma das probabilidades em cada LINHA deve ser 1.")
    
    # Usar uma chave diferente para o data_editor sempre que a trajetória mudar, para forçar o recarregamento
    edited_df = st.data_editor(
        matrix_to_edit_df,
        key=f"matrix_editor_{config_traj_name}",
        num_rows="dynamic", # Embora seja fixo, isso ajuda na renderização
        use_container_width=True,
        # Configurar colunas para terem min_value 0 e max_value 1 (opcional, mas bom)
        # column_config={col: st.column_config.NumberColumn(min_value=0.0, max_value=1.0, step=0.01, format="%.3f") for col in STATE_LABELS}
    )

    col_actions1, col_actions2 = st.columns(2)
    with col_actions1:
        if st.button("✔️ Normalizar e Salvar Matriz Alterada", key=f"save_matrix_{config_traj_name}"):
            normalized_df = normalize_matrix(edited_df)
            st.session_state.custom_transition_matrices[config_traj_name] = normalized_df.to_numpy(dtype=float)
            st.success(f"Matriz para '{config_traj_name}' normalizada e salva!")
            st.rerun() # Rerun para mostrar a matriz normalizada no editor
    with col_actions2:
        if st.button("↩️ Restaurar Matriz Padrão (esta trajetória)", key=f"reset_matrix_{config_traj_name}"):
            if config_traj_name in st.session_state.custom_transition_matrices:
                del st.session_state.custom_transition_matrices[config_traj_name]
            st.success(f"Matriz padrão para '{config_traj_name}' restaurada.")
            st.rerun() # Rerun para mostrar a matriz padrão no editor


# --- Página do Simulador ---
elif page == "🚀 Simulador":
    st.header("🚀 Simulador de Trajetórias Profissionais")
    st.markdown("Use as configurações da barra lateral e da página 'Configurações' para rodar as simulações.")

    # Controles do Simulador na Sidebar (já definidos antes da navegação de página)
    sidebar_traj1_name = st.sidebar.selectbox(
        "Escolha a trajetória inicial (Trajetória 1):",
        list(trajectories_options.keys()),
        key="sim_traj1_select"
    )
    sidebar_renda_boa_thresh = st.sidebar.number_input(
        "Limiar 'Renda Boa' (R$):", 
        min_value=0, value=st.session_state.get('renda_boa_thresh_val', 8000), step=500, format="%d", key="sim_renda_boa_thresh"
    )
    st.session_state.renda_boa_thresh_val = sidebar_renda_boa_thresh # Salvar para persistir

    sidebar_n_simul = st.sidebar.number_input(
        "Número de Simulações:", min_value=100, max_value=10000, 
        value=st.session_state.get('n_simul_val', 1000), step=100, key="sim_n_simul"
    )
    st.session_state.n_simul_val = sidebar_n_simul

    st.sidebar.markdown("---")
    sim_compare_mode = st.sidebar.checkbox("Ativar modo de comparação", key="sim_compare_check")
    sidebar_traj2_name = None
    if sim_compare_mode:
        available_for_compare = [t for t in trajectories_options.keys() if t != sidebar_traj1_name]
        if available_for_compare:
            sidebar_traj2_name = st.sidebar.selectbox(
                "Escolha a Trajetória 2 para Comparar:", available_for_compare, index=0, key="sim_traj2_select"
            )
        else:
            st.sidebar.warning("Apenas uma trajetória selecionada, não é possível comparar."); sim_compare_mode = False
    
    # Botão de Rodar Simulação
    if st.sidebar.button("📊 Rodar Simulação Agora", key="run_sim_button_main"):
        df_states_for_sim = CURRENT_DF_STATES.copy() # Usa os salários atuais
        
        # Simulação para Trajetória 1
        initial_state_idx_1 = trajectories_options[sidebar_traj1_name]
        P_base_1 = get_matrix_for_simulation(sidebar_traj1_name, N_CURRENT_STATES)
        
        st.subheader(f"Resultados para Trajetória 1: {sidebar_traj1_name}")
        with st.spinner(f"Rodando {sidebar_n_simul} simulações para {sidebar_traj1_name}..."):
            all_paths_1, all_incomes_1 = run_simulation(
                initial_state_idx_1, P_base_1, df_states_for_sim, sidebar_n_simul, N_YEARS, N_CURRENT_STATES
            )

        tab_titles_1 = ["📈 Renda Média", "📊 Dist. Estados", "SAL Dist. Renda", "👣 Caminhos", "🕸️ Grafo", "🎯 Renda Boa"]
        tabs_1 = st.tabs(tab_titles_1)
        with tabs_1[0]: plot_expected_income([all_incomes_1], [sidebar_traj1_name])
        with tabs_1[1]: plot_final_state_distribution([all_paths_1], [sidebar_traj1_name], df_states_for_sim)
        with tabs_1[2]: plot_final_income_distribution_hist([all_incomes_1], [sidebar_traj1_name])
        with tabs_1[3]: display_sample_paths(all_paths_1, df_states_for_sim, N_YEARS)
        with tabs_1[4]: plot_transition_graph_mpl(P_base_1, sidebar_traj1_name, df_states_for_sim, N_CURRENT_STATES)
        with tabs_1[5]: display_renda_boa_metrics(all_paths_1, all_incomes_1, sidebar_traj1_name, df_states_for_sim, sidebar_renda_boa_thresh, sidebar_n_simul)

        # Simulação e Abas para Comparação (se ativado)
        if sim_compare_mode and sidebar_traj2_name:
            initial_state_idx_2 = trajectories_options[sidebar_traj2_name]
            P_base_2 = get_matrix_for_simulation(sidebar_traj2_name, N_CURRENT_STATES)
            
            st.markdown("---"); st.subheader(f"Comparação: {sidebar_traj1_name} vs {sidebar_traj2_name}")
            with st.spinner(f"Rodando {sidebar_n_simul} simulações para {sidebar_traj2_name}..."):
                all_paths_2, all_incomes_2 = run_simulation(
                    initial_state_idx_2, P_base_2, df_states_for_sim, sidebar_n_simul, N_YEARS, N_CURRENT_STATES
                )
            
            comp_tab_titles = ["📈 Rendas Médias Comp.", "📊 Estados Finais Comp.", "SAL Rendas Finais Comp.", "🎯 Renda Boa Comp."]
            comp_tabs = st.tabs(comp_tab_titles)
            with comp_tabs[0]: plot_expected_income([all_incomes_1, all_incomes_2], [sidebar_traj1_name, sidebar_traj2_name], title_suffix=" (Comparativo)")
            with comp_tabs[1]: plot_final_state_distribution([all_paths_1, all_paths_2], [sidebar_traj1_name, sidebar_traj2_name], df_states_for_sim, title_suffix=" (Comparativo)")
            with comp_tabs[2]: plot_final_income_distribution_hist([all_incomes_1, all_incomes_2], [sidebar_traj1_name, sidebar_traj2_name], title_suffix=" (Comparativo)")
            with comp_tabs[3]: 
                col1, col2 = st.columns(2)
                with col1: display_renda_boa_metrics(all_paths_1, all_incomes_1, sidebar_traj1_name, df_states_for_sim, sidebar_renda_boa_thresh, sidebar_n_simul, title_prefix="Traj. 1: ")
                with col2: display_renda_boa_metrics(all_paths_2, all_incomes_2, sidebar_traj2_name, df_states_for_sim, sidebar_renda_boa_thresh, sidebar_n_simul, title_prefix="Traj. 2: ")
    else:
        st.info("Ajuste as configurações na barra lateral e clique em 'Rodar Simulação Agora'. Visite a página 'Configurações' para ajustes avançados de salários e probabilidades.")

st.sidebar.markdown("---")
st.sidebar.caption("Simulador Educacional de Carreiras")