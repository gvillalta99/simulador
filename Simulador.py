import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import networkx as nx

# --- Configurações e Dados Iniciais ---
N_YEARS = 10
N_SIMULATIONS = 10000  # Número de agentes simulados por trajetória

# Definição dos Estados
states_data = {
    0: {"Nome": "Técnico trabalhando", "Categoria": "Entrada", "Renda": 1500},
    1: {"Nome": "Faculdade de computação (sem trabalho)", "Categoria": "Entrada", "Renda": 0},
    2: {"Nome": "Faculdade comp. (trabalhando na área)", "Categoria": "Entrada", "Renda": 3500},
    3: {"Nome": "Faculdade comp. (fora da área)", "Categoria": "Entrada", "Renda": 2200},
    4: {"Nome": "Outra faculdade (trabalhando)", "Categoria": "Entrada", "Renda": 2000},
    5: {"Nome": "Empreendendo", "Categoria": "Entrada", "Renda": 1200},
    6: {"Nome": "Não estuda nem trabalha", "Categoria": "Entrada", "Renda": 0},
    7: {"Nome": "Desempregado", "Categoria": "Entrada", "Renda": 0},
    8: {"Nome": "Pequena empresa na área", "Categoria": "Progresso", "Renda": 2800},
    9: {"Nome": "Startup na área", "Categoria": "Progresso", "Renda": 4000},
    10: {"Nome": "Grande empresa na área", "Categoria": "Progresso", "Renda": 6500},
    11: {"Nome": "Empresa global na área", "Categoria": "Progresso", "Renda": 9000},
    12: {"Nome": "Serviço público municipal", "Categoria": "Progresso", "Renda": 2500},
    13: {"Nome": "Serviço público estadual", "Categoria": "Progresso", "Renda": 3500},
    14: {"Nome": "Serviço público federal", "Categoria": "Progresso", "Renda": 5000},
    15: {"Nome": "Sucesso elevado", "Categoria": "Sucesso extremo", "Renda": 20000}
}
df_states = pd.DataFrame.from_dict(states_data, orient='index')
N_STATES = len(df_states)

# Trajetórias Principais e seus estados iniciais
trajectories_options = {
    "Trabalha como Técnico": 0,
    "Faculdade de computação + não trabalha na área": 1,
    "Faculdade de computação + trabalha na área": 2,
    "Faculdade outra área + trabalha na área": 3,
    "Faculdade outra área + não trabalha na área": 4,
    "Empreender": 5,
    "Não estuda nem trabalha": 6
}

# --- Funções do Modelo de Markov ---

@st.cache_data
def get_base_transition_matrix(trajectory_name):
    P = np.zeros((N_STATES, N_STATES))
    for i in range(N_STATES):
        P[i, i] = 0.3

    # Trabalhando como Tecnico
    P[0, 0]     = 0.60
    P[0, 1]     = 0.05
    P[0, 2]     = 0.10
    P[0, 5]     = 0.05
    P[0, 6]     = 0.05
    P[0, 7]     = 0.10
    P[0, 8]     = 0.10
    P[0, 10]    = 0.05
    P[0, 12]    = 0.05
    # Faculdade sem trabalho
    P[1, 1]     = 0.05
    P[1, 2]     = 0.50
    P[1, 3]     = 0.20
    P[1, 4]     = 0.05
    P[1, 5]     = 0.10
    P[1, 7]     = 0.10
    # Faculdade de computação + trabalha na área
    P[2, 1]     = 0.50
    P[2, 2]     = 0.50
    P[2, 7]     = 0.05
    P[2, 9]     = 0.10
    P[2, 10]    = 0.20
    P[2, 11]    = 0.05
    # Faculdade + Trabalha fora da area
    P[3, 2]     = 0.20
    P[3, 4]     = 0.10
    # faculdade de outra area
    P[4,2]      = 0.01
    # Empreender
    P[5, 5]     = 0.45
    P[5, 6]     = 0.04
    P[5, 7]     = 0.20
    P[5, 8]     = 0.30
    P[5, 9]     = 0.05
    P[5, 10]    = 0.05
    P[5, 15]    = 0.01 # Sucesso Elevado
    
    # Não estuda nem trabalha
    P[6, 6]     = 0.70
    P[6, 0]     = 0.10
    P[6, 2]     = 0.10
    P[6, 4]     = 0.05
    P[6, 5]     = 0.05
    # Desempregado
    P[7, 7]     = 0.60
    P[7, 0]     = 0.10
    P[7, 6]     = 0.10
    P[7, 5]     = 0.10
    # Pequena empresa
    P[8, 9]     = 0.10
    P[8, 10]    = 0.10
    P[8, 15]    = 0.01
    # Startup
    P[9, 7]     = 0.20
    P[9, 10]    = 0.10
    P[9, 11]    = 0.05
    P[9, 15]    = 0.03
    # Grande empresa
    P[10, 7]    = 0.05
    P[10, 9]    = 0.10
    P[10, 15]   = 0.01
    # Empresa Global
    P[11, 7]    = 0.08
    P[11, 9]    = 0.05
    P[11, 10]   = 0.05
    P[11, 15]   = 0.02
    # Pub. Municipal
    P[12, 7]    = 0.001
    P[12, 12]   = 0.80
    P[12, 13]   = 0.12
    P[12, 14]   = 0.078
    P[12, 15]   = 0.001
    # Pub. Estadual
    P[13, 7]  = 0.001
    P[13, 12] = 0.047
    P[13, 13] = 0.85
    P[13, 14] = 0.10
    P[13, 15] = 0.002
    # Pub. Federal
    P[14, 7]  = 0.001
    P[14, 12] = 0.044
    P[14, 13] = 0.05
    P[14, 14] = 0.90
    P[14, 15] = 0.005

    # Grande Sucesso
    P[15, 7]  = 0.05
    P[15, 10] = 0.10
    P[15, 15] = 0.70

    for i in range(N_STATES):
        current_sum = np.sum(P[i, :])
        if current_sum == 0:
            P[i, i] = 1.0
        elif current_sum > 1.0:
            # Priorizar P[i,i] e P[i,15] se existirem, ajustar o resto
            stay_prob = P[i,i]
            success_prob = P[i,15]
            other_sum = current_sum - stay_prob - success_prob
            
            if other_sum > 0:
                target_other_sum = 1.0 - stay_prob - success_prob
                if target_other_sum < 0: # Acontece se P[i,i] + P[i,15] > 1
                    # Neste caso, precisa reduzir P[i,i] ou P[i,15] ou ambos
                    # Simplificação: reduzir P[i,i] primeiro se P[i,i] não for a única > 0
                    if stay_prob > 0 and (success_prob > 0 or other_sum > 0):
                         P[i,i] = max(0, 1.0 - success_prob) # Reduz P[i,i]
                         if P[i,i] + success_prob > 1.0: P[i,15] = 1.0 - P[i,i] # Garante que P[i,15] não exceda
                    elif success_prob > 0 : # Se P[i,i] é 0, mas P[i,15] é > 1
                        P[i,15] = 1.0
                    # Zera os outros
                    for j in range(N_STATES):
                        if i != j and j != 15: P[i,j] = 0.0
                else: # target_other_sum >=0
                    factor = target_other_sum / other_sum
                    for j in range(N_STATES):
                        if i != j and j != 15:
                            P[i,j] *= factor
        
        # Normalização final
        row_sum = np.sum(P[i, :])
        if row_sum > 0:
            P[i, :] = P[i, :] / row_sum
        else:
             P[i, i] = 1.0
    return P

@st.cache_data
def run_simulation(initial_state_idx, base_P):
    all_paths = np.zeros((N_SIMULATIONS, N_YEARS + 1), dtype=int)
    all_incomes = np.zeros((N_SIMULATIONS, N_YEARS + 1))

    for sim in range(N_SIMULATIONS):
        current_state = initial_state_idx
        all_paths[sim, 0] = current_state
        all_incomes[sim, 0] = df_states.loc[current_state, "Renda"]

        for year in range(N_YEARS):
            P_t = base_P
            probabilities = P_t[current_state, :]
            # Pequena verificação para garantir que as probabilidades somem 1 (devido a possíveis erros de ponto flutuante)
            if not np.isclose(np.sum(probabilities), 1.0):
                probabilities = probabilities / np.sum(probabilities)

            next_state = np.random.choice(N_STATES, p=probabilities)
            current_state = next_state
            all_paths[sim, year + 1] = current_state
            all_incomes[sim, year + 1] = df_states.loc[current_state, "Renda"]
            
    return all_paths, all_incomes

# --- Funções de Visualização ---
def plot_expected_income(all_incomes, trajectory_name, ax=None):
    expected_income_per_year = np.mean(all_incomes, axis=0)
    
    if ax is None:
        fig, ax_new = plt.subplots(figsize=(10, 6))
        ax_new.plot(range(N_YEARS + 1), expected_income_per_year, marker='o', linestyle='-', label=trajectory_name)
        ax_new.set_xlabel("Ano")
        ax_new.set_ylabel("Renda Média Esperada (R$)")
        ax_new.set_title(f"Curva de Renda Esperada: {trajectory_name}")
        ax_new.grid(True)
        ax_new.legend()
        st.pyplot(fig)
    else:
        ax.plot(range(N_YEARS + 1), expected_income_per_year, marker='o', linestyle='-', label=trajectory_name)
        return ax

def plot_final_state_distribution(all_paths, trajectory_name, ax=None):
    final_states = all_paths[:, -1]
    state_counts = pd.Series(final_states).value_counts(normalize=True).sort_index()
    state_labels = df_states.loc[state_counts.index, "Nome"]

    if ax is None:
        fig, ax_new = plt.subplots(figsize=(12, 7))
        state_counts.plot(kind='bar', ax=ax_new)
        ax_new.set_xticklabels(state_labels, rotation=45, ha="right")
        ax_new.set_xlabel("Estado Final")
        ax_new.set_ylabel("Proporção de Agentes")
        ax_new.set_title(f"Distribuição Final de Estados: {trajectory_name}")
        plt.tight_layout()
        st.pyplot(fig)
    else:
        # Se for comparar, usar a cor da linha correspondente para a barra
        bar_color = plt.gca().lines[-1].get_color() if plt.gca().lines else None
        state_counts.plot(kind='bar', ax=ax, color=bar_color, alpha=0.7)
        ax.set_xticklabels(state_labels, rotation=45, ha="right")
        return ax

def display_sample_paths(all_paths):
    st.subheader("Exemplos de Caminhos Individuais (Primeiros 20)")
    sample_df = pd.DataFrame(all_paths[:20, :]).applymap(lambda x: f"{x}: {df_states.loc[x, 'Nome']}")
    sample_df.columns = [f"Ano {i}" for i in range(N_YEARS + 1)]
    st.dataframe(sample_df)

def plot_transition_graph_mpl(P_matrix, trajectory_name):
    st.subheader(f"Grafo de Transição (Probabilidades > 0.05): {trajectory_name}")
    G = nx.DiGraph()
    node_labels = {}
    for i in range(N_STATES):
        # Adiciona nós apenas se eles participam de alguma transição significativa ou têm self-loop
        has_significant_transition = np.any(P_matrix[i, :] > 0.05) or np.any(P_matrix[:, i] > 0.05)
        if has_significant_transition:
            G.add_node(i)
            node_labels[i] = f"{i}: {df_states.loc[i, 'Nome'][:20]}..." # Nome mais curto

    edge_labels = {}
    for i in G.nodes(): # Itera sobre os nós adicionados ao grafo
        for j in G.nodes():
            if P_matrix[i, j] > 0.05: # Limiar para não poluir o grafo
                G.add_edge(i, j, weight=P_matrix[i, j])
                edge_labels[(i,j)] = f"{P_matrix[i, j]:.2f}"
    
    if not G.nodes():
        st.warning("Nenhuma transição significativa (>5%) para exibir no grafo para esta trajetória.")
        return

    fig, ax = plt.subplots(figsize=(18, 18)) # Aumentar o tamanho da figura
    
    # Tentar um layout que espalhe mais os nós
    try:
        # O layout 'kamada_kawai' pode ser bom para grafos menores e bem conectados.
        # 'spring_layout' é mais geral.
        pos = nx.kamada_kawai_layout(G) 
    except Exception: # Fallback para spring_layout se kamada_kawai falhar (ex: grafo desconectado)
        pos = nx.spring_layout(G, k=1.5/np.sqrt(len(G.nodes())) if len(G.nodes()) > 0 else 1, iterations=30)

    nx.draw_networkx_nodes(G, pos, ax=ax, node_size=3500, node_color="skyblue", alpha=0.9)
    nx.draw_networkx_edges(G, pos, ax=ax, edgelist=G.edges(), arrowstyle='-|>', arrowsize=20, 
                           edge_color="gray", alpha=0.7, node_size=3500) # node_size aqui afeta o encurtamento da aresta
    nx.draw_networkx_labels(G, pos, labels=node_labels, ax=ax, font_size=9)
    nx.draw_networkx_edge_labels(G, pos, edge_labels=edge_labels, ax=ax, font_color='darkred', font_size=8)
    
    ax.set_title(f"Grafo de Transição (Matplotlib): {trajectory_name}", fontsize=15)
    plt.axis('off') # Desligar eixos
    st.pyplot(fig)


# --- Interface Streamlit ---
st.set_page_config(layout="wide")
st.title("🧠 Simulador de Caminhos Profissionais")
st.markdown("""
Esta ferramenta simula trajetórias de vida profissional ao longo de 10 anos usando Cadeias de Markov.
Escolha uma trajetória inicial e veja os resultados esperados.
**Atenção:** As probabilidades de transição são ilustrativas e simplificadas.
""")

# --- Sidebar para Controles ---
st.sidebar.header("Configurações da Simulação")
selected_trajectory_name = st.sidebar.selectbox(
    "Escolha a trajetória inicial:",
    list(trajectories_options.keys())
)

st.sidebar.header("Comparação de Trajetórias")
compare_mode = st.sidebar.checkbox("Ativar modo de comparação")
selected_trajectory_name_2 = None
if compare_mode:
    available_for_compare = [t for t in trajectories_options.keys() if t != selected_trajectory_name]
    if available_for_compare:
        selected_trajectory_name_2 = st.sidebar.selectbox(
            "Escolha a segunda trajetória para comparar:",
            available_for_compare,
            index=0 # Default para o primeiro item diferente
        )
    else:
        st.sidebar.warning("Apenas uma trajetória disponível para seleção, não é possível comparar.")
        compare_mode = False # Desativa se não há o que comparar


# --- Execução e Exibição ---
if st.sidebar.button("🚀 Rodar Simulação"):
    initial_state_idx = trajectories_options[selected_trajectory_name]
    P_base_1 = get_base_transition_matrix(selected_trajectory_name)
    
    st.header(f"Resultados para: {selected_trajectory_name}")
    all_paths_1, all_incomes_1 = run_simulation(initial_state_idx, P_base_1)

    tab1, tab2, tab3, tab4 = st.tabs(["📈 Curva de Renda", "📊 Distribuição Final", "👣 Caminhos Exemplo", "🕸️ Grafo de Transição"])

    with tab1:
        plot_expected_income(all_incomes_1, selected_trajectory_name)
    with tab2:
        plot_final_state_distribution(all_paths_1, selected_trajectory_name)
    with tab3:
        display_sample_paths(all_paths_1)
    with tab4:
        plot_transition_graph_mpl(P_base_1, selected_trajectory_name)

    final_states_1 = all_paths_1[:, -1]
    success_freq_1 = np.sum(final_states_1 == 15) / N_SIMULATIONS
    st.subheader(f"🌟 Frequência de 'Sucesso Elevado' ({selected_trajectory_name}): {success_freq_1:.2%}")
    st.markdown(f"O estado 'Sucesso Elevado' (Renda: {df_states.loc[15, 'Renda']:,} BRL) representa uma ascensão profissional significativa. Esta frequência indica a proporção de simulações que alcançaram este estado após 10 anos.".replace(",", "."))

    if compare_mode and selected_trajectory_name_2:
        st.header(f"Comparação: {selected_trajectory_name} vs {selected_trajectory_name_2}")
        initial_state_idx_2 = trajectories_options[selected_trajectory_name_2]
        P_base_2 = get_base_transition_matrix(selected_trajectory_name_2)
        all_paths_2, all_incomes_2 = run_simulation(initial_state_idx_2, P_base_2)

        col1, col2 = st.columns(2)
        with col1:
            st.subheader("Curvas de Renda Esperada")
            fig_comp_income, ax_comp_income = plt.subplots(figsize=(10,6))
            plot_expected_income(all_incomes_1, selected_trajectory_name, ax=ax_comp_income)
            plot_expected_income(all_incomes_2, selected_trajectory_name_2, ax=ax_comp_income)
            ax_comp_income.set_xlabel("Ano")
            ax_comp_income.set_ylabel("Renda Média Esperada (R$)")
            ax_comp_income.set_title("Comparativo: Curvas de Renda Esperada")
            ax_comp_income.grid(True)
            ax_comp_income.legend()
            st.pyplot(fig_comp_income)

            final_states_2 = all_paths_2[:, -1]
            success_freq_2 = np.sum(final_states_2 == 15) / N_SIMULATIONS
            st.markdown(f"🌟 Frequência de 'Sucesso Elevado' ({selected_trajectory_name_2}): {success_freq_2:.2%}")

        with col2:
            st.subheader("Distribuições Finais de Estados")
            final_counts_1 = pd.Series(all_paths_1[:, -1]).value_counts(normalize=True).rename(selected_trajectory_name)
            final_counts_2 = pd.Series(all_paths_2[:, -1]).value_counts(normalize=True).rename(selected_trajectory_name_2)
            
            df_compare_final = pd.concat([final_counts_1, final_counts_2], axis=1).fillna(0)
            # Garantir que todos os estados estejam presentes para um índice consistente
            all_possible_states_idx = df_states.index
            df_compare_final = df_compare_final.reindex(all_possible_states_idx, fill_value=0)
            df_compare_final = df_compare_final[(df_compare_final.T != 0).any()] # Remover linhas com soma zero
            
            df_compare_final.index = df_states.loc[df_compare_final.index, "Nome"]

            fig_comp_dist, ax_comp_dist = plt.subplots(figsize=(12,8)) # Aumentar um pouco
            df_compare_final.plot(kind='bar', ax=ax_comp_dist, width=0.8)
            ax_comp_dist.set_xlabel("Estado Final")
            ax_comp_dist.set_ylabel("Proporção de Agentes")
            ax_comp_dist.set_title("Comparativo: Distribuição Final de Estados")
            plt.xticks(rotation=60, ha="right") # Ajustar rotação para melhor visualização
            plt.tight_layout()
            st.pyplot(fig_comp_dist)
else:
    st.info("Escolha uma trajetória na barra lateral e clique em 'Rodar Simulação'.")

st.sidebar.markdown("---")
st.sidebar.markdown("""
**Sobre o Modelo:**
- **Estados:** Representam sua condição profissional/educacional.
- **Cadeia de Markov:** Um modelo que descreve sequências de eventos possíveis onde a probabilidade de cada evento depende apenas do estado atual.
- **Não-Homogênea (Simplificado):** Idealmente, as probabilidades de transição mudariam a cada ano. Nesta simulação, a matriz de transição base de uma trajetória é a mesma ao longo dos 10 anos para simplificar.
- **Dados:** As rendas são baseadas na descrição. As probabilidades de transição são **ilustrativas**.
""")
st.sidebar.markdown("---")
st.sidebar.markdown("Desenvolvido como exemplo educacional.")

