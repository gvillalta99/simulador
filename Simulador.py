import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import networkx as nx

# --- Configurações e Dados Iniciais ---
N_YEARS = 10
N_SIMULATIONS = 1000  # Número de agentes simulados por trajetória

# Definição dos Estados
states_data = {
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
    15: {"Nome": "Sucesso elevado", "Categoria": "Sucesso extremo", "Renda": 20000}
}
df_states = pd.DataFrame.from_dict(states_data, orient='index')
N_STATES = len(df_states)

# Trajetórias Principais e seus estados iniciais
trajectories_options = {
    "Técnico e não faz faculdade": 0,
    "Faculdade de computação + trabalha na área": 2,
    "Faculdade de computação + não trabalha na área": 1, # Começa sem trabalho, pode evoluir para o estado 3
    "Empreender (baixo capital)": 5,
    "Faculdade outra área + trabalha": 4,
    "Não estuda nem trabalha": 6
}

# --- Funções do Modelo de Markov ---

@st.cache_data # Cache para otimizar
def get_base_transition_matrix(trajectory_name):
    """
    Cria uma MATRIZ DE TRANSIÇÃO BASE para uma dada trajetória.
    Para um modelo verdadeiramente não-homogêneo, esta função retornaria
    uma LISTA de N_YEARS matrizes.
    Aqui, para simplificar, criamos UMA matriz base.
    As probabilidades são ILUSTRATIVAS e precisam de ajuste fino.
    """
    P = np.zeros((N_STATES, N_STATES))

    # Probabilidades base (genéricas - precisa de muita personalização)
    # Manter-se no estado atual ou mover para estados adjacentes/prováveis
    for i in range(N_STATES):
        P[i, i] = 0.3  # Chance base de permanecer no estado

    # Exemplo para "Técnico e não faz faculdade" (inicia em 0)
    if trajectory_name == "Técnico e não faz faculdade":
        P[0, 0] = 0.6  # Ficar como técnico
        P[0, 8] = 0.1  # Ir para pequena empresa
        P[0, 10] = 0.05 # Grande empresa (mais difícil)
        P[0, 7] = 0.1  # Desempregar
        P[0, 6] = 0.05 # Desistir
        P[0, 5] = 0.05 # Tentar empreender
        P[0, 12] = 0.05 # Servico publico municipal

    # Exemplo para "Faculdade de computação + trabalha na área" (inicia em 2)
    elif trajectory_name == "Faculdade de computação + trabalha na área":
        P[2, 2] = 0.5 # Continuar faculdade + trabalho
        P[2, 10] = 0.2 # Grande empresa após/durante faculdade
        P[2, 9] = 0.1  # Startup
        P[2, 11] = 0.05 # Empresa global
        P[2, 7] = 0.05 # Desempregar
        P[1, 2] = 0.4 # Se estava só na faculdade (estado 1), conseguir trabalho na área
        P[1, 3] = 0.2 # Se estava só na faculdade (estado 1), conseguir trabalho fora da área
        P[1, 1] = 0.3 # Continuar só faculdade

    # Exemplo para "Empreender (baixo capital)" (inicia em 5)
    elif trajectory_name == "Empreender (baixo capital)":
        P[5, 5] = 0.4  # Continuar empreendendo
        P[5, 8] = 0.2  # Evoluir para pequena empresa
        P[5, 9] = 0.1  # Evoluir para startup
        P[5, 15] = 0.05 # Sucesso Elevado (5% ao ano, como especificado)
        P[5, 7] = 0.2  # Falhar e desempregar
        P[5, 6] = 0.05 # Desistir

    # ... (Definir lógicas similares para outras trajetórias) ...

    # Lógica para "Não estuda nem trabalha" (estado 6) e "Desempregado" (estado 7)
    P[6, 6] = 0.7
    P[6, 7] = 0.1 # Pode virar desempregado procurando
    P[6, 0] = 0.05 # Tentar ser técnico
    P[6, 5] = 0.05 # Tentar empreender com baixo capital
    P[7, 7] = 0.6
    P[7, 0] = 0.1 # Conseguir vaga de técnico
    P[7, 6] = 0.1 # Desistir de procurar
    P[7, 5] = 0.1 # Tentar empreender

    # Probabilidades de alcançar "Sucesso Elevado" (Estado 15) de outros estados de progresso
    # Estas são adicionais e podem sobrescrever ou complementar as anteriores
    # Garantindo que não ultrapasse 100% na linha
    if P[5,15] == 0: P[5,15] = 0.05 # Empreendedores de baixo capital (já definido acima, mas como exemplo)
    P[8, 15] = 0.02  # Pequena empresa -> Sucesso
    P[9, 15] = 0.03  # Startup -> Sucesso (1-3%)
    P[10, 15] = 0.01 # Grande empresa -> Sucesso (<1%)
    P[11, 15] = 0.02 # Empresa Global -> Sucesso (1-3%)
    # Servidores públicos tem chance muito baixa de ir para "Sucesso Elevado" por esta via
    P[12, 15] = 0.001
    P[13, 15] = 0.002
    P[14, 15] = 0.005

    # Garantir que estados 8-15 não sejam pontos de entrada diretos (já tratado pela escolha inicial)
    # Garantir que as linhas da matriz somem 1
    for i in range(N_STATES):
        # Se alguma transição para estado de progresso/sucesso foi definida e não havia antes
        current_sum = np.sum(P[i, :])
        if current_sum == 0: # Se não há transições definidas, fica no mesmo estado
            P[i, i] = 1.0
        elif current_sum > 1.0: # Se a soma ultrapassou 1 devido às adições de "Sucesso Elevado"
            # Reduzir proporcionalmente outras transições, exceto para o próprio estado e sucesso
            # Esta é uma normalização simples, pode ser mais sofisticada
            factor = (1.0 - P[i,i] - P[i,15]) / (current_sum - P[i,i] - P[i,15]) if (current_sum - P[i,i] - P[i,15]) > 0 else 0
            for j in range(N_STATES):
                if i != j and j != 15:
                    P[i,j] *= factor
            P[i,15] = min(P[i,15], 1.0 - P[i,i] - np.sum(P[i, [j for j in range(N_STATES) if j!=i and j!=15]]))


        # Normalização final para garantir que a soma seja 1
        row_sum = np.sum(P[i, :])
        if row_sum > 0:
            P[i, :] = P[i, :] / row_sum
        else: # Caso raro: se ainda for zero, fica no mesmo estado
             P[i, i] = 1.0
    return P

@st.cache_data
def run_simulation(initial_state_idx, base_P):
    """
    Roda N_SIMULATIONS por N_YEARS.
    Para não-homogêneo, base_P seria uma lista de matrizes [P_ano1, P_ano2, ...].
    Aqui, usamos a mesma base_P para todos os anos.
    """
    all_paths = np.zeros((N_SIMULATIONS, N_YEARS + 1), dtype=int)
    all_incomes = np.zeros((N_SIMULATIONS, N_YEARS + 1))

    for sim in range(N_SIMULATIONS):
        current_state = initial_state_idx
        all_paths[sim, 0] = current_state
        all_incomes[sim, 0] = df_states.loc[current_state, "Renda"]

        for year in range(N_YEARS):
            # Aqui seria P_t = list_of_P[year] para não-homogêneo
            # Por simplificação, usamos a base_P
            P_t = base_P
            
            # Pequena variação anual para simular não-homogeneidade (opcional, exemplo simples)
            # P_t_adjusted = P_t.copy()
            # if year > 5 and current_state < 8: # Ex: após 5 anos, se ainda em estado de entrada, aumenta chance de mudar
            #     P_t_adjusted[current_state, current_state] *= 0.9
            #     # Re-normalizar linha P_t_adjusted[current_state,:]
            #     row_sum = np.sum(P_t_adjusted[current_state, :])
            #     if row_sum > 0: P_t_adjusted[current_state, :] /= row_sum
            #     else: P_t_adjusted[current_state, current_state] = 1.0


            probabilities = P_t[current_state, :]
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
        state_counts.plot(kind='bar', ax=ax, color=plt.gca().lines[-1].get_color() if plt.gca().lines else None, alpha=0.7) # Use last color
        ax.set_xticklabels(state_labels, rotation=45, ha="right")
        return ax

def display_sample_paths(all_paths):
    st.subheader("Exemplos de Caminhos Individuais (Primeiros 20)")
    sample_df = pd.DataFrame(all_paths[:20, :]).applymap(lambda x: f"{x}: {df_states.loc[x, 'Nome']}")
    sample_df.columns = [f"Ano {i}" for i in range(N_YEARS + 1)]
    st.dataframe(sample_df)

def plot_transition_graph(P_matrix, trajectory_name):
    st.subheader(f"Grafo de Transição (Probabilidades > 0.05 para clareza): {trajectory_name}")
    G = nx.DiGraph()
    for i in range(N_STATES):
        G.add_node(i, label=f"{i}:{df_states.loc[i, 'Nome'][:15]}...") # Nome curto para o label

    for i in range(N_STATES):
        for j in range(N_STATES):
            if P_matrix[i, j] > 0.05: # Limiar para não poluir o grafo
                G.add_edge(i, j, weight=P_matrix[i, j], label=f"{P_matrix[i, j]:.2f}")

    # Tentar usar st.graphviz_chart
    dot_string = "digraph {\n"
    dot_string += 'node [shape=plaintext fontname="Helvetica"];\n' # Estilo para nós
    for node, data in G.nodes(data=True):
        dot_string += f'  {node} [label="{data["label"]}"];\n'
    for u, v, data in G.edges(data=True):
         dot_string += f'  {u} -> {v} [label="{data["label"]}", weight="{data["weight"]*10}"];\n' # weight para layout
    dot_string += "}"

    try:
        st.graphviz_chart(dot_string)
    except Exception as e:
        st.warning(f"Não foi possível gerar o grafo com Graphviz: {e}. Desenhando com Matplotlib (mais simples).")
        fig, ax = plt.subplots(figsize=(15, 15))
        pos = nx.spring_layout(G, k=0.5, iterations=20) # Layout
        labels = nx.get_node_attributes(G, 'label')
        edge_labels = nx.get_edge_attributes(G, 'label')
        nx.draw(G, pos, with_labels=True, labels=labels, node_size=3000, node_color="skyblue", font_size=8, ax=ax)
        nx.draw_networkx_edge_labels(G, pos, edge_labels=edge_labels, font_color='red', ax=ax)
        ax.set_title(f"Grafo de Transição (Matplotlib): {trajectory_name}")
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

# Para comparação
st.sidebar.header("Comparação de Trajetórias")
compare_mode = st.sidebar.checkbox("Ativar modo de comparação")
selected_trajectory_name_2 = None
if compare_mode:
    available_for_compare = [t for t in trajectories_options.keys() if t != selected_trajectory_name]
    if available_for_compare:
        selected_trajectory_name_2 = st.sidebar.selectbox(
            "Escolha a segunda trajetória para comparar:",
            available_for_compare
        )
    else:
        st.sidebar.warning("Apenas uma trajetória disponível para seleção.")


# --- Execução e Exibição ---
if st.sidebar.button("🚀 Rodar Simulação"):
    initial_state_idx = trajectories_options[selected_trajectory_name]
    
    # Para simplificação, P_base é usada para todos os anos.
    # Para não-homogêneo, P_base seria uma lista de 10 matrizes.
    P_base_1 = get_base_transition_matrix(selected_trajectory_name)
    
    st.header(f"Resultados para: {selected_trajectory_name}")
    all_paths_1, all_incomes_1 = run_simulation(initial_state_idx, P_base_1)

    # Abas para organizar os resultados
    tab1, tab2, tab3, tab4 = st.tabs(["📈 Curva de Renda", "📊 Distribuição Final", "👣 Caminhos Exemplo", "🕸️ Grafo de Transição"])

    with tab1:
        plot_expected_income(all_incomes_1, selected_trajectory_name)
    with tab2:
        plot_final_state_distribution(all_paths_1, selected_trajectory_name)
    with tab3:
        display_sample_paths(all_paths_1)
    with tab4:
        plot_transition_graph(P_base_1, selected_trajectory_name)

    # Frequência de Sucesso Elevado
    final_states_1 = all_paths_1[:, -1]
    success_freq_1 = np.sum(final_states_1 == 15) / N_SIMULATIONS
    st.subheader(f"🌟 Frequência de 'Sucesso Elevado' ({selected_trajectory_name}): {success_freq_1:.2%}")
    st.markdown(f"O estado 'Sucesso Elevado' (Renda: {df_states.loc[15, 'Renda']}) representa uma ascensão profissional significativa. Esta frequência indica a proporção de simulações que alcançaram este estado após 10 anos.")

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
            # Para plotar lado a lado ou sobreposto, precisa de mais ajuste com Matplotlib
            # Por simplicidade, vamos mostrar um após o outro ou criar um gráfico de barras agrupado
            
            # Criando dados para gráfico de barras agrupado
            final_counts_1 = pd.Series(all_paths_1[:, -1]).value_counts(normalize=True).rename(selected_trajectory_name)
            final_counts_2 = pd.Series(all_paths_2[:, -1]).value_counts(normalize=True).rename(selected_trajectory_name_2)
            
            df_compare_final = pd.concat([final_counts_1, final_counts_2], axis=1).fillna(0)
            df_compare_final.index = df_states.loc[df_compare_final.index, "Nome"] # Nomes dos estados no índice

            fig_comp_dist, ax_comp_dist = plt.subplots(figsize=(12,7))
            df_compare_final.plot(kind='bar', ax=ax_comp_dist, width=0.8)
            ax_comp_dist.set_xlabel("Estado Final")
            ax_comp_dist.set_ylabel("Proporção de Agentes")
            ax_comp_dist.set_title("Comparativo: Distribuição Final de Estados")
            plt.xticks(rotation=45, ha="right")
            plt.tight_layout()
            st.pyplot(fig_comp_dist)

else:
    st.info("Escolha uma trajetória na barra lateral e clique em 'Rodar Simulação'.")

st.sidebar.markdown("---")
st.sidebar.markdown("""
**Sobre o Modelo:**
- **Estados:** Representam sua condição profissional/educacional.
- **Cadeia de Markov:** Um modelo que descreve sequências de eventos possíveis onde a probabilidade de cada evento depende apenas do estado atual.
- **Não-Homogênea (Simplificado):** Idealmente, as probabilidades de transição mudariam a cada ano. Nesta simulação, a matriz de transição base de uma trajetória é a mesma ao longo dos 10 anos para simplificar, mas a estrutura do código `run_simulation` permitiria matrizes anuais diferentes.
- **Dados:** As rendas são baseadas na descrição. As probabilidades de transição são **ilustrativas** e podem ser ajustadas para refletir dados reais.
""")
st.sidebar.markdown("---")
st.sidebar.markdown("Desenvolvido como exemplo educacional.")

