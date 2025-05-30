"""
Main Streamlit application file for the Career Path Simulator.

This script orchestrates the user interface, manages session state,
and calls core logic functions from the `core` package to run simulations
and display results. It handles page navigation, user inputs, and the
layout of different views (Simulator, Configurations).
"""
import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import networkx as nx

from core.config import (
    initial_states_data,
    STATE_LABELS,
    trajectories_options,
    default_growth_configs,
    N_YEARS
)

# --- Configurações e Dados Iniciais ---
N_INITIAL_STATES = len(initial_states_data) # Deve ser 17


# --- INICIALIZAÇÃO GLOBAL DO SESSION STATE ---
if 'editable_salaries' not in st.session_state:
    st.session_state.editable_salaries = {
        state_id: data["Renda"] for state_id, data in initial_states_data.items()
    }
if 'custom_transition_matrices' not in st.session_state:
    st.session_state.custom_transition_matrices = {}

# default_growth_configs já é importado
if 'growth_configs' not in st.session_state:
    st.session_state.growth_configs = default_growth_configs.copy()
# --- FIM DA INICIALIZAÇÃO GLOBAL DO SESSION STATE ---

from core.simulation import (
    get_default_base_transition_matrix,
    get_matrix_for_simulation as get_matrix_for_simulation_core, # renamed to avoid conflict if we had a local one
    normalize_matrix as normalize_matrix_core, # renamed
    # get_state_category_for_growth, # These are used by run_simulation_core, no need to import here
    # state_allows_growth,
    # is_promotion,
    run_simulation as run_simulation_core # renamed
)

# --- Funções do Modelo de Markov ---
# Functions get_default_base_transition_matrix, get_matrix_for_simulation, normalize_matrix,
# get_state_category_for_growth, state_allows_growth, is_promotion, and run_simulation
# have been moved to core/simulation.py

# --- Wrappers for Core Logic with Streamlit Caching/UI Interaction ---

# Wrapper for run_simulation_core to apply caching
@st.cache_data
def run_simulation_cached(
    initial_state_idx, base_P, current_states_df,
    n_simulations_run, n_years_run, n_total_states_run,
    growth_configs
):
    """
    Cached wrapper for the core `run_simulation_core` function.
    Uses Streamlit's caching to avoid re-running simulations with the same inputs.

    Args:
        initial_state_idx: The starting state index.
        base_P: The base transition matrix.
        current_states_df: DataFrame with current state information.
        n_simulations_run: Number of simulations to run.
        n_years_run: Number of years to simulate.
        n_total_states_run: Total number of states.
        growth_configs: Dictionary of growth configurations.

    Returns:
        Tuple[np.ndarray, np.ndarray]: Result from `run_simulation_core`.
    """
    return run_simulation_core(
        initial_state_idx, base_P, current_states_df,
        n_simulations_run, n_years_run, n_total_states_run,
        growth_configs
    )

# Wrapper for normalize_matrix_core if it's used directly in Streamlit UI
# If normalize_matrix is only used by other functions in core.simulation, this isn't needed here.
# Based on current Simulador.py, normalize_matrix IS used in the UI.
def normalize_matrix(matrix_df: pd.DataFrame) -> pd.DataFrame: # This wrapper remains as it's used in UI
    """
    UI wrapper for the core `normalize_matrix_core` function.
    Ensures that matrix normalization logic called from the UI uses the
    centralized function from the simulation core.

    Args:
        matrix_df: Pandas DataFrame representing the matrix to be normalized.

    Returns:
        pd.DataFrame: The normalized matrix.
    """
    return normalize_matrix_core(matrix_df)

# --- UI Plotting Function Imports ---
from core.plotting import (
    plot_expected_income,
    plot_final_state_distribution,
    plot_final_income_distribution_hist,
    get_sample_paths_df, # Renamed from display_sample_paths
    plot_transition_graph_mpl,
    display_renda_boa_metrics
)

# --- Main Application Setup & UI Orchestration ---
st.set_page_config(layout="wide", page_title="Simulador de Carreiras")

# --- Global State Initialization (Derived from Config & Session State) ---
# Initialize editable salaries in session state if not already present
if 'editable_salaries' not in st.session_state: # This check was already here, kept for clarity
    st.session_state.editable_salaries = {
        state_id: data["Renda"] for state_id, data in initial_states_data.items()
    }

# DataFrame de estados dinâmico, construído com base nos salários editáveis da sessão
CURRENT_DF_STATES = pd.DataFrame.from_dict({
    s_id: {
        "Nome": data["Nome"],
        "Categoria": data["Categoria"],
        "Renda": st.session_state.editable_salaries.get(s_id, data["Renda"]) # Use .get for safety
    } for s_id, data in initial_states_data.items()
}, orient='index')
N_CURRENT_STATES = len(CURRENT_DF_STATES)

# --- Page Navigation ---
page = st.sidebar.radio("Navegar para:", ["🚀 Simulador", "⚙️ Configurações"], horizontal=True)

# --- Page Specific Logic ---
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
    matrix_to_edit_np = get_matrix_for_simulation_core(
        config_traj_name,
        N_CURRENT_STATES,
        st.session_state.custom_transition_matrices,
        get_default_base_transition_matrix # Pass the function from core.simulation
    )
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

    st.markdown("---")
    st.subheader("📈 Configurar Taxas de Crescimento Salarial e Bônus")
    st.caption("Estas taxas são anuais. Ex: 0.05 para 5%.")

    # default_growth_configs já é importado
    # if 'growth_configs' not in st.session_state: # Esta verificação já acontece mais acima globalmente
    #     st.session_state.growth_configs = default_growth_configs.copy()

    gc = st.session_state.growth_configs # Alias para facilitar
    cols_growth = st.columns(2)
    with cols_growth[0]:
        gc['grande_empresa_ti'] = st.number_input("Taxa Grande Empresa TI:", value=gc['grande_empresa_ti'], min_value=0.0, max_value=0.5, step=0.005, format="%.3f", key="gc_get")
        gc['pequena_empresa_startup_ti'] = st.number_input("Taxa Pequena/Startup TI:", value=gc['pequena_empresa_startup_ti'], min_value=0.0, max_value=0.5, step=0.005, format="%.3f", key="gc_pet")
        gc['servico_publico'] = st.number_input("Taxa Serviço Público:", value=gc['servico_publico'], min_value=0.0, max_value=0.5, step=0.005, format="%.3f", key="gc_sp")
        gc['outra_area'] = st.number_input("Taxa Outra Área:", value=gc['outra_area'], min_value=0.0, max_value=0.5, step=0.005, format="%.3f", key="gc_oa")
    with cols_growth[1]:
        gc['bonus_experiencia_marco_anos'] = st.number_input("Bônus Experiência: Marco (anos):", value=gc['bonus_experiencia_marco_anos'], min_value=1, max_value=10, step=1, key="gc_bema")
        gc['bonus_experiencia_valor_pc'] = st.number_input("Bônus Experiência: Valor (%):", value=gc['bonus_experiencia_valor_pc'], min_value=0.0, max_value=0.1, step=0.001, format="%.3f", key="gc_bevp")
        gc['aumento_promocao_pc'] = st.number_input("Aumento Percentual por Promoção (%):", value=gc['aumento_promocao_pc'], min_value=0.0, max_value=0.5, step=0.01, format="%.2f", key="gc_app")

    if st.button("Restaurar Taxas de Crescimento Padrão", key="reset_growth_configs"):
        st.session_state.growth_configs = default_growth_configs.copy()
        st.success("Taxas de crescimento padrão restauradas!")
        st.rerun()



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
        "O que você considera uma 'Renda Boa' (R$):", 
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
        df_states_for_sim = CURRENT_DF_STATES.copy()
        current_growth_configs = st.session_state.growth_configs.copy() # Pega as configs atuais
        
        # Simulação para Trajetória 1
        initial_state_idx_1 = trajectories_options[sidebar_traj1_name]
        P_base_1 = get_matrix_for_simulation_core(
            sidebar_traj1_name,
            N_CURRENT_STATES,
            st.session_state.custom_transition_matrices,
            get_default_base_transition_matrix # Pass the function from core.simulation
        )
        
        st.subheader(f"Resultados para Trajetória 1: {sidebar_traj1_name}")
        with st.spinner(f"Rodando {sidebar_n_simul} simulações para {sidebar_traj1_name}..."):
            all_paths_1, all_incomes_1 = run_simulation_cached( # Use the cached wrapper
                initial_state_idx_1, P_base_1, df_states_for_sim, 
                sidebar_n_simul, N_YEARS, N_CURRENT_STATES, # N_YEARS is from core.config
                current_growth_configs
            )

        tab_titles_1 = ["📈 Renda Média", "📊 Dist. Estados", "SAL Dist. Renda", "👣 Caminhos", "🕸️ Grafo", "🎯 Renda Boa"]
        tabs_1 = st.tabs(tab_titles_1)
        with tabs_1[0]: plot_expected_income([all_incomes_1], [sidebar_traj1_name])
        with tabs_1[1]: plot_final_state_distribution([all_paths_1], [sidebar_traj1_name], df_states_for_sim)
        with tabs_1[2]: plot_final_income_distribution_hist([all_incomes_1], [sidebar_traj1_name])
        with tabs_1[3]:
            st.subheader("Exemplos de Caminhos Individuais (Top 20)")
            sample_df_1 = get_sample_paths_df(all_paths_1, df_states_for_sim)
            st.dataframe(sample_df_1)
        with tabs_1[4]: plot_transition_graph_mpl(P_base_1, sidebar_traj1_name, df_states_for_sim, N_CURRENT_STATES)
        with tabs_1[5]: display_renda_boa_metrics(all_paths_1, all_incomes_1, sidebar_traj1_name, df_states_for_sim, sidebar_renda_boa_thresh, sidebar_n_simul)

        # Simulação e Abas para Comparação (se ativado)
        if sim_compare_mode and sidebar_traj2_name:
            initial_state_idx_2 = trajectories_options[sidebar_traj2_name]
            P_base_2 = get_matrix_for_simulation_core(
                sidebar_traj2_name,
                N_CURRENT_STATES,
                st.session_state.custom_transition_matrices,
                get_default_base_transition_matrix # Pass the function from core.simulation
            )
            
            st.markdown("---"); st.subheader(f"Comparação: {sidebar_traj1_name} vs {sidebar_traj2_name}")
            with st.spinner(f"Rodando {sidebar_n_simul} simulações para {sidebar_traj2_name}..."):
                all_paths_2, all_incomes_2 = run_simulation_cached( # Use the cached wrapper
                    initial_state_idx_2, P_base_2, df_states_for_sim, 
                    sidebar_n_simul, N_YEARS, N_CURRENT_STATES, # N_YEARS is from core.config
                    current_growth_configs
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