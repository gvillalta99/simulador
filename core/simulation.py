"""
Core simulation engine for the Career Path Simulator.

This module contains functions for:
- Generating and managing transition matrices.
- Running the core Markov chain simulation.
- Calculating state-dependent attributes like salary growth categories.
"""
import numpy as np
import pandas as pd
# N_YEARS might be used by run_simulation if not passed directly,
# and other config variables might be needed by functions if they were not passed as arguments.
# For now, N_YEARS is passed to run_simulation from Simulador.py
# from core.config import N_YEARS

def get_default_base_transition_matrix(trajectory_name: str, n_total_states: int) -> np.ndarray:
    """
    Generates a default base transition matrix for a given trajectory.

    Args:
        trajectory_name: The name of the career trajectory.
        n_total_states: The total number of possible states.

    Returns:
        A numpy array representing the transition matrix.
    """
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

    prob_sucesso = {5: 0.05, 8: 0.02, 9: 0.03, 10: 0.01, 11: 0.02, 12: 0.001, 13: 0.002, 14: 0.005, 16: 0.005}
    for estado_origem, chance in prob_sucesso.items():
        if estado_origem < n_total_states and 15 < n_total_states: # Check bounds for state 15
             P[estado_origem, 15] = max(P[estado_origem, 15], chance)

    for i in range(n_total_states):
        current_P_ii = P[i,i]
        P[i,i] = 0
        sum_off_diagonal = np.sum(P[i, :])

        if sum_off_diagonal >= 1.0:
            if sum_off_diagonal > 0:
                P[i, :] = P[i, :] / sum_off_diagonal
            P[i,i] = 0.0
        else:
            P[i,i] = 1.0 - sum_off_diagonal

        final_row_sum = np.sum(P[i, :])
        if not np.isclose(final_row_sum, 1.0):
            if final_row_sum > 0: P[i, :] = P[i, :] / final_row_sum
            else: P[i,i] = 1.0 # Fallback for all-zero rows after potential negative clipping
    return P

def get_matrix_for_simulation(
    trajectory_name: str,
    n_total_states: int,
    custom_transition_matrices: dict,
    default_matrix_provider_func=get_default_base_transition_matrix
) -> np.ndarray:
    """
    Retrieves the appropriate transition matrix for simulation.
    Returns a custom matrix if available, otherwise falls back to the default.

    Args:
        trajectory_name: The name of the career trajectory.
        n_total_states: The total number of possible states.
        custom_transition_matrices: A dictionary holding any user-defined matrices.
        default_matrix_provider_func: Function to call to get a default matrix.

    Returns:
        A numpy array representing the transition matrix.
    """
    if trajectory_name in custom_transition_matrices:
        custom_P = custom_transition_matrices[trajectory_name]
        if isinstance(custom_P, np.ndarray) and custom_P.shape == (n_total_states, n_total_states):
            return custom_P.copy()
    return default_matrix_provider_func(trajectory_name, n_total_states)

def normalize_matrix(matrix_df: pd.DataFrame) -> pd.DataFrame:
    """
    Normalizes a DataFrame (representing a transition matrix P) so that rows sum to 1.
    Negative values are clipped to 0 before normalization.

    Args:
        matrix_df: A pandas DataFrame representing the matrix.

    Returns:
        A pandas DataFrame with rows normalized.
    """
    P_array = matrix_df.to_numpy(dtype=float)
    P_array[P_array < 0] = 0

    for i in range(P_array.shape[0]):
        row_sum = np.sum(P_array[i, :])
        if row_sum > 0:
            P_array[i, :] = P_array[i, :] / row_sum
        else:
            P_array[i, :] = 0.0 # Set row to zeros
            if i < P_array.shape[1]: # Check bounds
                 P_array[i, i] = 1.0 # Set diagonal to 1 as a fallback
    return pd.DataFrame(P_array, index=matrix_df.index, columns=matrix_df.columns)

def get_state_category_for_growth(state_id: int, df_states_info: pd.DataFrame) -> str:
    """
    Determines the salary growth category for a given state.

    Args:
        state_id: The ID of the state.
        df_states_info: DataFrame containing information about each state, including "Nome".

    Returns:
        A string representing the growth category.
    """
    if state_id not in df_states_info.index: return "sem_crescimento" # Guard clause
    nome_estado = df_states_info.loc[state_id, "Nome"].lower()
    if state_id in [10, 11]: # Grande empresa, Empresa global
        return "grande_empresa_ti"
    elif state_id in [0, 2, 3, 8, 9]: # Técnico, Fac. Comp. Trab., Pequena Emp., Startup
        return "pequena_empresa_startup_ti"
    elif state_id in [12, 13, 14]: # Serviço Público Mun, Est, Fed
        return "servico_publico"
    elif state_id == 16: # Trabalhar em outra área (não TI)
        return "outra_area"
    elif state_id == 4: # Outra faculdade (trabalhando)
         return "outra_area"
    return "sem_crescimento"

def state_allows_growth(state_id: int, df_states_info: pd.DataFrame) -> bool:
    """
    Checks if a given state allows for salary growth.

    Args:
        state_id: The ID of the state.
        df_states_info: DataFrame containing information about each state.

    Returns:
        True if the state allows growth, False otherwise.
    """
    return get_state_category_for_growth(state_id, df_states_info) != "sem_crescimento"

def is_promotion(previous_state_id: int, current_state_id: int, df_states_info: pd.DataFrame) -> bool:
    """
    Determines if a transition between two states constitutes a promotion.

    Args:
        previous_state_id: The ID of the previous state.
        current_state_id: The ID of the current state.
        df_states_info: DataFrame containing information about each state.

    Returns:
        True if the transition is a promotion, False otherwise.
    """
    if not state_allows_growth(previous_state_id, df_states_info) or \
       not state_allows_growth(current_state_id, df_states_info):
        return False
    if previous_state_id in [0,8,9] and current_state_id in [10,11]: return True # Tech/Small/Startup to Big/Global
    if previous_state_id == 12 and current_state_id in [13,14]: return True # Municipal to State/Federal Public
    if previous_state_id == 13 and current_state_id == 14: return True # State to Federal Public
    # Comp Sci student/grad working to more senior roles
    if previous_state_id in [0,2,3] and current_state_id in [8,9,10,11]: return True
    return False

def run_simulation(
    initial_state_idx: int,
    base_P: np.ndarray,
    current_states_df: pd.DataFrame,
    n_simulations_run: int,
    n_years_run: int, # N_YEARS from config will be passed here
    n_total_states_run: int,
    growth_configs: dict
) -> tuple[np.ndarray, np.ndarray]:
    """
    Runs the career path simulation.

    Args:
        initial_state_idx: The starting state index for all simulations.
        base_P: The base transition matrix.
        current_states_df: DataFrame with current state information (Nome, Renda, etc.).
        n_simulations_run: The number of individual simulations (agents) to run.
        n_years_run: The number of years to simulate for each agent.
        n_total_states_run: The total number of states in the simulation.
        growth_configs: Dictionary containing parameters for salary growth.

    Returns:
        A tuple containing:
        - all_paths (np.ndarray): An array of shape (n_simulations_run, n_years_run + 1)
                                  storing the state path for each simulation.
        - all_incomes (np.ndarray): An array of shape (n_simulations_run, n_years_run + 1)
                                   storing the income path for each simulation.
    """
    all_paths = np.zeros((n_simulations_run, n_years_run + 1), dtype=int)
    all_incomes = np.zeros((n_simulations_run, n_years_run + 1))

    for sim in range(n_simulations_run):
        agent_vars = {
            'salario': current_states_df.loc[initial_state_idx, "Renda"],
            'exp_TI': 0,
            'exp_outra_area': 0,
            'desemp_acum': 0,
            'fora_TI_acum': 0,
            'anos_no_estado_cont': 0
        }

        current_agent_state = initial_state_idx
        all_paths[sim, 0] = current_agent_state
        all_incomes[sim, 0] = agent_vars['salario']

        previous_state_for_counter = -1

        for year_idx in range(n_years_run):
            state_spent_this_year = all_paths[sim, year_idx]
            salary_at_start_of_this_year = all_incomes[sim, year_idx]

            if state_spent_this_year == previous_state_for_counter:
                agent_vars['anos_no_estado_cont'] += 1
            else:
                agent_vars['anos_no_estado_cont'] = 1
            previous_state_for_counter = state_spent_this_year

            categoria_estado_atual = get_state_category_for_growth(state_spent_this_year, current_states_df)

            if categoria_estado_atual in ["grande_empresa_ti", "pequena_empresa_startup_ti"]:
                agent_vars['exp_TI'] += 1
            elif categoria_estado_atual == "outra_area":
                agent_vars['exp_outra_area'] += 1

            if state_spent_this_year == 7: # Desempregado
                agent_vars['desemp_acum'] += 1

            if categoria_estado_atual not in ["grande_empresa_ti", "pequena_empresa_startup_ti"] and \
               state_spent_this_year not in [1,6,7]:
                agent_vars['fora_TI_acum'] +=1

            probabilities = base_P[state_spent_this_year, :]
            # Ensure probabilities sum to 1, handling potential floating point issues or bad rows
            if not np.isclose(np.sum(probabilities), 1.0):
                if np.sum(probabilities) <= 0:
                    probabilities = np.zeros(n_total_states_run)
                    probabilities[state_spent_this_year] = 1.0 # Stay in current state if row is invalid
                else:
                    probabilities = probabilities / np.sum(probabilities) # Normalize

            next_state = np.random.choice(n_total_states_run, p=probabilities)
            all_paths[sim, year_idx + 1] = next_state

            new_salary_for_next_year = 0.0
            piso_salarial_next_state = current_states_df.loc[next_state, "Renda"]

            if state_allows_growth(next_state, current_states_df):
                if next_state == state_spent_this_year: # Same job
                    base_growth_rate = growth_configs.get(get_state_category_for_growth(state_spent_this_year, current_states_df), 0.0)
                    experience_bonus_rate = 0.0
                    if agent_vars['anos_no_estado_cont'] > 0 and \
                       agent_vars['anos_no_estado_cont'] % growth_configs.get('bonus_experiencia_marco_anos', 3) == 0:
                        experience_bonus_rate = growth_configs.get('bonus_experiencia_valor_pc', 0.005)
                    new_salary_for_next_year = salary_at_start_of_this_year * (1 + base_growth_rate + experience_bonus_rate)
                else: # Changed job
                    salary_after_potential_promotion_bump = salary_at_start_of_this_year
                    if state_allows_growth(state_spent_this_year, current_states_df) and \
                       is_promotion(state_spent_this_year, next_state, current_states_df):
                        salary_after_potential_promotion_bump = salary_at_start_of_this_year * (1 + growth_configs.get('aumento_promocao_pc', 0.10))

                    new_salary_for_next_year = max(salary_after_potential_promotion_bump, piso_salarial_next_state)
                    if not is_promotion(state_spent_this_year, next_state, current_states_df) and \
                        state_allows_growth(state_spent_this_year, current_states_df) :
                        new_salary_for_next_year = max(salary_at_start_of_this_year, piso_salarial_next_state)
            else:
                new_salary_for_next_year = piso_salarial_next_state

            agent_vars['salario'] = new_salary_for_next_year
            all_incomes[sim, year_idx + 1] = agent_vars['salario']

    return all_paths, all_incomes
