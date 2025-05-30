"""
Configuration file for the Career Path Simulator.

This file stores initial data definitions, state labels, trajectory options,
and default growth configurations used throughout the application.
Separating this configuration data makes the main application logic cleaner
and easier to manage.
"""

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

# Nomes dos estados para rótulos de matriz
# Defined after initial_states_data as it depends on it.
STATE_LABELS = [f"{id}: {data['Nome'][:30]}..." for id, data in initial_states_data.items()]


trajectories_options = {
    "Técnico e não faz faculdade": 0,
    "Faculdade de computação + trabalha na área": 2,
    "Faculdade de computação + não trabalha na área": 1,
    "Empreender (baixo capital)": 5,
    "Faculdade outra área + trabalha": 4,
    "Não estuda nem trabalha": 6
}

default_growth_configs = {
    'grande_empresa_ti': 0.07,
    'pequena_empresa_startup_ti': 0.05,
    'servico_publico': 0.02,
    'outra_area': 0.03,
    'bonus_experiencia_marco_anos': 3,
    'bonus_experiencia_valor_pc': 0.005,
    'aumento_promocao_pc': 0.10
}
