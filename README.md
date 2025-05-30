# Simulador de Caminhos Profissionais com Histórico Individual

## Objetivo do Projeto

Olá! Seja bem-vindo(a) ao Simulador de Caminhos Profissionais. Sabemos que
pensar sobre o futuro, especialmente sobre carreira e finanças, pode ser um
grande desafio. São muitas opções, incertezas e "e se..." pipocando na cabeça,
não é mesmo?

Este simulador foi criado para te ajudar a:

1. **Explorar** diferentes trajetórias de vida e trabalho ao longo de 10 anos.
2. **Visualizar** o impacto potencial de escolhas educacionais e profissionais
na sua renda e progressão de carreira.
3. **Compreender** conceitos como risco, retorno, e como decisões de curto
prazo podem influenciar o longo prazo.
4. **Experimentar** com diferentes cenários, ajustando salários, probabilidades
e até mesmo taxas de crescimento, para ver como o futuro poderia se desenhar.

A ideia não é prever o futuro (isso é impossível!), mas sim oferecer uma
ferramenta para refletir, aprender sobre modelagem e discutir as diversas
variáveis que influenciam nossa vida profissional.

## Como Funciona? Uma Visão Geral

O simulador utiliza uma combinação de conceitos para criar essas projeções:

1. **Estados:** Representam sua situação profissional e educacional em um
determinado ano (ex: "Técnico Trabalhando", "Fazendo Faculdade e Trabalhando na
Área", "Desempregado", "Empresa Global na Área", etc.). Atualmente, temos 17
estados definidos.
2. **Trajetórias Iniciais:** Você começa escolhendo um "plano de vida" inicial,
como "Cursar Faculdade de Computação e trabalhar na área" ou "Focar em ser
Técnico sem fazer faculdade". São 6 opções iniciais.
3. **Simulação (Monte Carlo):** Para cada trajetória escolhida, o simulador
cria muitos "agentes" virtuais (por exemplo, 1000) e acompanha cada um deles por
10 anos. A cada ano, cada agente pode mudar de estado.
4. **O Modelo por Trás da Mágica:**
    * **Base em Cadeias de Markov (Personalizável):** A "tendência" de um agente
    mudar de um estado para outro é definida por uma **matriz de
    probabilidades de transição**. Se um agente está no "Estado A", qual a
    chance dele ir para o "Estado B", "Estado C" ou permanecer no "Estado A"?
    Essas probabilidades base podem ser visualizadas e **editadas por você**
    na página "Configurações"!
    * **Histórico Individual do Agente (A Novidade!):** Aqui o simulador vai
    além de uma Cadeia de Markov simples. Para cada agente, nós rastreamos:
        * **Salário Individual:** Seu salário não é fixo apenas pelo estado, ele
        progride individualmente!
        * **Anos de Experiência na Área de TI:** Quanto tempo trabalhando
        efetivamente com TI.
        * **Anos de Experiência em Outra Área:** Tempo trabalhando em áreas não
        relacionadas à TI.
        * **Anos Desempregado (Acumulado):** Total de tempo que passou
        desempregado.
        * **Anos Fora da Área de TI (Acumulado):** Tempo em empregos ou
        situações fora do foco principal em TI.
        * **Anos Contínuos no Estado Atual:** Quanto tempo está na mesma
        "posição" ou empresa.
    * **Impacto do Histórico (Fase 1 Implementada):** Atualmente, esse histórico
    individual influencia principalmente a **progressão salarial**. Por
    exemplo, seu salário pode crescer mais rápido em uma grande empresa do que
    em uma pequena, e bônus podem ser aplicados por tempo de casa ou
    experiência acumulada.

5. **Tecnologias Utilizadas:**
    * **Python:** A linguagem de programação por trás de tudo.
    * **Streamlit:** Para criar a interface web interativa que você está usando.
    * **NumPy & Pandas:** Para cálculos numéricos e manipulação de dados (como
    as matrizes e os resultados das simulações).
    * **Matplotlib & NetworkX:** Para gerar os gráficos de curvas de renda,
    distribuição de estados e o grafo de transições.

## Detalhes da Modelagem e Premissas Adotadas

Entender como o simulador "pensa" é fundamental para interpretar os resultados.

### 1. Estados Profissionais e Educacionais

São 17 estados que tentam cobrir uma gama de situações comuns para quem está
começando na área de informática ou em transição:

* **Estados de Entrada:** Situações iniciais como estar cursando um técnico, uma
faculdade (trabalhando ou não), empreendendo com baixo capital, ou
infelizmente, não estudando nem trabalhando ou desempregado.

* **Estados de Progresso:** Representam evolução na carreira, como conseguir um
emprego em uma pequena, média ou grande empresa da área, startups, empresas
globais, ou seguir carreira no serviço público (municipal, estadual, federal).
Inclui também o novo estado "Trabalhar em outra área (não TI)".

* **Estado de Sucesso Extremo:** Um estado de "Sucesso Elevado" com renda
significativamente alta, representando casos de grande destaque profissional
ou empreendedor.

*Premissa:* Alguns estados de progresso mais avançado (como "Grande Empresa" ou
*"Sucesso Elevado") não podem ser escolhidos como ponto de partida; eles são
*alcançados por meio da progressão ao longo do tempo.

### 2. Trajetórias Iniciais

As 6 trajetórias iniciais definem o estado em que o agente começa a simulação no
Ano 0. Cada uma tem uma "vocação" ou tendência inicial, que se reflete nas
probabilidades base de transição.

### 3. Matrizes de Probabilidade de Transição

* Para cada uma das 6 trajetórias iniciais, existe uma **matriz de probabilidade
base (padrão)**. Essa matriz (17x17) define, para cada estado atual (linha), a
probabilidade de transição para cada um dos outros estados (coluna) no próximo
ano.

* **Você Pode Editar!** Na página "⚙️ Configurações", você pode visualizar e
alterar essas probabilidades para qualquer uma das trajetórias base. Isso
permite que você crie cenários mais otimistas, pessimistas ou simplesmente
diferentes.

* **Normalização:** É crucial que a soma das probabilidades em cada LINHA da
matriz seja sempre igual a 1 (ou 100%). Se você editar uma matriz e ela não
estiver normalizada, o simulador tentará normalizá-la automaticamente ao
salvar.

* **Premissa (Fase 1):** Atualmente, essa matriz de transição (seja a padrão ou
a sua customizada) é usada para decidir a mudança de estado. O histórico
individual do agente (experiência, etc.) **ainda não influencia diretamente
qual estado ele irá em seguida**, mas isso é um desenvolvimento futuro
planejado (Fase 2)!

### 4. Progressão Salarial Individual

Esta é uma das partes mais dinâmicas do novo modelo!

* **Salário Individualizado:** Cada agente tem seu próprio salário, que começa
com o valor base do seu estado inicial (configurável na página
"Configurações").
* **Crescimento Anual:** Se um agente está em um estado de trabalho que permite
crescimento (ex: não está desempregado ou apenas estudando sem renda), seu
salário é ajustado anualmente.
* **Taxas de Crescimento Diferenciadas:** Você pode configurar, na página
"Configurações", taxas de crescimento salarial anuais diferentes para:
  * Grandes Empresas de TI
  * Pequenas Empresas e Startups de TI
  * Serviço Público
  * Trabalhar em Outra Área (não TI)
* **Bônus por Experiência/Tempo de Casa:** Um bônus percentual pode ser
adicionado ao crescimento salarial quando o agente atinge um certo número de
anos contínuos no mesmo estado (configurável, ex: +0.5% a cada 3 anos).
* **Aumento por Promoção:** Ao transitar para um estado que é considerado uma
"promoção" (ex: de Pequena para Grande Empresa), o agente pode receber um
aumento percentual sobre seu salário anterior (configurável). O salário no
novo estado será, no mínimo, o piso daquele novo estado.
* **Premissa:** As taxas de crescimento, bônus e aumentos são aplicados conforme
configurados e seguem uma lógica determinística uma vez que o agente está em
um estado ou faz uma transição. Não há variação aleatória *nesses percentuais*
em si, apenas na trajetória de estados que o agente segue.

### 5. Histórico Individual do Agente

Conforme mencionado, rastreamos:

* `anos_experiencia_area_TI`
* `anos_experiencia_outra_area`
* `anos_desempregado_acumulado`
* `anos_fora_area_TI_acumulado`
* `anos_no_estado_atual_continuo`

* **Impacto Atual (Fase 1):**
  * `anos_no_estado_atual_continuo` é usado para o "Bônus por Experiência/Tempo
  de Casa" no cálculo do salário.
  * Os outros contadores são atualizados e armazenados. Eles são importantes
  para a **Fase 2** do desenvolvimento do simulador.
* **Visão de Futuro (Fase 2 - Não implementado completamente ainda):** A ideia é
que, no futuro, esses contadores influenciem diretamente as **probabilidades
de transição**. Por exemplo:
  * Muitos anos de experiência em TI poderiam aumentar a chance de promoção.
  * Muito tempo desempregado poderia dificultar a obtenção de um emprego de alta
  qualificação.
  * Muito tempo fora da área de TI poderia reduzir a chance de voltar para um
  cargo sênior em TI.

### 6. Premissas Fundamentais Adicionais

* **Simplificação da Realidade:** O simulador é uma simplificação. Fatores como
networking, sorte, habilidades interpessoais, crises econômicas, saúde
pessoal, etc., não são explicitamente modelados (embora alguns possam ser
implicitamente representados nas probabilidades).
* **Foco em 10 Anos:** A simulação olha para um horizonte de uma década.
* **Dados Configuráveis:** Você tem controle sobre os salários base, as
probabilidades de transição e as taxas de crescimento, o que permite testar
diferentes hipóteses.
* **Interpretação dos Resultados:** Lembre-se que os resultados são
estatísticos, baseados em muitas simulações. A "Curva de Renda Esperada" é uma
média, e a "Distribuição de Renda Final" mostra a variedade de resultados
possíveis.

## 🚀 Como Usar o Simulador

1. **Navegação:**
    * Use o menu na barra lateral para alternar entre "🚀 Simulador" e "⚙️
    Configurações".
2. **Página "⚙️ Configurações":**
    * **Salários:** Edite a renda mensal base para cada um dos 17 estados.
    * **Taxas de Crescimento Salarial:** Defina as taxas anuais de crescimento
    para diferentes categorias de emprego, bônus por tempo de casa/experiência
    e aumento por promoção.
    * **Matrizes de Probabilidade:** Selecione uma das 6 trajetórias base e
    edite sua matriz de transição 17x17. Lembre-se de que as linhas devem
    somar 1 (o sistema tentará normalizar ao salvar). Você pode restaurar a
    matriz padrão a qualquer momento.
3. **Página "🚀 Simulador":**
    * **Configurações da Simulação (Sidebar):**
        * Escolha a "Trajetória 1" (seu cenário principal).
        * Defina o "Limiar de Renda Boa" para análise.
        * Escolha o "Número de Simulações" (agentes). Mais simulações dão
        resultados mais estáveis, mas demoram mais.
        * Ative o "Modo de Comparação" e escolha uma "Trajetória 2" se quiser
        comparar dois cenários.
    * Clique em "📊 Rodar Simulação Agora".
4. **Analisando os Resultados:**
    * Explore as abas para ver:
        * Curva de Renda Média ao longo dos 10 anos.
        * Distribuição final dos agentes pelos estados.
        * Distribuição da Renda Final (histograma dos salários no 10º ano).
        * Exemplos de caminhos individuais.
        * Grafo de transição (visualização das probabilidades).
        * Análise de "Renda Boa" (chance de atingir seu limiar, chance de
        "Sucesso Elevado", etc.).
    * Se o modo de comparação estiver ativo, um novo conjunto de abas aparecerá
    para comparar os resultados das duas trajetórias escolhidas.

## 🎓 Valor Educacional

Esperamos que este simulador te ajude a:

* Desenvolver o pensamento estatístico e a compreensão de probabilidades.
* Entender como modelos matemáticos podem ser usados para simular sistemas
complexos.
* Refletir sobre planejamento de carreira e o impacto de decisões de longo
prazo.
* Ver como diferentes fatores (experiência, tipo de empresa, tempo desempregado)
podem interagir.
* Apreciar tanto as capacidades quanto as limitações de um modelo de simulação.
* Discutir sobre oportunidades, meritocracia, risco e retorno de forma mais
embasada.

## ⚠️ Limitações Importantes

* **Simplificação:** A vida real é muito mais complexa!
* **Qualidade das Probabilidades:** As probabilidades de transição (padrão ou as
que você define) são o coração do modelo. Se elas não forem realistas, os
resultados também não serão ("Garbage In, Garbage Out").
* **Fatores Não Modelados:** Eventos macroeconômicos, sorte, contatos
(networking), habilidades pessoais específicas, etc., não estão diretamente no
modelo.
* **Impacto do Histórico nas Transições (Fase 2):** A funcionalidade de o
histórico do agente (experiência, etc.) alterar *dinamicamente* as
probabilidades de ele mudar de estado ainda é um desenvolvimento futuro.
* Atualmente, o histórico afeta principalmente o salário.

## 💡 Ideias para o Futuro (e para você pensar!)

* Implementar a "Fase 2": fazer com que `anos_experiencia_area_TI`,
`anos_desempregado_acumulado`, etc., alterem dinamicamente as probabilidades
de transição de estado para cada agente.
* Adicionar mais estados ou maior granularidade (ex: diferentes níveis dentro de
"Grande Empresa").
* Modelar eventos aleatórios ("choques" positivos ou negativos na carreira).
* Incluir custos de vida ou endividamento.
* Permitir que os alunos criem suas próprias "categorias de crescimento
salarial" e as atribuam aos estados.

Divirta-se explorando e aprendendo!