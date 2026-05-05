# FIA - Lunar Lander Meta 2

## Autores

- André Raposo — 2023234762
- Francisco Martins — 2023211141
- Paulo Vilar — 2023223817

## Descrição

Neste trabalho, para controlar uma nave no pygame **Lunar Lander** com a melhor accuracy possível, implementámos um sistema de produções com diferentes perceções e ações, a partir das quais a nave vai agir quando certas condições forem atingidas. Assim, controlamos a queda da nave para aterrar com a maior frequência possível na sua landing area proposta.

## 1. Perceções

O agente usa perceções de dois tipos: **perceções base** (diretas da observação) e **perceções derivadas** (cálculos que ajudam a decisão).

### 1.1 Perceções base

A observação (`observation` no trabalho) guarda 8 valores diferentes, dos quais o agente usa:

- `x`: posição horizontal
- `y`: altura
- `vx`: velocidade horizontal
- `vy`: velocidade vertical
- `th`: ângulo da nave
- `vth`: velocidade angular
- contacto da perna esquerda
- contacto da perna direita

Estas perceções são usadas por várias funções ao longo do trabalho todo.

### 1.2 Perceções de contacto

- `both_legs_touching`: verdadeiro quando as duas pernas tocam no solo
- `one_leg_touching`: verdadeiro quando só uma perna toca

Assim, conseguimos perceber qual o tipo de queda quando a nave aterra, para depois podermos concluir se foi uma queda válida ou não.

### 1.3 Perceção de banda de altitude

A altura é discretizada em quatro opções:

- `very_high`: `y > 1.0`
- `high`: `0.6 < y <= 1.0`
- `medium`: `0.3 < y <= 0.6`
- `low`: `y <= 0.3`

Isto permite ao agente mudar a sua agressividade com a altura. Em cima pode corrigir mais, mas perto do solo deve ser mais suave para evitar oscilações.

### 1.4 Perceção de previsão lateral

A previsão lateral estima onde a nave vai estar horizontalmente num horizonte curto:

```text
x_future = x + vx * horizon
```

O `horizon` depende da altura.

Em altitude alta, o agente olha mais “para a frente” para corrigir cedo; perto do solo, olha menos para evitar correções bruscas.

Existe ainda reforço de correção:

- Se `|x| > 0.22` e `x` e `vx` têm o mesmo sinal (a nave está a afastar-se do centro), soma-se:

```text
np.sign(x) * 0.16 * min(abs(vx), 1.0)
```

Isto aumenta a urgência da correção quando há deriva persistente para fora da zona de aterragem.

### 1.5 Perceção de pressão lateral

```text
abs(lateral_prediction(obs)) + 0.95 * abs(x_vel(obs))
```

Esta perceção combina uma previsão da posição lateral futura com a velocidade lateral atual. Quanto maior este valor, mais o agente deve travar a descida para se recentrar primeiro.

### 1.6 Perceções de travão vertical

- `emergency_fall`: `vy < -1.15` e `|th| < 0.55`
- `heavy_fall`: `vy < -0.90` e `|th| < 0.45`

Também criámos estas perceções para controlar o travão de queda quando seja necessário, de modo a aumentar a accuracy da nave.

## 2. Ações

O ambiente contínuo usa ação com dois componentes:

1. Motor principal (valores variam entre 0 e 1)
2. Controlo lateral/rotação (de -1 a 1)

### 2.1 Ações discretas de apoio

Temos implementadas algumas ações base para permitir controlar a nave:

- `do_nothing` (no caso de ambas as pernas estarem a tocar na landing area e não ser preciso fazer nada)
- `full_thrust` (para ser usado para `emergency_fall`)
- `strong_thrust` (para `heavy_fall`)
- rotações com thrust

### 2.2 Ação contínua final

A ação final é construída por:

- `thrust` (após regras verticais)
- `rotation` (após regras angulares)
- clipping para intervalos válidos:
  - `thrust` em `[0, 1]`
  - `rotation` em `[-1, 1]`

## 3. Sistema de Produções

Agora já temos o nosso sistema de produções dentro do `reactive_agent`, que vai verificar se certas condições ou perceções se aplicam, para decidir que ação escolher para melhor controlar a nave.

### 3.1 Verificação de se as pernas tocam no chão

Se as duas pernas estão a tocar no chão, não fazemos nada.

### 3.2 Travagem vertical de emergência

- Se for o caso de `emergency_fall`, usamos o `full_thrust`
- Se for uma `heavy_fall`, usamos o `strong_thrust` para controlar

Assim, usamos um travão de queda para controlar melhor as descidas e garantir que a nave cai seguramente na zona de queda.

### 3.3 Definição do ângulo-alvo

Para isto, seguimos alguns passos:

1. Calculamos `x_future` com previsão lateral
2. Escolhemos a inclinação preferida por grupo de altura
3. Usamos uma regra extra de aproximação final:
   - Se `y < 0.30` e ainda estiver fora da landing area, o agente força recentragem perto do solo
4. Se só uma perna estiver a tocar no chão, o agente vai-se endireitar para não cair

### 3.4 Controlo angular

Calculamos o erro angular (`th - target_th`).

- `rot_raw` simboliza a intensidade (não “normalizada”) do quanto queremos fazer rotação da nave
- `rot_cmd` é o valor “normalizado” para o input do nosso agente, calculado com base na intensidade não normalizada (entre -1 e 1)

Perto do solo (`y < 0.20`) multiplica por `1.05`, para dar um pequeno “boost” lateral na fase final, porque é precisamente aí que o vento costuma empurrar o agente para fora da zona de aterragem.

### 3.5 Objetivo vertical adaptativo

A velocidade vertical alvo varia consoante a altura:

> Quanto mais perto do chão, menor deverá ser a minha velocidade.

Depois, o agente adapta o alvo com base na pressão lateral e nas suas pernas.

Assim:

- se `y < 0.35` e erro lateral forte (`|x| > 0.12` ou `|vx| > 0.12`), parar de descer (`vy_target >= 0.00`)
- se erro moderado (`|x| > 0.07` ou `|vx| > 0.08`), descer muito devagar (`vy_target >= -0.03`)
- se muito perto do chão (`y < 0.22`) e ainda fora da área de pouso (`|x| > 0.09` ou `|vx| > 0.10`), forçar `vy_target >= 0.02` (mini subida para recentrar)

### 3.6 Conversão erro vertical

A variável `excess` representa o quanto o agente está a descer mais rápido do que devia.

Se `excess` for grande, o `thrust` é maior.

Finalmente, fazemos proteção de impacto para a nave cair de forma mais suave:

- se `y < 0.18` e `vy < -0.10`, `thrust` mínimo de `0.32`

## 4. Conclusões e resultados

O agente reativo registou uma taxa de sucesso sem vento de aproximadamente **96–98%**, e de cerca de **55–61%** com vento.

Para ambos os casos, usámos o mesmo agente reativo, sendo que as ações, perceções e sistema de produções que implementámos funcionam para ambos os cenários, controlando a nave de uma forma estável e com boa eficácia.
