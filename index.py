import pandas as pd
import streamlit as st
from sklearn import tree
# INTRODUÇÃO:
# - TECNOLOGIAS (PANDAS, SKLEARN (tree))


st.title('Tópicos Especiais em Informática - Trabalho')

# carregando a base
database1= 'base-teste.csv' #caminho da base de dados
database3 = 'base-teste3.csv' #caminho da base de dados
df_estatistica= pd.read_csv(database1, sep=",") # dataset estatísticas
df_dados = pd.read_csv(database3, sep=",") # dataset dados full


#limpando base estatística (partidas que não aconteceram ou muito antigos (poucas informações))  2992 rows × 13 columns
df_estatistica= df_estatistica.dropna(subset=df_estatistica.select_dtypes(int).columns, how='all')
df_estatistica= df_estatistica[df_estatistica["chutes"] != 0]
df_estatistica= df_estatistica[df_estatistica["precisao_passes"] != "None"]

#criando arvore
quali = ['clube'] #Variáveis qualitativas
quant = ['chutes', 'chutes_no_alvo'] #Variáveis quantitativas
dfQualiDummies = pd.get_dummies(df_estatistica[quali]) #Dataframe com qualitativas dummy
dfQuant = df_estatistica[quant] #Dataframe com quantitativas
dfWork = pd.concat([dfQualiDummies, dfQuant ], axis=1 ) #Dataframe com quali dummy e quant
target = df_estatistica['posse_de_bola'] # variavel alvo da predição (string)
arv = tree.DecisionTreeClassifier() #árvore de classificação 
arv.fit(dfWork, target)

clubes_df = ["America-MG","Athletico-PR","Atletico-GO","Atletico-MG","Avai","Bahia","Botafogo-RJ","Bragantino","CSA","Ceara","Chapecoense","Corinthians","Coritiba","Cruzeiro","Cuiaba",
          "Flamengo","Fluminense","Fortaleza","Goias","Gremio","Internacional","Juventude","Palmeiras","Parana","Santos","Sao Paulo","Sport","Vasco","Vitoria"]

# Coletando informações do usuario
# - Clube
st.header('Faremos uma análise de partidas de futebol com base nos dados do usuário!')
st.write('- ESCOLHA UM TIME PARA A ANÁLISE')
clube_user = st.selectbox('', clubes_df) # time escolhido do usuário
st.text(f'clube escolhido: {clube_user}')
# Variaveis para a predição zerados
vals = [0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0] #31
tester = 0
for i in range(0,29): # 29 times
    if clube_user == clubes_df[i]:
        vals[i] = 1 # Colocando resultado nas variavel de predição
        tester = 1
if tester == 0:
    st.write("erro1")
df_user = df_estatistica[df_estatistica["clube"] == clube_user] #filtrando base de dados com o clube escolhidp
# - Chutes - Finalizações
st.write('- INSIRA UM NÚMERO DE CHUTES')
chutes_min = int(df_user["chutes"].min())
chutes_max = int(df_user["chutes"].max())
chutes_user = st.slider("",chutes_min,chutes_max) # Número de chutes escolhidos pelo usuário
st.text(f'quantidade de chutes selecionados: {chutes_user}')
# - Chutes no alvo - Finalizações na direção do gol
st.write('- INSIRA UM NÚMERO DE CHUTES NO ALVO')
chutes_alvo_min = int(df_user["chutes_no_alvo"].min())
chutes_alvo_max = int(df_user["chutes_no_alvo"].max())
chutes_alvo_user = st.slider("",chutes_alvo_min,chutes_alvo_max)
st.text(f'quantidade de chutes no alvo selecionado: {chutes_alvo_user}')  # Número de chutes no alvo escolhidos pelo usuário
# colocando resultados nas variaveis de predição
vals[29] = chutes_user
vals[30] = chutes_alvo_user

#predição
res = arv.predict([vals]) #predição
resul = str(res[0]) #coletando resultado em string
st.subheader(f'De acordo com a análise, seu time nesses parametros possui uma porcentagem de: {resul} em posse de bola')
st.write("Quantidade de partidas em que seu time possuiu essa porcentagem de posse de bola, e seus dados estatísticos")

# Avaliando partidas
df_estatistica_resul = df_user[df_user["posse_de_bola"]==resul] #coletando partidas do time escolhido com posse de bola predita
id_partida = df_estatistica_resul["partida_id"].values # id das partidas
st.write(df_estatistica_resul)
st.subheader("Agora, faremos uma analise dessas partidas, indicando se dentro desses parametros calculados ele, clube escolhido, venceria, perderia ou ganharia o jogo!")

#Variáveis de análise final zerados
st.write('- PARTIDAS:')
contador = 1 
empate = 0 
vitoria = 0
derrota = 0
for i in range (0,len(id_partida)): # detalhar cada partida
    clube_rival = ""
    pontos_rival = 0
    partida = df_dados[df_dados["ID"]== id_partida[i]] #filtrando partidas em outra base
    dados = ['data','mandante',"visitante","vencedor","mandante_Placar","visitante_Placar"] # colunas específicas
    partida = partida[dados] #filtrando
    partida_data = partida["data"].values # datando
    if partida["mandante"].values == clube_user: # identificando mandante e visitante (recolhendo placar)
        clube_rival = partida["visitante"].values
        pontos_rival = int(partida["visitante_Placar"].values)
        pontos_user = int(partida["mandante_Placar"].values)
    else:
        clube_rival = partida["mandante"].values
        pontos_rival = int(partida["mandante_Placar"].values)
        pontos_user = int(partida["visitante_Placar"].values)
    if partida["vencedor"].values == clube_user: # identificando vencedor
        vitoria += 1
    elif partida["vencedor"].values == "-":
        empate += 1 
    else:
        derrota += 1 
    st.write(f"{contador} - {partida_data[0]} --- {clube_user} {pontos_user} x {pontos_rival} {clube_rival[0]}") #apresentando dados
    contador +=1

#apresentando dados
st.write('- DADOS DAS PARTIDAS:')
st.write('vitorias: ',vitoria)
st.write('empate: ',empate)
st.write('derrota: ',derrota)

# Plotando Gráfico
data = pd.DataFrame({
    
     'x': ['VITÓRIA', 'DERROTA', 'EMPATE'],
          'Resultado': [vitoria, derrota, empate],
 }).set_index('x')
st.bar_chart(data)

# Aprensentando resultado final
if vitoria == 0 and derrota == 0 and empate == 0:
    st.subheader("Nesse cenário, nao podemos prever (sem experiencias com essa porcentagem de posse de bola)")
elif vitoria > derrota and vitoria == empate:
    st.subheader("Nesse cenário, seu time provavelmente ganharia ou empataria")
elif derrota > vitoria and derrota == empate:
    st.subheader("Nesse cenário, seu time provavelmente perderia ou empataria")
elif empate > vitoria and empate > derrota:
    st.subheader("Nesse cenário, seu time provavelmente empataria")
elif vitoria > empate and vitoria > derrota:
    st.subheader("Nesse cenário, seu time provavelmente ganharia")
elif derrota > empate and derrota > vitoria:
    st.subheader("Nesse cenário, seu time provavelmente perderia")
elif derrota == empate and derrota == vitoria:
    st.subheader("Nesse cenário, seu time pode tanto ganhar, empatar ou perder")
else:
    st.write('Erro3')




