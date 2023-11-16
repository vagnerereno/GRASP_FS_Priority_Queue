import pandas as pd
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import f1_score
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import random
import time

# Carregar o dataset e preparar os dados
data = pd.read_csv('WSN-DS.csv')
df = data.copy()
df["Attack type"] = df["Attack type"].map({
    "Normal": 0,
    "Grayhole": 1,
    "Blackhole": 2,
    "TDMA": 3,
    "Flooding": 4
}.get)

# Lista de features (ignorando a coluna 'Attack type')
feature_names = df.drop('Attack type', axis=1).columns.tolist()
print(len(feature_names))

# Avaliação com J48
def evaluate_with_j48(features_idx, dataset=df):
    try:
        # Mapear índices para nomes das colunas
        features = [feature_names[i] for i in features_idx]
    except IndexError as e:
        print(f"Erro ao acessar índice. Detalhes:")
        print(f"features_idx: {features_idx}")
        print(f"Tamanho de feature_names: {len(feature_names)}")
        raise e  # re-lança o erro

    # Filtrar colunas com base nas features fornecidas
    X = df[features]
    y = df['Attack type']

    # Dividir o dataset em treino e teste
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

    # Criar e treinar o classificador
    clf = DecisionTreeClassifier()
    clf.fit(X_train, y_train)

    # Prever os rótulos para o conjunto de teste
    y_pred = clf.predict(X_test)

    # Retornar o F1-Score
    return f1_score(y_test, y_pred, average="macro")
def busca_local(solution):
    RCL = [i for i in range(len(feature_names)) if i not in solution]
    max_f1_score = evaluate_with_j48(solution)

    for _ in range(10):
        new_solution = solution.copy()
        replace_index = random.choice(range(len(solution)))
        new_feature = random.choice(RCL)
        new_solution[replace_index] = new_feature
        f1_score = evaluate_with_j48(new_solution)

        if f1_score > max_f1_score:
            max_f1_score = f1_score
            solution = new_solution

    return max_f1_score, solution

def GRASP_FS():
    RCL = list(range(len(feature_names)))
    all_solutions = []
    local_search_improvements = {}  # Dicionário para armazenar os resultados da busca local

    priority_queue = MaxPriorityQueue()
    max_f1_score = -1
    best_solution = []

    start_time = time.perf_counter()

    for iteration in range(10):
        random.shuffle(RCL)
        solution = RCL[:5]
        f1_score = evaluate_with_j48(solution)
        print(f"F1-Score: {f1_score} for solution: {solution}")
        all_solutions.append((iteration, f1_score, solution))

        if f1_score > 0.70:
            # Se a fila de prioridade não estiver cheia, simplesmente insere o novo F1-Score.
            if len(priority_queue.heap) < 5:
                priority_queue.insert((f1_score, solution))
            else:
                # Se a fila de prioridade estiver cheia, encontre o menor F1-Score na fila.
                lowest_f1 = min(priority_queue.heap, key=lambda x: x[0])[0]
                if f1_score > lowest_f1:
                    # Remove o item com o menor F1-Score antes de inserir o novo item.
                    priority_queue.heap.remove((lowest_f1, [item[1] for item in priority_queue.heap if item[0] == lowest_f1][0]))
                    priority_queue.insert((f1_score, solution))
        local_search_improvements[tuple(solution)] = 0
        # visualize_heap(priority_queue.heap)
    total_elapsed_time = time.perf_counter() - start_time
    print(f"Tempo total de execução da Fase Construtiva: {total_elapsed_time} segundos")
    print_priority_queue(priority_queue)
    plot_solutions_with_priority(all_solutions, priority_queue)

    start_time = time.perf_counter()  # Busca Local

    while not priority_queue.is_empty():
        _, current_solution = priority_queue.extract_max()
        original_f1_score = evaluate_with_j48(current_solution)  # Avaliar a solução atual uma única vez
        improved_f1_score, improved_solution = busca_local(current_solution)

        # Verifica se houve melhoria em relação ao F1-Score original da solução específica
        if improved_f1_score > original_f1_score:
            local_search_improvements[tuple(current_solution)] = improved_f1_score - original_f1_score
            # Imprime a melhoria específica para essa solução
            print(f"Melhoria na Solução Local! F1-Score: {improved_f1_score} para solução: {current_solution}. Nova solução: {improved_solution}")

        # Verifica se a solução melhorada é a melhor solução global
        if improved_f1_score > max_f1_score:
            max_f1_score = improved_f1_score
            best_solution = improved_solution
            print(f"Nova Melhor Solução Global! F1-Score: {max_f1_score} para solução: {best_solution}")

    total_local_search_time = time.perf_counter() - start_time  # Busca Local


    local_search_results = [local_search_improvements.get(tuple(sol), 0) for _, _, sol in all_solutions]
    plot_solutions(all_solutions, priority_queue, local_search_improvements)

    print("Melhor F1-Score:", max_f1_score)
    print("Melhor conjunto de features:", best_solution)
    print(f"Tempo total de execução da Fase de Busca Local: {total_local_search_time} segundos")

class MaxPriorityQueue:
    def __init__(self):
        self.heap = []

    def insert(self, item):
        self.heap.append(item)
        self._sift_up(len(self.heap) - 1)

    def maximum(self):
        return self.heap[0] if self.heap else None

    def extract_max(self):
        if not self.heap:
            return None

        lastelt = self.heap.pop()
        if self.heap:
            minitem = self.heap[0]
            self.heap[0] = lastelt
            self._sift_down(0)
            return minitem
        return lastelt

    def _sift_up(self, pos):
        startpos = pos
        newitem = self.heap[pos]
        while pos > 0:
            parentpos = (pos - 1) >> 1
            parent = self.heap[parentpos]
            if newitem <= parent:
                break
            self.heap[pos] = parent
            pos = parentpos
        self.heap[pos] = newitem

    def _sift_down(self, pos):
        endpos = len(self.heap)
        startpos = pos
        newitem = self.heap[pos]
        childpos = 2 * pos + 1
        while childpos < endpos:
            rightpos = childpos + 1
            if rightpos < endpos and self.heap[childpos] <= self.heap[rightpos]:
                childpos = rightpos
            self.heap[pos] = self.heap[childpos]
            pos = childpos
            childpos = 2 * pos + 1
        self.heap[pos] = newitem

    def is_empty(self):
        return len(self.heap) == 0

def plot_solutions_with_priority(all_solutions, priority_queue):
    # Convertendo a fila de prioridade em um set para busca rápida
    priority_set = set([tuple(sol) for _, sol in priority_queue.heap])

    # Pegando índices de iteração e F1-Scores
    iterations = [iteration for _, iteration, _ in all_solutions]
    f1_scores = [f1 for f1, _, _ in all_solutions]

    # Verificando quais soluções estão no top 10
    priority_colors = ['red' if tuple(sol) in priority_set else 'blue' for _, _, sol in all_solutions]

    plt.scatter(f1_scores, iterations, color=priority_colors)
    plt.ylabel('F1-Score')
    plt.xlabel('Índice da Solução')
    plt.legend(handles=[plt.Line2D([0], [0], marker='o', color='w', label='Top 10', markersize=10, markerfacecolor='red'),
                       plt.Line2D([0], [0], marker='o', color='w', label='Outras Soluções', markersize=10, markerfacecolor='blue')],
               loc='lower right')
    plt.savefig("priority_plot.png")

def print_priority_queue(priority_queue):
    print("Fila de prioridade:")
    for score, solution in priority_queue.heap:
        print(f"F1-Score: {-score}, Solution: {solution}")  # Note o sinal negativo para converter de volta


def plot_solutions(all_solutions, priority_queue, local_search_improvements):
    # Configura o tamanho da figura
    plt.figure(figsize=(15, 6))

    # Convertendo a fila de prioridade em um set para busca rápida
    priority_set = set([tuple(sol) for _, sol in priority_queue.heap])

    # Pegando índices de iteração e F1-Scores
    iterations = [iteration for iteration, _, _ in all_solutions]
    f1_scores = [f1 for _, f1, _ in all_solutions]
    solutions = [sol for _, _, sol in all_solutions]

    # Desenha todas as barras azuis primeiro
    plt.bar(iterations, f1_scores, color='blue', label='Outras Soluções')

    # Desenha as barras das soluções da fila de prioridade em vermelho
    for i, sol in enumerate(solutions):
        if tuple(sol) in priority_set:
            plt.bar(iterations[i], f1_scores[i], color='red', label='Top 10')

    # Sobrepinta a melhoria em verde onde aplicável
    for i, sol in enumerate(solutions):
        improvement = local_search_improvements.get(tuple(sol), 0)
        if improvement > 0:
            plt.bar(iterations[i], improvement, bottom=f1_scores[i], color='green', label='Melhoradas')

    # Adiciona legendas e rótulos
    plt.xlabel('Índice da Solução')
    plt.ylabel('F1-Score')
    # plt.xticks(iterations, rotation=90)
    # Cria legendas únicas
    handles, labels = plt.gca().get_legend_handles_labels()
    by_label = dict(zip(labels, handles))
    by_label = {label: handle for label, handle in by_label.items() if label in ['Outras Soluções', 'Top 10', 'Melhoradas']}
    plt.legend(by_label.values(), by_label.keys(), loc='lower right')

    # Mostra o gráfico
    plt.savefig("all.png")

if __name__ == '__main__':
    GRASP_FS()