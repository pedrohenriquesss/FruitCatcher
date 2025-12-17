import numpy as np

class DecisionTree:

    def __init__(self, X, y, threshold=1.0, max_depth=None, depth=0): # Additional optional arguments can be added, but the default value needs to be provided
        # Implement this
        self.is_leaf = False
        self.class_label = None
        self.split_attribute = None
        self.children = {}
        self.majority_class = self._compute_majority_class(y)
        
        # casos base para parar a recursividade
        if not X or not y or depth == max_depth or self._all_same(y):
            self.is_leaf = True
            self.class_label = self._compute_majority_class(y)
            return

        self.majority_class = self._compute_majority_class(y)
        best_attr, best_gain = self._select_best_attribute(X, y) # seleciona o melhor atributo para a divisão (maior ganho de informação)
        
        # se o ganho for insuficiente (menor que o treshold), não divide (fica uma folha)
        if best_gain < threshold:
            self.is_leaf = True
            self.class_label = self.majority_class
            return
        
        self.split_attribute = best_attr
        
        # dividir os dados pelos atributos e cria subárvores
        value_subsets = self._split_data(X, y, best_attr)
        for value, (sub_X, sub_y) in value_subsets.items():
            if not sub_X:  # sets vazios
                child = DecisionTree([], [], threshold, max_depth, depth + 1)
                child.is_leaf = True
                child.class_label = self.majority_class
            else:
                child = DecisionTree(sub_X, sub_y, threshold, max_depth, depth + 1)
            self.children[value] = child
    
    # calcula a moda (1 ou -1) numa lista de classificações
    def _compute_majority_class(self, y):
        if not y:
            return -1  # dados vazios
        count_1 = sum(1 for label in y if label == 1)
        return 1 if count_1 >= len(y) - count_1 else -1
    
    def _all_same(self, y):
        return len(set(y)) <= 1
    
    def _select_best_attribute(self, X, y):
        best_gain = -1
        best_attr = -1
        for attr in range(len(X[0])):
            gain = self._information_gain(X, y, attr)
            if gain > best_gain:
                best_gain, best_attr = gain, attr
        return best_attr, best_gain
    
    def _information_gain(self, X, y, attr):
        entropy_before = self._entropy(y)
        value_counts = {}

        # agrupa as classificações de acordo com o valor do atributo
        for i in range(len(X)):
            value = X[i][attr]
            if value not in value_counts:
                value_counts[value] = []
            value_counts[value].append(y[i])

        # calcula a entropia após a divisão
        entropy_after = sum(
            (len(labels) / len(y)) * self._entropy(labels)
            for labels in value_counts.values()
        )
        return entropy_before - entropy_after
    
    @staticmethod
    def _entropy(labels):
        if not labels:
            return 0.0
        p = sum(1 for label in labels if label == 1) / len(labels)
        if p == 0 or p == 1:
            return 0.0
        return -p * np.log2(p) - (1 - p) * np.log2(1 - p)
    
    def _split_data(self, X, y, attr):
        subsets = {}
        for i in range(len(X)):
            value = X[i][attr]
            if value not in subsets:
                subsets[value] = ([], [])
            subsets[value][0].append(X[i])
            subsets[value][1].append(y[i])
        return subsets

    def predict(self, x): # (e.g. x = ['apple', 'green', 'circle'] -> 1 or -1)
        # Implement this
        if self.is_leaf:
            return self.class_label
        value = x[self.split_attribute]
        if value in self.children:
            return self.children[value].predict(x)
        else:             # valor desconhecido → retorna a moda
            return self.majority_class


def train_decision_tree(X, y):
    # Replace with your configuration
    return DecisionTree(X, y, threshold=0.0, max_depth=3)