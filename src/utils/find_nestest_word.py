import numpy as np
from collections import Counter
import string
from src.utils.get_data import load_vocabulary
from src.containts.containts import similar_pairs

#leven
def levenshtein_distance(s1, s2):
    len_s1, len_s2 = len(s1), len(s2)
    dp = np.zeros((len_s1 + 1, len_s2 + 1), dtype=int)

    for i in range(len_s1 + 1):
        dp[i][0] = i
    for j in range(len_s2 + 1):
        dp[0][j] = j

    for i in range(1, len_s1 + 1):
        for j in range(1, len_s2 + 1):
            cost = 0 if s1[i - 1] == s2[j - 1] else 1
            dp[i][j] = min(dp[i - 1][j] + 1,  # Xóa
                           dp[i][j - 1] + 1,  # Chèn
                           dp[i - 1][j - 1] + cost)  # Thay thế
    return dp[len_s1][len_s2]


def create_weight_matrix():
    alphabet = "aăâbcdđeêghiklmnoôơpqrstuưvxyáàảãạăắằẳẵặâấầẩẫậéèẻẽẹêếềểễệíìỉĩịóòỏõọôốồổỗộơớờởỡợúùủũụưứừửữựýỳỷỹỵ"
    size = len(alphabet)
    weights = np.ones((size, size)) * 2 

    for a, b in similar_pairs:
        if a in alphabet and b in alphabet:  
            i, j = alphabet.index(a), alphabet.index(b)
            weights[i][j] = weights[j][i] = 1  

    np.fill_diagonal(weights, 0)
    return weights, alphabet

#WLeven
def weighted_levenshtein(s1, s2, weights, alphabet):
    len_s1, len_s2 = len(s1), len(s2)
    dp = np.zeros((len_s1 + 1, len_s2 + 1), dtype=float)

    for i in range(1, len_s1 + 1):
        dp[i][0] = i
    for j in range(1, len_s2 + 1):
        dp[0][j] = j

    for i in range(1, len_s1 + 1):
        for j in range(1, len_s2 + 1):
            char1 = s1[i - 1]
            char2 = s2[j - 1]

            if char1 in alphabet and char2 in alphabet:
                idx1, idx2 = alphabet.index(char1), alphabet.index(char2)
                substitute_cost = weights[idx1][idx2]
            else:
                substitute_cost = 2 

         
            dp[i][j] = min(
                dp[i - 1][j] + 1,  
                dp[i][j - 1] + 1, 
                dp[i - 1][j - 1] + substitute_cost             )
    return dp[len_s1][len_s2]

#n-gram
def ngram_similarity(s1, s2, n=2):
    def ngrams(string, n):
        return [string[i:i + n] for i in range(len(string) - n + 1)]

    ngrams1 = Counter(ngrams(s1, n))
    ngrams2 = Counter(ngrams(s2, n))

    intersection = sum((ngrams1 & ngrams2).values())
    total = sum((ngrams1 | ngrams2).values())
    return intersection / total if total > 0 else 0


def find_closest_word(vocabulary, query, metric="levenshtein", **kwargs):
    best_match = None
    best_score = float('inf') if metric != "ngram" else -1

    for word in vocabulary:
        if metric == "levenshtein":
            score = levenshtein_distance(word, query)
        elif metric == "weighted_levenshtein":
            score = weighted_levenshtein(word, query, **kwargs)
        elif metric == "ngram":
            score = ngram_similarity(word, query, **kwargs)
        else:
            raise ValueError("Unknown metric")

        # Cập nhật kết quả tốt nhất
        if (metric != "ngram" and score < best_score) or (metric == "ngram" and score > best_score):
            best_match = word
            best_score = score

    return best_match, best_score


def find(query):
    vocabulary = load_vocabulary()

    weight_matrix, alphabet = create_weight_matrix()
    print("Levenshtein:", find_closest_word(vocabulary, query, metric="levenshtein"))
    print("Weighted Levenshtein:",
          find_closest_word(vocabulary, query, metric="weighted_levenshtein", weights=weight_matrix, alphabet=alphabet))
    print("N-gram similarity:", find_closest_word(vocabulary, query, metric="ngram", n=2))
