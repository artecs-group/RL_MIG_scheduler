import math

# Función para calcular el producto de combinaciones a partir de los valores de alpha
def comb_product(X, alphas):
    result = 1
    total = 0
    for i, alpha in enumerate(alphas):
        result *= math.comb(X - total, alpha)
        total += alpha
    return result

def print_suma_alphas(X, alphas):
    ceros = X - sum(alphas)
    alphas = alphas + [ceros]
    print(alphas[::-1])
    suma = 0
    for i, alpha in enumerate(alphas[::-1]):
        print(f"alpha_{i} = {alpha}")
        suma += i*alpha
    print(f"Suma: {suma}")

# Función recursiva para implementar la suma
def sum_recursive(X, index, remaining_N, current_alphas, sum_before):
    if index == 1:
        last = N - sum_before
        if X < sum(current_alphas + [last]):
            return 0
        #print_suma_alphas(X, current_alphas + [last])
        return comb_product(X, current_alphas + [last])
    
    total_sum = 0
    for alpha in range(min(remaining_N // index, X - sum(current_alphas))+ 1):
        total_sum += sum_recursive(X, index - 1, remaining_N - index * alpha, current_alphas + [alpha], sum_before + index * alpha)
    return total_sum

# Función principal
def compute_sum(X, N):
    return sum_recursive(X, N, N, [], 0)


result_matrix = []
for N in range(1, 8):
    for M in range(1, 8):
        X = math.comb(M+4, 5)
        #print(f"N: {N}, M: {M}")
        binario_comb = math.comb(X, N)
        if binario_comb >= 10**4:
            binario_comb = "{:.1E}".format(binario_comb)
        #print(f"Binario: {binario_comb}")
        n_comb = compute_sum(X, N)
        if n_comb >= 10**4:
            n_comb = "{:.1E}".format(n_comb)
        #print(f"Hasta N: {n_comb}")
        salida = f"({binario_comb}/{n_comb})"
        print(f"{salida:^20}", end="")
        result_matrix.append((binario_comb, n_comb))
    print()

