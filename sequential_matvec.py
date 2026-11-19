# -*- coding: utf-8 -*-
import random
import sys
import time


def read_input(input_file_name):
    # Input format: "N M seed"
    f = open(input_file_name, "r")
    parts = f.read().strip().split()
    f.close()
    if not parts:
        raise ValueError("Input file '%s' is empty." % str(input_file_name))
    rows_n = int(parts[0])
    cols_m = int(parts[1]) if len(parts) > 1 else rows_n
    seed = int(parts[2]) if len(parts) > 2 else 42
    return rows_n, cols_m, seed


def write_output(output_file_name, output, elapsed, num_workers, rows_n, cols_m, seed):
    summary = "mode=sequential size=%dx%d seed=%d workers=%d time=%.6fs" % (
        rows_n, cols_m, seed, num_workers, elapsed
    )
    preview = " ".join("%.6f" % val for val in output[:10])
    print(summary)
    print("preview: %s" % preview)
    if output_file_name:
        f = open(output_file_name, "w")
        f.write(summary + "\n")
        f.write(preview + "\n")
        f.close()


def matvec_sequential(matrix_a, vec_x):
    result = []
    for row in matrix_a:
        total = 0.0
        for a_ij, x_j in zip(row, vec_x):
            total += a_ij * x_j
        result.append(total)
    return result


def main():
    input_file_name = sys.argv[1] if len(sys.argv) > 1 else "input.txt"
    output_file_name = sys.argv[2] if len(sys.argv) > 2 else None

    rows_n, cols_m, seed = read_input(input_file_name)

    rng = random.Random(seed)
    matrix_a = [[rng.random() for _ in range(cols_m)] for _ in range(rows_n)]
    vec_x = [rng.random() for _ in range(cols_m)]

    start_time = time.time()
    res = matvec_sequential(matrix_a, vec_x)
    elapsed = time.time() - start_time

    write_output(output_file_name, res, elapsed, 1, rows_n, cols_m, seed)


if __name__ == "__main__":
    main()
