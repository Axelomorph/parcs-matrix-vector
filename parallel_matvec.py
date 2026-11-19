import random
import time
from Pyro4 import expose


class Solver:
    def __init__(self, workers=None, input_file_name=None, output_file_name=None):
        self.input_file_name = input_file_name
        self.output_file_name = output_file_name
        self.workers = workers or []
        print("Inited")

    def solve(self):
        print("Job Started")
        num_workers = len(self.workers)
        print(f"Workers: {num_workers}")
        if num_workers == 0:
            raise RuntimeError("No workers available for parallel execution.")

        rows_n, cols_m, seed = self.read_input()

        rng = random.Random(seed)
        matrix_a = [[rng.random() for _ in range(cols_m)] for _ in range(rows_n)]
        vec_x = [rng.random() for _ in range(cols_m)]

        t_start = time.perf_counter()

        step = rows_n // num_workers
        mapped = []
        for i in range(num_workers):
            start_idx = i * step
            end_idx = rows_n if i == num_workers - 1 else (i + 1) * step
            chunk = matrix_a[start_idx:end_idx]
            mapped.append(self.workers[i].mymap(chunk, vec_x))

        result_y = self.myreduce(mapped)
        t_elapsed = time.perf_counter() - t_start

        self.write_output(result_y, t_elapsed, num_workers, rows_n, cols_m, seed)
        print("Job Finished")

    @staticmethod
    @expose
    def mymap(rows_chunk, vec_x):
        part_y = []
        for row in rows_chunk:
            total = 0.0
            for a_ij, x_j in zip(row, vec_x):
                total += a_ij * x_j
            part_y.append(total)
        return part_y

    @staticmethod
    @expose
    def myreduce(mapped):
        output = []
        for item in mapped:
            val = item.value if hasattr(item, "value") else item
            output.extend(val)
        return output

    def read_input(self):
        # Формат вхідного файлу: "N M seed"
        with open(self.input_file_name, "r", encoding="utf-8") as f:
            parts = f.read().strip().split()
        if not parts:
            raise ValueError(f"Input file '{self.input_file_name}' is empty.")
        rows_n = int(parts[0])
        cols_m = int(parts[1]) if len(parts) > 1 else rows_n
        seed = int(parts[2]) if len(parts) > 2 else 42
        return rows_n, cols_m, seed

    def write_output(self, output, elapsed, num_workers, rows_n, cols_m, seed):
        summary = f"mode=parallel size={rows_n}x{cols_m} seed={seed} workers={num_workers} time={elapsed:.6f}s"
        preview = " ".join(f"{val:.6f}" for val in output[:10])
        print(summary)
        print(f"preview: {preview}")
        if self.output_file_name:
            with open(self.output_file_name, "w", encoding="utf-8") as f:
                f.write(summary + "\n")
                f.write(preview + "\n")