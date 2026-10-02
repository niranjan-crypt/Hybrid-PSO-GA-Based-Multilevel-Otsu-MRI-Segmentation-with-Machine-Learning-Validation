# ============================================================
# BASELINE: PSO vs GA FOR MULTILEVEL OTSU MRI IMAGE SEGMENTATION
# ============================================================
import argparse
import csv
import os
import time
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image

# ============================================================
# DEFAULT SETTINGS
# ============================================================
DEFAULT_IMAGE_DIR = (
    "/Users/niranjankrishnakumar/Downloads/archive/Training/notumor"
    if os.path.isdir("/Users/niranjankrishnakumar/Downloads/archive/Training/notumor")
    else "./data"
)
DEFAULT_OUTPUT_DIR = "./results"
DEFAULT_CSV_PATH = "./results/baseline_results.csv"

IMAGE_FILES = [
    "Tr-no_995.jpg",
    "Tr-no_996.jpg",
    "Tr-no_997.jpg",
    "Tr-no_998.jpg",
    "Tr-no_999.jpg",
]

NUMBER_OF_THRESHOLDS = 2      # 2 thresholds = 3 segmentation classes
NUMBER_OF_ITERATIONS = 100    # PSO iterations and GA generations
RANDOM_SEED = 42


# ============================================================
# 1. LOAD IMAGE
# ============================================================
def load_grayscale_image(path, resize_to=256):
    """Load image, convert to grayscale, and resize to 256 x 256."""
    img = Image.open(path).convert("L")
    if resize_to is not None:
        img = img.resize((resize_to, resize_to))
    return np.array(img, dtype=np.uint8)


# ============================================================
# 2. COMPUTE HISTOGRAM
# ============================================================
def compute_histogram(image):
    """Compute normalized 256-bin histogram."""
    hist, _ = np.histogram(image.flatten(), bins=256, range=(0, 256))
    return hist / hist.sum()


# ============================================================
# 3. OTSU FITNESS FUNCTION
# ============================================================
def otsu_between_class_variance(thresholds, hist):
    """Calculate Otsu between-class variance. Higher fitness = better."""
    t = np.clip(np.round(thresholds), 0, 255).astype(int)
    t = np.unique(t)
    t = np.sort(t)

    boundaries = [0] + list(t) + [256]
    levels = np.arange(256)
    global_mean = np.sum(levels * hist)

    variance = 0.0
    for i in range(len(boundaries) - 1):
        lo, hi = boundaries[i], boundaries[i + 1]
        class_prob = np.sum(hist[lo:hi])
        if class_prob < 1e-12:
            continue
        class_mean = np.sum(levels[lo:hi] * hist[lo:hi]) / class_prob
        variance += class_prob * (class_mean - global_mean) ** 2

    return variance


# ============================================================
# 4. SEGMENT IMAGE
# ============================================================
def segment_image(image, thresholds):
    """Segment image using optimized thresholds."""
    t = np.clip(np.round(thresholds), 0, 255).astype(int)
    t = np.unique(t)
    t = np.sort(t)

    boundaries = [0] + list(t) + [256]
    n_classes = len(boundaries) - 1

    output = np.zeros_like(image, dtype=np.uint8)
    gray_levels = np.linspace(0, 255, n_classes).astype(np.uint8)

    for i in range(n_classes):
        lo, hi = boundaries[i], boundaries[i + 1]
        mask = (image >= lo) & (image < hi)
        output[mask] = gray_levels[i]

    return output


# ============================================================
# 5. PSO OPTIMIZATION
# ============================================================
def pso_optimize(hist, n_thresholds=2, n_particles=30, n_iterations=100,
                  w=0.7, c1=1.5, c2=1.5, seed=None):
    """Particle Swarm Optimization. Objective: maximize Otsu variance."""
    rng = np.random.default_rng(seed)

    positions = rng.uniform(0, 255, size=(n_particles, n_thresholds))
    velocities = rng.uniform(-10, 10, size=(n_particles, n_thresholds))

    pbest_positions = positions.copy()
    pbest_fitness = np.array([
        otsu_between_class_variance(positions[i], hist)
        for i in range(n_particles)
    ])

    gbest_idx = np.argmax(pbest_fitness)
    gbest_position = pbest_positions[gbest_idx].copy()
    gbest_fitness = pbest_fitness[gbest_idx]

    convergence_curve = []

    for _ in range(n_iterations):
        for i in range(n_particles):
            r1 = rng.random(n_thresholds)
            r2 = rng.random(n_thresholds)

            velocities[i] = (
                w * velocities[i]
                + c1 * r1 * (pbest_positions[i] - positions[i])
                + c2 * r2 * (gbest_position - positions[i])
            )
            positions[i] = np.clip(positions[i] + velocities[i], 0, 255)

            fitness = otsu_between_class_variance(positions[i], hist)

            if fitness > pbest_fitness[i]:
                pbest_fitness[i] = fitness
                pbest_positions[i] = positions[i].copy()

                if fitness > gbest_fitness:
                    gbest_fitness = fitness
                    gbest_position = positions[i].copy()

        convergence_curve.append(gbest_fitness)

    return gbest_position, gbest_fitness, convergence_curve


# ============================================================
# 6. GA TOURNAMENT SELECTION
# ============================================================
def tournament_select(population, fitness, rng, k=3):
    """Tournament selection."""
    idx = rng.integers(0, len(population), size=k)
    best_idx = idx[np.argmax(fitness[idx])]
    return population[best_idx].copy()


# ============================================================
# 7. GA OPTIMIZATION
# ============================================================
def ga_optimize(hist, n_thresholds=2, population_size=30, n_generations=100,
                 crossover_rate=0.8, mutation_rate=0.1, mutation_std=15, seed=None):
    """Genetic Algorithm. Objective: maximize Otsu variance."""
    rng = np.random.default_rng(seed)

    population = rng.uniform(0, 255, size=(population_size, n_thresholds))
    fitness = np.array([
        otsu_between_class_variance(population[i], hist)
        for i in range(population_size)
    ])

    best_idx = np.argmax(fitness)
    best_position = population[best_idx].copy()
    best_fitness = fitness[best_idx]

    convergence_curve = []

    for _ in range(n_generations):
        new_population = [best_position.copy()]  # elitism

        while len(new_population) < population_size:
            parent1 = tournament_select(population, fitness, rng)
            parent2 = tournament_select(population, fitness, rng)

            # Crossover
            if rng.random() < crossover_rate and n_thresholds > 1:
                point = rng.integers(1, n_thresholds)
                child1 = np.concatenate([parent1[:point], parent2[point:]])
                child2 = np.concatenate([parent2[:point], parent1[point:]])
            else:
                child1 = parent1.copy()
                child2 = parent2.copy()

            # Mutation
            for child in (child1, child2):
                mutate_mask = rng.random(n_thresholds) < mutation_rate
                noise = rng.normal(0, mutation_std, size=n_thresholds)
                child[mutate_mask] += noise[mutate_mask]
                np.clip(child, 0, 255, out=child)

            new_population.append(child1)
            if len(new_population) < population_size:
                new_population.append(child2)

        population = np.array(new_population)
        fitness = np.array([
            otsu_between_class_variance(population[i], hist)
            for i in range(population_size)
        ])

        gen_best_idx = np.argmax(fitness)
        if fitness[gen_best_idx] > best_fitness:
            best_fitness = fitness[gen_best_idx]
            best_position = population[gen_best_idx].copy()

        convergence_curve.append(best_fitness)

    return best_position, best_fitness, convergence_curve


# ============================================================
# 8. CONVERGENCE ITERATION
# ============================================================
def iters_to_converge(curve, final_val):
    """Find first iteration where fitness reaches 99% of final fitness."""
    target = 0.99 * final_val
    for i, value in enumerate(curve):
        if value >= target:
            return i + 1
    return len(curve)


# ============================================================
# 9. PROCESS ONE IMAGE
# ============================================================
def process_image(image_path, output_dir):
    name = os.path.splitext(os.path.basename(image_path))[0]
    print()
    print("=" * 70)
    print("PROCESSING:", name)
    print("=" * 70)

    image = load_grayscale_image(image_path, resize_to=256)
    hist = compute_histogram(image)

    # ---------------- RUN PSO ----------------
    print("Running PSO...")
    start_time = time.time()
    pso_thresholds, pso_fitness, pso_curve = pso_optimize(
        hist,
        n_thresholds=NUMBER_OF_THRESHOLDS,
        n_particles=30,
        n_iterations=NUMBER_OF_ITERATIONS,
        seed=RANDOM_SEED,
    )
    pso_time = time.time() - start_time

    # ---------------- RUN GA ----------------
    print("Running GA...")
    start_time = time.time()
    ga_thresholds, ga_fitness, ga_curve = ga_optimize(
        hist,
        n_thresholds=NUMBER_OF_THRESHOLDS,
        population_size=30,
        n_generations=NUMBER_OF_ITERATIONS,
        seed=RANDOM_SEED,
    )
    ga_time = time.time() - start_time

    # ---------------- PRINT RESULTS ----------------
    print()
    print("-" * 60)
    print("PSO Thresholds:", sorted(np.round(pso_thresholds, 1).tolist()))
    print("PSO Fitness:", round(pso_fitness, 3))
    print("PSO Runtime:", round(pso_time, 4), "seconds")
    print()
    print("GA Thresholds:", sorted(np.round(ga_thresholds, 1).tolist()))
    print("GA Fitness:", round(ga_fitness, 3))
    print("GA Runtime:", round(ga_time, 4), "seconds")
    print("-" * 60)

    # ---------------- CONVERGENCE GRAPH ----------------
    plt.figure(figsize=(8, 5))
    plt.plot(pso_curve, label="PSO", linewidth=2)
    plt.plot(ga_curve, label="GA", linewidth=2)
    plt.xlabel("Iteration / Generation")
    plt.ylabel("Best Otsu Fitness")
    plt.title(f"PSO vs GA Convergence - {name}")
    plt.legend()
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, f"{name}_convergence.png"), dpi=150)
    plt.close()

    # ---------------- SEGMENTATION ----------------
    pso_segmented = segment_image(image, pso_thresholds)
    ga_segmented = segment_image(image, ga_thresholds)

    fig, axes = plt.subplots(1, 3, figsize=(12, 4))
    axes[0].imshow(image, cmap="gray")
    axes[0].set_title("Original")
    axes[0].axis("off")

    axes[1].imshow(pso_segmented, cmap="gray")
    axes[1].set_title("PSO Segmented")
    axes[1].axis("off")

    axes[2].imshow(ga_segmented, cmap="gray")
    axes[2].set_title("GA Segmented")
    axes[2].axis("off")

    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, f"{name}_segmented.png"), dpi=150)
    plt.close()

    # ---------------- CSV ROWS ----------------
    pso_row = {
        "image": name,
        "algorithm": "PSO",
        "final_fitness": round(pso_fitness, 3),
        "thresholds": sorted(np.round(pso_thresholds, 1).tolist()),
        "runtime_sec": round(pso_time, 4),
        "iters_to_converge": iters_to_converge(pso_curve, pso_fitness),
    }
    ga_row = {
        "image": name,
        "algorithm": "GA",
        "final_fitness": round(ga_fitness, 3),
        "thresholds": sorted(np.round(ga_thresholds, 1).tolist()),
        "runtime_sec": round(ga_time, 4),
        "iters_to_converge": iters_to_converge(ga_curve, ga_fitness),
    }

    return [pso_row, ga_row]


# ============================================================
# 10. MAIN
# ============================================================
def main():
    parser = argparse.ArgumentParser(description="Baseline PSO vs GA for Multilevel Otsu Image Segmentation")
    parser.add_argument("--image_dir", type=str, default=DEFAULT_IMAGE_DIR, help="Path to input MRI images directory")
    parser.add_argument("--output_dir", type=str, default=DEFAULT_OUTPUT_DIR, help="Path to output directory")
    parser.add_argument("--csv_path", type=str, default=DEFAULT_CSV_PATH, help="Path to output CSV file")
    args = parser.parse_args()

    print()
    print("=" * 70)
    print("BASELINE: PSO vs GA MRI IMAGE SEGMENTATION")
    print("=" * 70)
    print()
    print("Dataset folder:", args.image_dir)
    print("Output folder:", args.output_dir)
    print("CSV output:", args.csv_path)
    print("Thresholds:", NUMBER_OF_THRESHOLDS)
    print("Iterations:", NUMBER_OF_ITERATIONS)

    if not os.path.isdir(args.image_dir):
        print()
        print("ERROR: Dataset folder not found:", args.image_dir)
        return

    os.makedirs(args.output_dir, exist_ok=True)

    image_paths = []
    for filename in IMAGE_FILES:
        full_path = os.path.join(args.image_dir, filename)
        if os.path.isfile(full_path):
            image_paths.append(full_path)
        else:
            print("WARNING: Image not found:", full_path)

    if len(image_paths) == 0:
        all_exts = {".jpg", ".jpeg", ".png", ".bmp"}
        image_paths = [
            os.path.join(args.image_dir, f) for f in sorted(os.listdir(args.image_dir))
            if os.path.splitext(f)[1].lower() in all_exts
        ][:5]

    if len(image_paths) == 0:
        print()
        print("ERROR: No valid images found to process.")
        return

    print()
    print("=" * 70)
    print("IMAGES TO BE PROCESSED")
    print("=" * 70)
    for i, path in enumerate(image_paths, start=1):
        print(f"{i}. {os.path.basename(path)}")
    print("=" * 70)

    all_results = []
    for i, image_path in enumerate(image_paths, start=1):
        print()
        print(f"IMAGE {i} OF {len(image_paths)}")
        results = process_image(image_path, args.output_dir)
        all_results.extend(results)

    with open(args.csv_path, "w", newline="") as file:
        writer = csv.DictWriter(
            file,
            fieldnames=[
                "image",
                "algorithm",
                "final_fitness",
                "thresholds",
                "runtime_sec",
                "iters_to_converge",
            ],
        )
        writer.writeheader()
        writer.writerows(all_results)

    print()
    print("=" * 70)
    print("PROCESSING COMPLETE")
    print("=" * 70)
    print("Images processed:", len(image_paths))
    print("CSV saved to:", args.csv_path)
    print("Visualizations saved to:", args.output_dir)
    print("=" * 70)


if __name__ == "__main__":
    main()
