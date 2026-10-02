# Hybrid Evolutionary Optimization for Multilevel Otsu MRI Image Segmentation with ML-Based Validation
import csv
import os
import time
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image

try:
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.metrics import accuracy_score, confusion_matrix, f1_score, precision_score, recall_score
    from sklearn.model_selection import train_test_split
    SKLEARN_AVAILABLE = True
except ImportError:
    SKLEARN_AVAILABLE = False

# ----------------------------- SETTINGS -----------------------------
RUN_MULTIPLE_RUNS = True
RUN_PARAMETER_ANALYSIS = False
RUN_ML = True

NUMBER_OF_THRESHOLDS = 2
NUMBER_OF_ITERATIONS = 100
POPULATION_SIZE = 30
N_RUNS = 10 if RUN_MULTIPLE_RUNS else 1
RANDOM_SEED = 42

# Default directory paths (configured for local environment with fallback)
DEFAULT_LOCAL_DIR = "/Users/niranjankrishnakumar/Downloads/archive/Training/notumor"
IMAGE_DIR = DEFAULT_LOCAL_DIR if os.path.isdir(DEFAULT_LOCAL_DIR) else "./data"
OUTPUT_DIR = "./results"
CSV_PATH = os.path.join(OUTPUT_DIR, "results_table.csv")
ML_CSV_PATH = os.path.join(OUTPUT_DIR, "ml_results.csv")

IMAGE_FILES = ["Tr-no_995.jpg", "Tr-no_996.jpg", "Tr-no_997.jpg", "Tr-no_998.jpg", "Tr-no_999.jpg"]
ALGORITHMS = ["GA", "PSO", "DE", "Hybrid PSO-GA"]
COLORS = ["#e76f51", "#2a9d8f", "#e9c46a", "#1d3557"]


def out(name):
    return os.path.join(OUTPUT_DIR, name)


def savefig(name):
    plt.tight_layout()
    plt.savefig(out(name), dpi=150)
    plt.close()


# ----------------------------- IMAGE / FITNESS -----------------------------
def load_grayscale_image(path, resize_to=256):
    img = Image.open(path).convert("L")
    if resize_to is not None:
        img = img.resize((resize_to, resize_to))
    return np.array(img, dtype=np.uint8)


def compute_histogram(image):
    hist, _ = np.histogram(image.flatten(), bins=256, range=(0, 256))
    total = hist.sum()
    return hist / total if total > 0 else hist


def _bounds(thresholds):
    t = np.sort(np.unique(np.clip(np.round(thresholds), 0, 255).astype(int)))
    return [0] + list(t) + [256]


def otsu_between_class_variance(thresholds, hist):
    b = _bounds(thresholds)
    levels = np.arange(256)
    global_mean = np.sum(levels * hist)
    variance = 0.0
    for lo, hi in zip(b[:-1], b[1:]):
        p = np.sum(hist[lo:hi])
        if p < 1e-12:
            continue
        mean = np.sum(levels[lo:hi] * hist[lo:hi]) / p
        variance += p * ((mean - global_mean) ** 2)
    return float(variance)


def segment_image(image, thresholds):
    b = _bounds(thresholds)
    n = len(b) - 1
    output = np.zeros_like(image, dtype=np.uint8)
    gray_levels = np.linspace(0, 255, n).astype(np.uint8)
    for i in range(n):
        output[(image >= b[i]) & (image < b[i + 1])] = gray_levels[i]
    return output


def iters_to_converge(curve, final_val):
    target = 0.99 * final_val
    for i, v in enumerate(curve):
        if v >= target:
            return i + 1
    return len(curve)


# ----------------------------- OPTIMIZERS -----------------------------
def pso_optimize(hist, n_thresholds=2, n_particles=30, n_iterations=100,
                 w=0.7, c1=1.5, c2=1.5, seed=None):
    rng = np.random.default_rng(seed)
    pos = rng.uniform(0, 255, size=(n_particles, n_thresholds))
    vel = rng.uniform(-10, 10, size=(n_particles, n_thresholds))

    pbest = pos.copy()
    pbest_fit = np.array([otsu_between_class_variance(p, hist) for p in pos])
    n_evals = n_particles
    g = np.argmax(pbest_fit)
    gbest, gbest_fit = pbest[g].copy(), pbest_fit[g]
    curve = []

    for _ in range(n_iterations):
        for i in range(n_particles):
            r1, r2 = rng.random(n_thresholds), rng.random(n_thresholds)
            vel[i] = w * vel[i] + c1 * r1 * (pbest[i] - pos[i]) + c2 * r2 * (gbest - pos[i])
            pos[i] = np.clip(pos[i] + vel[i], 0, 255)
            fit = otsu_between_class_variance(pos[i], hist)
            n_evals += 1
            if fit > pbest_fit[i]:
                pbest_fit[i], pbest[i] = fit, pos[i].copy()
                if fit > gbest_fit:
                    gbest_fit, gbest = fit, pos[i].copy()
        curve.append(gbest_fit)
    return gbest, gbest_fit, curve, n_evals


def tournament_select(population, fitness, rng, k=3):
    idx = rng.integers(0, len(population), size=k)
    return population[idx[np.argmax(fitness[idx])]].copy()


def ga_optimize(hist, n_thresholds=2, population_size=30, n_generations=100,
                crossover_rate=0.8, mutation_rate=0.1, mutation_std=15, seed=None):
    rng = np.random.default_rng(seed)
    pop = rng.uniform(0, 255, size=(population_size, n_thresholds))
    fitness = np.array([otsu_between_class_variance(p, hist) for p in pop])
    n_evals = population_size
    b = np.argmax(fitness)
    best, best_fit = pop[b].copy(), fitness[b]
    curve = []

    for _ in range(n_generations):
        new_pop = [best.copy()]
        while len(new_pop) < population_size:
            p1 = tournament_select(pop, fitness, rng)
            p2 = tournament_select(pop, fitness, rng)
            if rng.random() < crossover_rate and n_thresholds > 1:
                pt = rng.integers(1, n_thresholds)
                c1 = np.concatenate([p1[:pt], p2[pt:]])
                c2 = np.concatenate([p2[:pt], p1[pt:]])
            else:
                c1, c2 = p1.copy(), p2.copy()
            for child in (c1, c2):
                mask = rng.random(n_thresholds) < mutation_rate
                noise = rng.normal(0, mutation_std, size=n_thresholds)
                child[mask] += noise[mask]
                np.clip(child, 0, 255, out=child)
            new_pop.append(c1)
            if len(new_pop) < population_size:
                new_pop.append(c2)

        pop = np.array(new_pop[:population_size])
        fitness = np.zeros(population_size)
        fitness[0] = best_fit
        for i in range(1, population_size):
            fitness[i] = otsu_between_class_variance(pop[i], hist)
            n_evals += 1

        b = np.argmax(fitness)
        if fitness[b] > best_fit:
            best_fit, best = fitness[b], pop[b].copy()
        curve.append(best_fit)
    return best, best_fit, curve, n_evals


def de_optimize(hist, n_thresholds=2, population_size=30, n_iterations=100,
                F=0.5, CR=0.8, seed=None):
    rng = np.random.default_rng(seed)
    pop = rng.uniform(0, 255, size=(population_size, n_thresholds))
    fitness = np.array([otsu_between_class_variance(p, hist) for p in pop])
    n_evals = population_size
    b = np.argmax(fitness)
    best, best_fit = pop[b].copy(), fitness[b]
    curve = []

    for _ in range(n_iterations):
        for i in range(population_size):
            candidates = [k for k in range(population_size) if k != i]
            r1, r2, r3 = rng.choice(candidates, size=3, replace=False)
            mutant = np.clip(pop[r1] + F * (pop[r2] - pop[r3]), 0, 255)

            j_rand = rng.integers(0, n_thresholds)
            trial = np.empty(n_thresholds)
            for j in range(n_thresholds):
                trial[j] = mutant[j] if (rng.random() < CR or j == j_rand) else pop[i, j]

            t_fit = otsu_between_class_variance(trial, hist)
            n_evals += 1
            if t_fit >= fitness[i]:
                pop[i], fitness[i] = trial, t_fit
                if t_fit > best_fit:
                    best_fit, best = t_fit, trial.copy()
        curve.append(best_fit)
    return best, best_fit, curve, n_evals


def hybrid_pso_ga_optimize(hist, n_thresholds=2, swarm_size=30, n_iterations=100,
                           w=0.7, c1=1.5, c2=1.5, crossover_rate=0.8,
                           mutation_rate=0.15, mutation_std=15, seed=None):
    """Hybrid PSO-GA: PSO update, then GA crossover/mutation offspring replace the least-fit particles."""
    rng = np.random.default_rng(seed)
    pos = rng.uniform(0, 255, size=(swarm_size, n_thresholds))
    vel = rng.uniform(-10, 10, size=(swarm_size, n_thresholds))
    fitness = np.array([otsu_between_class_variance(p, hist) for p in pos])
    n_evals = swarm_size

    pbest, pbest_fit = pos.copy(), fitness.copy()
    g = np.argmax(fitness)
    gbest, gbest_fit = pos[g].copy(), fitness[g]
    curve = []

    for _ in range(n_iterations):
        for i in range(swarm_size):
            r1, r2 = rng.random(n_thresholds), rng.random(n_thresholds)
            vel[i] = w * vel[i] + c1 * r1 * (pbest[i] - pos[i]) + c2 * r2 * (gbest - pos[i])
            pos[i] = np.clip(pos[i] + vel[i], 0, 255)
            fit = otsu_between_class_variance(pos[i], hist)
            n_evals += 1
            fitness[i] = fit
            if fit > pbest_fit[i]:
                pbest_fit[i], pbest[i] = fit, pos[i].copy()
                if fit > gbest_fit:
                    gbest_fit, gbest = fit, pos[i].copy()

        n_off = max(2, swarm_size // 4)
        offspring = []
        for _ in range(n_off):
            p1 = tournament_select(pos, fitness, rng, k=3)
            p2 = tournament_select(pos, fitness, rng, k=3)
            if rng.random() < crossover_rate and n_thresholds > 1:
                pt = rng.integers(1, n_thresholds)
                child = np.concatenate([p1[:pt], p2[pt:]])
            else:
                alpha = rng.random()
                child = alpha * p1 + (1 - alpha) * p2
            mask = rng.random(n_thresholds) < mutation_rate
            if np.any(mask):
                noise = rng.normal(0, mutation_std, size=n_thresholds)
                child[mask] += noise[mask]
            offspring.append(np.clip(child, 0, 255))

        for child, idx in zip(offspring, np.argsort(fitness)[:n_off]):
            c_fit = otsu_between_class_variance(child, hist)
            n_evals += 1
            if c_fit > fitness[idx]:
                pos[idx], fitness[idx] = child, c_fit
                vel[idx] = rng.uniform(-5, 5, size=n_thresholds)
                if c_fit > pbest_fit[idx]:
                    pbest_fit[idx], pbest[idx] = c_fit, child.copy()
                if c_fit > gbest_fit:
                    gbest_fit, gbest = c_fit, child.copy()
        curve.append(gbest_fit)
    return gbest, gbest_fit, curve, n_evals


# ----------------------------- FEATURES -----------------------------
def compute_glcm_features(image):
    q = (image // 16).astype(np.int32)
    glcm = np.zeros((16, 16), dtype=np.float64)
    np.add.at(glcm, (q[:, :-1].ravel(), q[:, 1:].ravel()), 1.0)
    glcm = glcm + glcm.T
    total = np.sum(glcm)
    p = glcm / total if total > 0 else np.zeros((16, 16))

    i, j = np.indices((16, 16))
    contrast = float(np.sum(((i - j) ** 2) * p))
    homogeneity = float(np.sum(p / (1.0 + np.abs(i - j))))
    energy = float(np.sum(p ** 2))
    mu_i, mu_j = np.sum(i * p), np.sum(j * p)
    s_i = np.sqrt(np.sum(((i - mu_i) ** 2) * p))
    s_j = np.sqrt(np.sum(((j - mu_j) ** 2) * p))
    correlation = float(np.sum((i - mu_i) * (j - mu_j) * p) / (s_i * s_j)) if s_i * s_j > 1e-12 else 0.0
    return contrast, correlation, energy, homogeneity


def extract_features(img):
    f = img.astype(np.float64)
    hist, _ = np.histogram(img.ravel(), bins=256, range=(0, 256), density=True)
    hist = hist[hist > 0]
    entropy = float(-np.sum(hist * np.log2(hist + 1e-12)))

    fg = (img > 0).astype(np.int32)
    area = float(np.sum(fg)) / float(img.size)
    perimeter = float(np.sum(np.abs(np.diff(fg, axis=1))) + np.sum(np.abs(np.diff(fg, axis=0))))
    return [float(np.mean(f)), float(np.std(f)), float(np.var(f)), entropy,
            *compute_glcm_features(img), area, perimeter]


# ----------------------------- DATASET CHECKS -----------------------------
def check_ground_truth_masks(image_dir, image_files):
    parent = os.path.dirname(os.path.abspath(image_dir))
    for d in (os.path.join(image_dir, "masks"), os.path.join(parent, "masks"),
              os.path.join(image_dir, "ground_truth")):
        if os.path.isdir(d) and any(os.path.isfile(os.path.join(d, f)) for f in image_files):
            return d
    return None


def check_ml_dataset_availability(image_dir):
    parent = os.path.dirname(os.path.abspath(image_dir))
    if not os.path.isdir(parent):
        return False, []
    exts = {".jpg", ".jpeg", ".png", ".bmp"}
    valid = []
    for item in sorted(os.listdir(parent)):
        sub = os.path.join(parent, item)
        if os.path.isdir(sub) and not item.startswith("."):
            count = sum(1 for f in os.listdir(sub) if os.path.splitext(f)[1].lower() in exts)
            if count >= 5:
                valid.append((item, sub, count))
    return (True, valid) if len(valid) >= 2 else (False, [])


# ----------------------------- DOWNSTREAM ML -----------------------------
def run_downstream_ml(valid_classes, n_thresholds, seed):
    if not SKLEARN_AVAILABLE:
        print("[ML WARNING] scikit-learn is not available. ML evaluation skipped.")
        return None

    print("\n" + "=" * 70)
    print("DOWNSTREAM ML CLASSIFICATION EXTENSION (RANDOM FOREST)")
    print("=" * 70)
    print(f"Available classes: {[c[0] for c in valid_classes]}")

    dataset = []
    for label, (_, class_dir, _) in enumerate(valid_classes):
        files = [os.path.join(class_dir, f) for f in sorted(os.listdir(class_dir))
                 if os.path.splitext(f)[1].lower() in {".jpg", ".jpeg", ".png"}][:20]
        dataset += [(f, label) for f in files]
    print(f"Total balanced samples loaded: {len(dataset)}")

    kw = dict(n_thresholds=n_thresholds, seed=seed)
    pipelines = {
        "GA Segmented": lambda h: ga_optimize(h, population_size=20, n_generations=40, **kw)[0],
        "PSO Segmented": lambda h: pso_optimize(h, n_particles=20, n_iterations=40, **kw)[0],
        "DE Segmented": lambda h: de_optimize(h, population_size=20, n_iterations=40, **kw)[0],
        "Hybrid PSO-GA": lambda h: hybrid_pso_ga_optimize(h, swarm_size=20, n_iterations=40, **kw)[0],
    }
    methods = ["Original MRI", "GA Segmented", "PSO Segmented", "DE Segmented", "Hybrid PSO-GA"]
    features = {m: [] for m in methods}
    labels = []

    for idx, (path, label) in enumerate(dataset, start=1):
        raw = load_grayscale_image(path)
        hist = compute_histogram(raw)
        labels.append(label)
        features["Original MRI"].append(extract_features(raw))
        for m, fn in pipelines.items():
            features[m].append(extract_features(segment_image(raw, fn(hist))))
        if idx % 10 == 0 or idx == len(dataset):
            print(f"  Processed {idx}/{len(dataset)} MRI samples for feature extraction...")

    labels = np.array(labels)
    summary = []
    plt.figure(figsize=(15, 3))
    for m_idx, method in enumerate(methods):
        X_tr, X_te, y_tr, y_te = train_test_split(np.array(features[method]), labels,
                                                  test_size=0.3, random_state=seed, stratify=labels)
        clf = RandomForestClassifier(n_estimators=50, random_state=seed).fit(X_tr, y_tr)
        y_pred = clf.predict(X_te)
        acc = accuracy_score(y_te, y_pred)
        summary.append({
            "method": method,
            "accuracy": round(float(acc), 4),
            "precision": round(float(precision_score(y_te, y_pred, average="weighted", zero_division=0)), 4),
            "recall": round(float(recall_score(y_te, y_pred, average="weighted", zero_division=0)), 4),
            "f1_score": round(float(f1_score(y_te, y_pred, average="weighted", zero_division=0)), 4),
        })
        plt.subplot(1, 5, m_idx + 1)
        plt.imshow(confusion_matrix(y_te, y_pred), cmap="Blues", interpolation="nearest")
        plt.title(f"{method}\nAcc: {acc * 100:.1f}%", fontsize=9)
        plt.colorbar(fraction=0.046, pad=0.04)
        plt.tight_layout()
    plt.savefig(out("ml_confusion_matrices.png"), dpi=150)
    plt.close()

    with open(ML_CSV_PATH, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["method", "accuracy", "precision", "recall", "f1_score"])
        writer.writeheader()
        writer.writerows(summary)

    plt.figure(figsize=(8, 4.5))
    x = np.arange(len(methods))
    width = 0.35
    plt.bar(x - width / 2, [r["accuracy"] for r in summary], width=width, label="Accuracy", color="#2b5c8f")
    plt.bar(x + width / 2, [r["f1_score"] for r in summary], width=width, label="F1-Score", color="#e07a5f")
    plt.xticks(x, methods, rotation=15, ha="right")
    plt.ylim(0, 1.05)
    plt.ylabel("Score")
    plt.title("Downstream Random Forest Classification Performance")
    plt.legend()
    plt.grid(axis="y", alpha=0.3)
    savefig("ml_classification_performance.png")

    print("\nDownstream ML Results Summary:")
    for r in summary:
        print(f"  {r['method']:<18} | Acc: {r['accuracy']:.4f} | F1: {r['f1_score']:.4f}")
    return summary


# ----------------------------- PSO PARAMETER ANALYSIS -----------------------------
def run_pso_parameter_analysis(hist, seed):
    print("\n" + "=" * 70)
    print("PSO INERTIA-WEIGHT PARAMETER ANALYSIS (w = 0.4, 0.6, 0.8)")
    print("=" * 70)
    plt.figure(figsize=(8, 5))
    for w in [0.4, 0.6, 0.8]:
        t0 = time.time()
        _, fit, curve, _ = pso_optimize(hist, n_thresholds=NUMBER_OF_THRESHOLDS, n_particles=POPULATION_SIZE,
                                        n_iterations=NUMBER_OF_ITERATIONS, w=w, seed=seed)
        dur = time.time() - t0
        c_iter = iters_to_converge(curve, fit)
        plt.plot(curve, label=f"w = {w} (Fit: {fit:.2f}, Iter: {c_iter})", linewidth=2)
        print(f"  w = {w:<4} -> Fitness: {fit:.3f} | Runtime: {dur:.4f}s | Converged at Iter: {c_iter}")
    plt.xlabel("Iteration")
    plt.ylabel("Otsu Between-Class Variance")
    plt.title("PSO Inertia Weight Sensitivity Analysis")
    plt.legend()
    plt.grid(alpha=0.3)
    savefig("pso_parameter_analysis.png")


# ----------------------------- PER-IMAGE PROCESSING -----------------------------
def process_image(image_path, n_runs=10):
    name = os.path.splitext(os.path.basename(image_path))[0]
    print("\n" + "=" * 70)
    print(f"PROCESSING IMAGE: {name} ({n_runs} independent run{'s' if n_runs > 1 else ''})")
    print("=" * 70)

    image = load_grayscale_image(image_path)
    hist = compute_histogram(image)
    n, it, pop = NUMBER_OF_THRESHOLDS, NUMBER_OF_ITERATIONS, POPULATION_SIZE

    runners = {
        "GA": lambda s: ga_optimize(hist, n_thresholds=n, population_size=pop, n_generations=it, seed=s),
        "PSO": lambda s: pso_optimize(hist, n_thresholds=n, n_particles=pop, n_iterations=it, seed=s),
        "DE": lambda s: de_optimize(hist, n_thresholds=n, population_size=pop, n_iterations=it, seed=s),
        "Hybrid PSO-GA": lambda s: hybrid_pso_ga_optimize(hist, n_thresholds=n, swarm_size=pop,
                                                          n_iterations=it, seed=s),
    }
    records = {a: [] for a in ALGORITHMS}
    for run_idx in range(n_runs):
        seed = RANDOM_SEED + run_idx
        for alg in ALGORITHMS:
            t0 = time.time()
            t, f, c, ev = runners[alg](seed)
            records[alg].append({"thresholds": t, "fitness": f, "curve": c, "runtime": time.time() - t0,
                                 "evals": ev, "conv_iter": iters_to_converge(c, f)})

    stats, csv_rows = {}, []
    for alg in ALGORITHMS:
        recs = records[alg]
        fits = [r["fitness"] for r in recs]
        best = recs[int(np.argmax(fits))]
        s = stats[alg] = {
            "mean_fitness": float(np.mean(fits)),
            "best_fitness": float(np.max(fits)),
            "std_fitness": float(np.std(fits)),
            "mean_runtime": float(np.mean([r["runtime"] for r in recs])),
            "mean_conv_iter": float(np.mean([r["conv_iter"] for r in recs])),
            "best_thresholds": sorted(np.round(best["thresholds"], 1).tolist()),
            "best_curve": best["curve"],
            "evals": best["evals"],
        }
        csv_rows.append({
            "image": name, "algorithm": alg,
            "final_fitness": round(s["best_fitness"], 3),
            "thresholds": s["best_thresholds"],
            "runtime_sec": round(s["mean_runtime"], 4),
            "iters_to_converge": int(round(s["mean_conv_iter"])),
            "run": n_runs,
            "mean_fitness": round(s["mean_fitness"], 3),
            "std_fitness": round(s["std_fitness"], 3),
            "function_evaluations": s["evals"],
        })

    print("-" * 75)
    print(f"{'Algorithm':<15} | {'Best Fit':<10} | {'Mean Fit':<10} | {'Std':<8} | {'Mean Time':<10} | {'Thresholds'}")
    print("-" * 75)
    for alg in ALGORITHMS:
        s = stats[alg]
        print(f"{alg:<15} | {s['best_fitness']:<10.3f} | {s['mean_fitness']:<10.3f} | {s['std_fitness']:<8.3f} | {s['mean_runtime']:<9.4f}s | {s['best_thresholds']}")
    print("-" * 75)

    # Convergence: PSO vs GA
    plt.figure(figsize=(8, 5))
    plt.plot(stats["PSO"]["best_curve"], label="PSO", linewidth=2)
    plt.plot(stats["GA"]["best_curve"], label="GA", linewidth=2)
    plt.xlabel("Iteration / Generation")
    plt.ylabel("Best Otsu Fitness")
    plt.title(f"PSO vs GA Convergence - {name}")
    plt.legend()
    plt.grid(alpha=0.3)
    savefig(f"{name}_convergence.png")

    # Convergence: all algorithms
    styles = {"GA": ("#e76f51", "--"), "PSO": ("#2a9d8f", "-."), "DE": ("#e9c46a", ":"), "Hybrid PSO-GA": ("#1d3557", "-")}
    plt.figure(figsize=(9, 5.5))
    for alg in ALGORITHMS:
        col, ls = styles[alg]
        plt.plot(stats[alg]["best_curve"], label=f"{alg} (Final: {stats[alg]['best_fitness']:.2f})",
                 color=col, linestyle=ls, linewidth=2.2)
    plt.xlabel("Iteration / Generation")
    plt.ylabel("Otsu Between-Class Variance")
    plt.title(f"Comparative Evolutionary Convergence - {name}")
    plt.legend()
    plt.grid(alpha=0.3)
    savefig(f"{name}_convergence_all.png")

    # Segmentations
    seg = {a: segment_image(image, stats[a]["best_thresholds"]) for a in ALGORITHMS}

    fig, axes = plt.subplots(1, 3, figsize=(12, 4))
    for ax, (im, title) in zip(axes, [(image, "Original"), (seg["PSO"], "PSO Segmented"), (seg["GA"], "GA Segmented")]):
        ax.imshow(im, cmap="gray")
        ax.set_title(title)
        ax.axis("off")
    savefig(f"{name}_segmented.png")

    fig, axes = plt.subplots(1, 5, figsize=(18, 4))
    panels = [(image, "Original MRI")] + [
        (seg[a], f"{a}\n{stats[a]['best_thresholds']}") for a in ["GA", "PSO", "DE", "Hybrid PSO-GA"]]
    for ax, (im, title) in zip(axes, panels):
        ax.imshow(im, cmap="gray")
        ax.set_title(title, fontsize=10)
        ax.axis("off")
    savefig(f"{name}_segmented_comparison.png")

    return csv_rows, stats


# ----------------------------- MAIN -----------------------------
def main():
    print("\n" + "=" * 70)
    print("HYBRID EVOLUTIONARY OPTIMIZATION FOR MULTILEVEL OTSU MRI SEGMENTATION")
    print("WITH DOWNSTREAM ML VALIDATION")
    print("=" * 70)

    if not os.path.isdir(IMAGE_DIR):
        print(f"\n[ERROR] Baseline dataset folder not found: {IMAGE_DIR}")
        return
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    image_paths = []
    for fn in IMAGE_FILES:
        full = os.path.join(IMAGE_DIR, fn)
        if os.path.isfile(full):
            image_paths.append(full)
        else:
            print(f"[WARNING] Baseline image missing: {full}")
    if not image_paths:
        print("[ERROR] None of the specified baseline images were found.")
        return

    print(f"\nDataset Folder: {IMAGE_DIR}")
    print(f"Output Directory: {OUTPUT_DIR}")
    print(f"CSV Path: {CSV_PATH}")
    print(f"Threshold Count: {NUMBER_OF_THRESHOLDS}")
    print(f"Iterations: {NUMBER_OF_ITERATIONS} | Swarm/Pop Size: {POPULATION_SIZE}")
    print(f"Stochastic Runs: {N_RUNS} (RANDOM_SEED = {RANDOM_SEED})")

    gt_dir = check_ground_truth_masks(IMAGE_DIR, IMAGE_FILES)
    if gt_dir is None:
        print("\nGround truth unavailable — supervised segmentation metrics skipped.")
    else:
        print(f"\nLegitimate ground truth masks detected in: {gt_dir}")

    all_rows, overall = [], {}
    for i, path in enumerate(image_paths, start=1):
        print(f"\n>>> PROCESSING IMAGE {i}/{len(image_paths)}: {os.path.basename(path)}")
        rows, stats = process_image(path, n_runs=N_RUNS)
        all_rows.extend(rows)
        overall[os.path.basename(path)] = stats

    fieldnames = ["image", "algorithm", "final_fitness", "thresholds", "runtime_sec", "iters_to_converge",
                  "run", "mean_fitness", "std_fitness", "function_evaluations"]
    with open(CSV_PATH, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(all_rows)
    print(f"\n[SAVED] Main CSV output written to: {CSV_PATH}")

    def avg(key):
        return [np.mean([overall[img][a][key] for img in overall]) for a in ALGORITHMS]

    x = np.arange(len(ALGORITHMS))

    plt.figure(figsize=(9, 5))
    plt.bar(x, avg("mean_fitness"), yerr=avg("std_fitness"), capsize=6, color=COLORS, alpha=0.85)
    plt.xticks(x, ALGORITHMS)
    plt.ylabel("Mean Otsu Between-Class Variance")
    plt.title(f"Multi-Run Fitness Comparison ({N_RUNS} Independent Runs per Image)")
    plt.grid(axis="y", alpha=0.3)
    savefig("multirun_performance_comparison.png")

    plt.figure(figsize=(8, 4.5))
    plt.bar(x, avg("mean_runtime"), color=COLORS, alpha=0.85)
    plt.xticks(x, ALGORITHMS)
    plt.ylabel("Mean Execution Time (seconds)")
    plt.title("Computational Runtime Comparison")
    plt.grid(axis="y", alpha=0.3)
    savefig("runtime_comparison.png")

    plt.figure(figsize=(9, 4.5))
    for idx, alg in enumerate(ALGORITHMS):
        t1 = [overall[img][alg]["best_thresholds"][0] for img in overall]
        t2 = [overall[img][alg]["best_thresholds"][1] for img in overall]
        plt.scatter([idx - 0.15] * len(t1), t1, color=COLORS[idx], marker="o")
        plt.scatter([idx + 0.15] * len(t2), t2, color=COLORS[idx], marker="s")
    plt.xticks(x, ALGORITHMS)
    plt.ylabel("Intensity Threshold Value (0-255)")
    plt.title("Optimal Threshold Distribution Across Images")
    plt.grid(alpha=0.3)
    savefig("threshold_comparison.png")

    if RUN_PARAMETER_ANALYSIS:
        run_pso_parameter_analysis(compute_histogram(load_grayscale_image(image_paths[0])), seed=RANDOM_SEED)
    else:
        print("\nParameter analysis skipped (RUN_PARAMETER_ANALYSIS = False).")

    if RUN_ML:
        available, valid_classes = check_ml_dataset_availability(IMAGE_DIR)
        if not available:
            ml_status = "ML classification skipped: valid labeled classes are unavailable."
            print("\n" + ml_status)
        else:
            ml_status = f"ML classification executed successfully using {len(valid_classes)} classes."
            run_downstream_ml(valid_classes, NUMBER_OF_THRESHOLDS, RANDOM_SEED)
    else:
        ml_status = "ML classification skipped (RUN_ML = False)."
        print("\n" + ml_status)

    print("\n" + "=" * 70)
    print("EXPERIMENT COMPLETE: FINAL SUMMARY")
    print("=" * 70)
    print("Algorithms Evaluated: GA, PSO, DE, Hybrid PSO-GA")
    print(f"ML Status: {ml_status}")
    print(f"CSV Output: {CSV_PATH}")
    print(f"Generated Visualizations Directory: {OUTPUT_DIR}")
    print("=" * 70)


if __name__ == "__main__":
    main()