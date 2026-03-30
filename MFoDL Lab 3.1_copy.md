## Cell 0 — Title

```markdown
# Lab 3 (OPT-1) — Gradient Descent, Step Size Stability, Armijo Line Search, and 2D Trajectories

**Course:** Mathematical Foundations of Machine and Deep Learning (beginner+)  
**Format:** 1.5h guided lab (Colab/Jupyter)  
**Allowed libs:** numpy, matplotlib (seaborn optional)

**Content description (what we’ll do):**
In this lab, you implement gradient descent (GD) from scratch and explore how the learning rate controls stability and convergence speed.
You compare convex and non-convex objectives using iteration histories and visualizations.
Next, you implement Armijo backtracking line search to adapt step sizes automatically and observe improved stability.
Finally, you move to 2D objectives and visualize trajectories on contour plots to understand conditioning and non-convex basins.
```

---

## Cell 1 — Learning objectives + plan

```markdown
## Learning objectives (you should be able to...)
1. Implement gradient descent in 1D and 2D and return a history of iterates.
2. Explain (visually + intuitively) why too-large learning rates cause oscillation/divergence.
3. Compare convex vs non-convex behavior: sensitivity to learning rate and initialization.
4. Implement Armijo backtracking and interpret the step-size sequence η_k.
5. Diagnose “zig-zag” trajectories caused by ill-conditioning in 2D.

## Minute-by-minute plan (10 / 70 / 10)
**Intro (10 min)**
- GD update rule + intuition (convex vs non-convex)
- Step size stability
- Why line search helps
- 2D view: conditioning ⇒ narrow valleys ⇒ zig-zag

**Hands-on (70 min)**
- Module 1 (20): 1D GD from scratch + learning rate sweep
- Module 2 (20): Armijo backtracking line search (1D)
- Module 3 (25): 2D trajectories (conditioning + non-convex basins)
- Module 4 (5): micro reflection + mini checklist

**Wrap-up (10 min)**
- Key takeaways + GD debugging checklist + concept questions
```

---

## Cell 2 — Imports + global settings

```python
import numpy as np
import matplotlib.pyplot as plt

# (Optional) seaborn is allowed, but not required.
# import seaborn as sns

np.set_printoptions(precision=4, suppress=True)

# For reproducibility in any random choices we might add later
rng = np.random.default_rng(0)
```

---

## Cell 3 — Helper plotting functions (provided)

```python
def plot_1d_function_with_path(f, xs, history, title="", num_points=400):
    """
    Plot f(x) over [min(xs), max(xs)] and overlay GD iterates.
    history: array-like of x_k
    """
    x_grid = np.linspace(xs[0], xs[1], num_points)
    y_grid = np.array([f(x) for x in x_grid])

    plt.figure(figsize=(7, 4))
    plt.plot(x_grid, y_grid, linewidth=2)
    hk = np.array(history, dtype=float)
    plt.scatter(hk, [f(x) for x in hk], s=35)
    plt.plot(hk, [f(x) for x in hk], linestyle="--", linewidth=1)
    plt.title(title)
    plt.xlabel("x")
    plt.ylabel("f(x)")
    plt.grid(True, alpha=0.3)
    plt.show()


def plot_history(history, f, title_prefix=""):
    """
    Two plots:
    - x_k vs k
    - f(x_k) vs k
    """
    hk = np.array(history, dtype=float)
    fk = np.array([f(x) for x in hk])

    plt.figure(figsize=(7, 3.5))
    plt.plot(hk, marker="o")
    plt.title(f"{title_prefix} x_k vs iteration")
    plt.xlabel("k")
    plt.ylabel("x_k")
    plt.grid(True, alpha=0.3)
    plt.show()

    plt.figure(figsize=(7, 3.5))
    plt.plot(fk, marker="o")
    plt.title(f"{title_prefix} f(x_k) vs iteration")
    plt.xlabel("k")
    plt.ylabel("f(x_k)")
    plt.grid(True, alpha=0.3)
    plt.show()


def contour_with_trajectory(f, traj, xlim=(-5, 5), ylim=(-5, 5), levels=30, title=""):
    """
    Contour plot of f(x,y) with a trajectory overlay.
    traj: array-like shape (T, 2)
    """
    traj = np.array(traj, dtype=float)
    xs = np.linspace(xlim[0], xlim[1], 300)
    ys = np.linspace(ylim[0], ylim[1], 300)
    X, Y = np.meshgrid(xs, ys)
    Z = f(X, Y)

    plt.figure(figsize=(6, 5))
    plt.contour(X, Y, Z, levels=levels)
    plt.plot(traj[:, 0], traj[:, 1], marker="o", linewidth=2)
    plt.scatter(traj[0, 0], traj[0, 1], s=90, marker="s", label="start")
    plt.scatter(traj[-1, 0], traj[-1, 1], s=90, marker="*", label="end")
    plt.title(title)
    plt.xlabel("x")
    plt.ylabel("y")
    plt.legend()
    plt.grid(True, alpha=0.2)
    plt.show()
```

---

## Cell 4 — Intro (10 min): short instructor notes

```markdown
## Intro: Gradient Descent in one line

**Update rule (GD):**  
\[
x_{k+1} = x_k - \eta \nabla f(x_k)
\]
- If **f is convex**, GD tends to be well-behaved: step sizes mainly affect speed vs stability.
- If **f is non-convex**, GD is sensitive to **initialization** and can get trapped in different basins/minima.

**Why does too-large η diverge or oscillate?**  
Because you “overstep” the region where the local linear approximation is valid. You can bounce across the minimum and amplify errors.

**Line search idea (Armijo):**  
Choose η adaptively so each step gives a **sufficient decrease** in f. This often stabilizes GD without manually tuning η.

**2D view (conditioning):**  
For ill-conditioned quadratics (narrow valleys), gradients point mostly across the valley, so GD zig-zags instead of going straight to the minimum.
```

---

# Hands-on (70 min)

---

## Cell 5 — Module 1: define 1D functions + gradients (provided)

```python
# MODULE 1 — 1D GD from scratch + learning rate sweep (20 min)

def f1(x):  # convex
    return x**2

def g1(x):
    return 2*x

def f2(x):  # non-convex
    return x**3 - 10*x**2

def g2(x):
    return 3*x**2 - 20*x

def f3(x):  # non-convex
    return x**4 - 10*x**2 - 2*x

def g3(x):
    return 4*x**3 - 20*x - 2


# Quick sanity check: plot functions in interval -5,5
x = np.linspace(-5, 5, 400)

plt.figure(figsize=(9, 5))
plt.plot(x, f1(x), label=r"$f_1(x)=x^2$", linewidth=2)
plt.plot(x, f2(x), label=r"$f_2(x)=x^3-10x^2$", linewidth=2)
plt.plot(x, f3(x), label=r"$f_3(x)=x^4-10x^2-2x$", linewidth=2)
plt.xlim(-5, 5)
plt.axhline(0, color="black", linewidth=1, alpha=0.5)
plt.axvline(0, color="black", linewidth=1, alpha=0.5)
plt.grid(alpha=0.3)
plt.legend()
plt.title("Sanity check: f1, f2, f3 on [-5, 5]")
plt.show()
```

---

## Cell 6 — Module 1 Task 1: implement gradient descent (TODO)

```markdown
### MODULE 1 — Task 1: Implement GD update (1D)

We implement GD that stores all iterates `x_k` in a list called `history`.
This will let us plot both the path on the function and iteration curves.
Keep it simple: fixed learning rate, fixed number of iterations.

**What do you observe?** (after running later)
- With small η: slow but stable movement.
- With a good η: faster convergence.
- With too-large η: oscillation or divergence.

**Expected takeaway:** Step size is a stability–speed tradeoff, and the “safe” range depends on the function curvature.
```

```python
def gradient_descent(start, lr, iters, grad):
    """
    Fixed-step gradient descent in 1D.
    Returns:
      history: list of x_k values (length iters+1)
    """
    x = float(start)
    history = [x]

    for k in range(iters):
        # TODO: implement the GD update x <- x - lr * grad(x)
        # --- SOLUTION (instructor) ---
        # x = x - lr * grad(x)

        # fallback so the notebook runs (student can replace)
        x = x - lr * grad(x)

        history.append(x)

    return history


# Quick smoke test
h_test = gradient_descent(start=5.0, lr=0.1, iters=10, grad=g1)
print("last iterate:", h_test[-1])
```

---

## Cell 7 — Module 1 Task 2: learning rate sweep on convex f1

```markdown
### MODULE 1 — Task 2: Plot f1(x)=x^2 with iterates for different learning rates

We’ll try 2–3 learning rates:
- **small** (slow but stable),
- **good** (fast, stable),
- **too large** (oscillates / diverges).

Use the same start and iterations so comparisons are fair.

**What do you observe?**
- Do iterates move monotonically toward 0?
- Does the path “bounce” across the minimum?

**Expected takeaway:** On a simple convex quadratic, GD behavior is easy to see: too-large η causes bouncing and may not decrease f.
```

```python
start = 6.0
iters = 20
lrs = [0.05, 0.3, 1.1]  # small / good / too large (for x^2, lr>1 typically unstable)

xs_plot = (-8, 8)

for lr in lrs:
    history = gradient_descent(start=start, lr=lr, iters=iters, grad=g1)
    plot_1d_function_with_path(
        f1, xs_plot, history,
        title=f"f1(x)=x^2 | start={start}, lr={lr}, iters={iters}"
    )
```

---

## Cell 8 — Module 1 Task 3: plot x_k and f(x_k) curves for f1

```markdown
### MODULE 1 — Task 3: Plot iteration histories (x_k and f(x_k))

Iteration plots are a debugging superpower:
- `x_k vs k` shows oscillations or slow drift.
- `f(x_k) vs k` shows whether we actually decrease the objective.

**What do you observe?**
- Which lr decreases f fastest?
- For the unstable lr, does f explode?

**Expected takeaway:** Always plot `f(x_k)` when diagnosing optimization: the path might “look okay” but f might not decrease.
```

```python
for lr in lrs:
    history = gradient_descent(start=start, lr=lr, iters=iters, grad=g1)
    plot_history(history, f1, title_prefix=f"[f1] lr={lr} |")
```

---

## Cell 9 — Module 1 Task 4: repeat for non-convex + compare initializations

```markdown
### MODULE 1 — Task 4: Non-convex sensitivity (f2 or f3)

Non-convex objectives can have:
- multiple critical points (minima, maxima, saddles),
- basins of attraction that depend on start,
- more fragile stability ranges for η.

We will run GD on a non-convex function and compare:
1) different learning rates, and  
2) different starting points.

**What do you observe?**
- Do you end up in different places for different starts?
- Does the “good” lr for convex f1 still look good here?

**Expected takeaway:** Non-convex GD is more sensitive: initialization + step size can change the outcome dramatically.
```

```python
# Choose non-convex function: f2 or f3
f_nc, g_nc, name = f3, g3, "f3(x)=x^4 - 10x^2 - 2x"

starts = [-3.0, 3.0]
lrs_nc = [0.001, 0.01, 0.05]  # keep small-ish; non-convex polynomials can blow up quickly
iters_nc = 60
xs_plot_nc = (-4.5, 4.5)

for start in starts:
    for lr in lrs_nc:
        history = gradient_descent(start=start, lr=lr, iters=iters_nc, grad=g_nc)
        plot_1d_function_with_path(
            f_nc, xs_plot_nc, history,
            title=f"{name} | start={start}, lr={lr}, iters={iters_nc}"
        )
        plot_history(history, f_nc, title_prefix=f"[{name}] start={start}, lr={lr} |")
```

---

## Cell 10 — Module 1 mini reflection

```markdown
### MODULE 1 — Quick reflection (30–60 seconds)

**What do you observe?**
- On f1, what is the “too large” lr signature in `x_k` and `f(x_k)`?
- On f3, do two different starts converge to different regions/minima?

**Expected takeaway:** Always connect *plots* (trajectory + f(x_k)) to *diagnosis* (stable/unstable, slow/fast, basin dependence).
```

---

## Cell 11 — Module 2: Armijo backtracking (setup)

```markdown
## MODULE 2 — Armijo backtracking line search (20 min)

Goal: instead of picking a fixed η by hand, we choose η adaptively each step.

**Armijo condition (informal):** choose η so that
\[
f(x - \eta g) \le f(x) - c \eta \|g\|^2
\]
In 1D, \|g\|^2 is just g^2.  
If the condition fails, shrink η ← β η and try again (backtracking).

We will:
1) implement `armijo_backtracking(...)` (1D),
2) implement GD that uses it, and
3) compare fixed-lr vs Armijo on a non-convex function.
```

---

## Cell 12 — Module 2 Task 1: implement Armijo backtracking (TODOs)

```markdown
### MODULE 2 — Task 1: Implement Armijo backtracking (1D)

We’ll implement a function that returns a step size η.
Inputs:
- f: objective
- x: current point
- g: gradient at x (a scalar in 1D)
- eta0: initial trial step size
- c: Armijo parameter (small, like 1e-4)
- beta: shrink factor (e.g., 0.5)
- max_steps: stop after some shrink attempts

**What do you observe?** (later)
- When gradients are large or curvature is harsh, η shrinks.
- When things are smooth, η can stay near eta0.

**Expected takeaway:** Line search automates stability: “if decrease is not sufficient, reduce step”.
```

```python
def armijo_backtracking(f, x, g, eta0=1.0, c=1e-4, beta=0.5, max_steps=25):
    """
    1D Armijo backtracking.
    Returns:
      eta: accepted step size
      steps: number of backtracking reductions performed
    """
    eta = float(eta0)
    fx = float(f(x))

    # In 1D, directional step is x_new = x - eta * g
    # Armijo sufficient decrease: f(x_new) <= f(x) - c * eta * (g^2)

    steps = 0
    for j in range(max_steps):
        x_new = x - eta * g
        # TODO: implement the Armijo condition check
        # --- SOLUTION (instructor) ---
        # if f(x_new) <= fx - c * eta * (g**2):
        #     return eta, steps

        # fallback so the notebook runs
        if f(x_new) <= fx - c * eta * (g**2):
            return eta, steps

        # TODO: backtracking update eta *= beta if condition fails
        # --- SOLUTION (instructor) ---
        # eta = eta * beta

        # fallback so the notebook runs
        eta = eta * beta
        steps += 1

    return eta, steps  # return the final (small) eta if max_steps reached


# Smoke test on f1 at x=10
eta, steps = armijo_backtracking(f1, x=10.0, g=g1(10.0), eta0=1.0)
print("eta:", eta, "backtracking steps:", steps)
```

---

## Cell 13 — Module 2 Task 2: implement GD with Armijo (TODOs) + collect eta_k

```markdown
### MODULE 2 — Task 2: GD with Armijo (track η_k)

We implement GD that calls Armijo each iteration and stores:
- `history`: x_k
- `eta_history`: η_k

This allows us to visualize:
- path on f(x),
- f(x_k) vs k,
- η_k vs k.

**What do you observe?**
- Does η shrink early then stabilize?
- Does Armijo prevent divergence compared to a fixed lr?

**Expected takeaway:** Adaptive step sizes can make GD robust when fixed lr is hard to tune.
```

```python
def gd_with_armijo(start, iters, f, grad, eta0=1.0, c=1e-4, beta=0.5, max_steps=25):
    x = float(start)
    history = [x]
    eta_history = []

    for k in range(iters):
        g = float(grad(x))
        eta, _ = armijo_backtracking(f, x, g, eta0=eta0, c=c, beta=beta, max_steps=max_steps)
        eta_history.append(eta)

        # TODO: GD update using the Armijo-chosen eta
        # --- SOLUTION (instructor) ---
        # x = x - eta * g

        # fallback so the notebook runs
        x = x - eta * g

        history.append(x)

    return history, eta_history


# Smoke test
hA, eA = gd_with_armijo(start=6.0, iters=20, f=f1, grad=g1, eta0=1.0)
print("last x:", hA[-1], "eta first/last:", eA[0], eA[-1])
```

---

## Cell 14 — Module 2 Task 3: compare fixed lr vs Armijo on non-convex (plots)

```markdown
### MODULE 2 — Task 3: Compare fixed-lr GD vs Armijo on a non-convex objective

We’ll use the same start point for a fair comparison.
You should see that:
- fixed lr might be too aggressive (blow up) or too small (slow),
- Armijo adapts η_k to maintain sufficient decrease.

We’ll plot:
1) f(x) with iterates
2) f(x_k) vs k
3) η_k vs k (Armijo only)

**What do you observe?**
- Does Armijo reduce large steps in steep regions?
- Compared to fixed lr, is the objective curve more stable?

**Expected takeaway:** Armijo often “saves” you from instability with minimal manual tuning.
```

```python
# Choose a non-convex function for comparison
f_cmp, g_cmp, name = f2, g2, "f2(x)=x^3 - 10x^2"
start = 8.0
iters = 60
xs_plot = (-2, 12)

# Fixed learning rates to compare
fixed_lrs = [0.001, 0.02]  # one conservative, one potentially aggressive

# Run fixed-lr GD
hist_fixed = {}
for lr in fixed_lrs:
    hist_fixed[lr] = gradient_descent(start=start, lr=lr, iters=iters, grad=g_cmp)

# Run Armijo GD
hist_armijo, eta_hist = gd_with_armijo(
    start=start, iters=iters, f=f_cmp, grad=g_cmp,
    eta0=1.0, c=1e-4, beta=0.5, max_steps=25
)

# Plot function + paths
for lr in fixed_lrs:
    plot_1d_function_with_path(
        f_cmp, xs_plot, hist_fixed[lr],
        title=f"{name} | Fixed lr={lr} | start={start}"
    )

plot_1d_function_with_path(
    f_cmp, xs_plot, hist_armijo,
    title=f"{name} | Armijo (eta0=1.0, beta=0.5) | start={start}"
)

# Plot objective histories
for lr in fixed_lrs:
    plot_history(hist_fixed[lr], f_cmp, title_prefix=f"[{name}] fixed lr={lr} |")

plot_history(hist_armijo, f_cmp, title_prefix=f"[{name}] Armijo |")

# Plot eta_k
plt.figure(figsize=(7, 3.5))
plt.plot(eta_hist, marker="o")
plt.title(f"{name} | Armijo step sizes η_k")
plt.xlabel("k")
plt.ylabel("η_k")
plt.grid(True, alpha=0.3)
plt.show()
```

---

## Cell 15 — Module 2 “What do you observe?” + expected takeaway

```markdown
### MODULE 2 — Discussion

**What do you observe?**
- When does η_k shrink the most? (early iterations? near steep regions?)
- Does Armijo produce smoother f(x_k) decrease than fixed lr?
- Did fixed lr ever increase f(x_k) or diverge?

**Expected takeaway:**
Armijo is a practical “auto-tuner” that enforces sufficient decrease. It can be slower per-iteration (extra function calls),
but it reduces the risk of catastrophic step sizes when the landscape changes.
```

---

## Cell 16 — Module 3: 2D functions (ill-conditioned quadratic + Himmelblau) (provided)

```python
# MODULE 3 — 2D trajectories: conditioning and non-convexity (25 min)

# (A) Ill-conditioned quadratic: f(x,y) = 0.5*(a x^2 + b y^2), a != b
def make_ill_conditioned_quadratic(a=50.0, b=1.0):
    def f(x, y):
        return 0.5*(a*x**2 + b*y**2)

    def grad(xy):
        x, y = xy[0], xy[1]
        return np.array([a*x, b*y], dtype=float)

    return f, grad


# (B) Non-convex: Himmelblau function
# f(x,y) = (x^2 + y - 11)^2 + (x + y^2 - 7)^2
def himmelblau(x, y):
    return (x**2 + y - 11)**2 + (x + y**2 - 7)**2

def grad_himmelblau(xy):
    x, y = xy[0], xy[1]
    # Provided gradient (we’ll add a small TODO later to sanity-check with finite differences)
    dfdx = 4*x*(x**2 + y - 11) + 2*(x + y**2 - 7)
    dfdy = 2*(x**2 + y - 11) + 4*y*(x + y**2 - 7)
    return np.array([dfdx, dfdy], dtype=float)


# Quick check of Himmelblau values at known minima-ish points (should be near 0)
test_pts = [(3,2), (-2.805118,3.131312), (-3.779310,-3.283186), (3.584428,-1.848126)]
for p in test_pts:
    print(p, himmelblau(p[0], p[1]))
```

---

## Cell 17 — Module 3 Task 1: implement 2D GD (TODO)

```markdown
### MODULE 3 — Task 1: Implement GD in 2D (trajectory)

We implement GD for vectors in ℝ² and store a trajectory:
\[
\mathbf{x}_{k+1} = \mathbf{x}_k - \eta \nabla f(\mathbf{x}_k)
\]

**What do you observe?** (later with plots)
- On ill-conditioned quadratics: zig-zag in a narrow valley.
- On non-convex Himmelblau: different starts can converge to different minima.

**Expected takeaway:** In 2D, *geometry* (contours) explains optimization behavior.
```

```python
def gd_2d(start_xy, lr, iters, grad):
    """
    Fixed-step GD in 2D.
    Returns:
      traj: list of points (each is shape (2,))
    """
    x = np.array(start_xy, dtype=float)
    traj = [x.copy()]

    for k in range(iters):
        g = grad(x)

        # TODO: implement the 2D update x <- x - lr * g
        # --- SOLUTION (instructor) ---
        # x = x - lr * g

        # fallback so the notebook runs
        x = x - lr * g

        traj.append(x.copy())

    return traj


# Smoke test on a simple quadratic
fQ, gQ = make_ill_conditioned_quadratic(a=10.0, b=1.0)
traj_test = gd_2d(start_xy=(2, 2), lr=0.1, iters=10, grad=gQ)
print("last point:", traj_test[-1])
```

---

## Cell 18 — Module 3 Task 2: ill-conditioning zig-zag experiment (contours + trajectory)

```markdown
### MODULE 3 — Task 2: Ill-conditioned quadratic ⇒ zig-zag

We use:
\[
f(x,y) = \tfrac12(ax^2 + by^2), \quad a \gg b
\]
Contours are long ellipses (narrow valley).  
GD often moves across the valley, overshoots, and zig-zags.

**Experiment:**
- Fix (a,b) = (50, 1)
- Start at (x,y) = (3, 3)
- Try a learning rate that is stable but shows zig-zag.

**What do you observe?**
- Does the path alternate sides of the valley?
- Is progress along the valley slow?

**Expected takeaway:** Ill-conditioning (a/b large) makes GD inefficient with a single global learning rate.
```

```python
f_ic, g_ic = make_ill_conditioned_quadratic(a=50.0, b=1.0)

start_xy = (3.0, 3.0)
iters = 25
lr = 0.05  # try adjusting if needed

traj = gd_2d(start_xy=start_xy, lr=lr, iters=iters, grad=g_ic)

# TODO: call contour_with_trajectory(...) to visualize the trajectory
# --- SOLUTION (instructor) ---
# contour_with_trajectory(
#     f=lambda X, Y: f_ic(X, Y),
#     traj=traj,
#     xlim=(-4, 4), ylim=(-4, 4),
#     levels=30,
#     title=f"Ill-conditioned quadratic (a=50,b=1) | lr={lr}"
# )

# fallback so the notebook runs
contour_with_trajectory(
    f=lambda X, Y: f_ic(X, Y),
    traj=traj,
    xlim=(-4, 4), ylim=(-4, 4),
    levels=30,
    title=f"Ill-conditioned quadratic (a=50,b=1) | lr={lr}"
)
```

---

## Cell 19 — Module 3 Task 3: effect of changing lr (and a “preconditioning hint”)

```markdown
### MODULE 3 — Task 3: Change lr (and a hint about scaling)

Try 2–3 learning rates and compare trajectories:
- too small: very slow
- moderate: zig-zag but converges
- too large: divergence or wild bouncing

**Preconditioning hint (not required):**
If x-direction curvature is much larger (a≫b), then scaling coordinates or using a diagonal preconditioner can reduce zig-zag.
We won’t implement full preconditioning here, but notice the symptom.

**What do you observe?**
- Is there a single lr that is “great” in both directions?
- What happens to zig-zag as lr changes?

**Expected takeaway:** A single lr struggles when curvature differs by orders of magnitude.
```

```python
lrs_try = [0.01, 0.05, 0.12]
for lr in lrs_try:
    traj = gd_2d(start_xy=start_xy, lr=lr, iters=iters, grad=g_ic)
    contour_with_trajectory(
        f=lambda X, Y: f_ic(X, Y),
        traj=traj,
        xlim=(-4, 4), ylim=(-4, 4),
        levels=30,
        title=f"Ill-conditioned quadratic (a=50,b=1) | lr={lr}"
    )
```

---

## Cell 20 — Module 3 Task 4: gradient sanity-check via finite differences (TODO)

```markdown
### MODULE 3 — Task 4: Sanity-check a gradient (finite differences)

Even small gradient bugs can ruin optimization.
We’ll verify the Himmelblau gradient numerically at one point using finite differences.

**What do you observe?**
- Is the numerical gradient close to the analytic one?
- If not, something is likely wrong in the derivative expressions.

**Expected takeaway:** Gradient checking is a basic safety tool when implementing optimization from scratch.
```

```python
def finite_diff_grad_2d(f_scalar, xy, eps=1e-6):
    x, y = float(xy[0]), float(xy[1])
    fx = f_scalar(x, y)

    # central differences
    dfdx = (f_scalar(x + eps, y) - f_scalar(x - eps, y)) / (2*eps)
    dfdy = (f_scalar(x, y + eps) - f_scalar(x, y - eps)) / (2*eps)
    return np.array([dfdx, dfdy], dtype=float)

xy0 = np.array([0.1, -0.2], dtype=float)
g_analytic = grad_himmelblau(xy0)
g_numeric = finite_diff_grad_2d(himmelblau, xy0, eps=1e-6)

print("analytic:", g_analytic)
print("numeric :", g_numeric)

# TODO: compute and print the relative error norm ||ga-gn|| / (||ga||+||gn||+1e-12)
# --- SOLUTION (instructor) ---
# rel_err = np.linalg.norm(g_analytic - g_numeric) / (np.linalg.norm(g_analytic) + np.linalg.norm(g_numeric) + 1e-12)
# print("relative error:", rel_err)

# fallback so the notebook runs
rel_err = np.linalg.norm(g_analytic - g_numeric) / (np.linalg.norm(g_analytic) + np.linalg.norm(g_numeric) + 1e-12)
print("relative error:", rel_err)
```

---

## Cell 21 — Module 3 Task 5: non-convex basins on Himmelblau (two initializations)

```markdown
### MODULE 3 — Task 5: Non-convex basins (Himmelblau) with two initializations

Himmelblau has multiple minima. GD can converge to different minima depending on start.

**Experiment:**
- choose lr small enough to be stable
- run GD from two different initial points
- plot trajectories on the contour plot

**What do you observe?**
- Do trajectories end in different minima locations?
- Does one start take a “longer path”?

**Expected takeaway:** In non-convex landscapes, initialization can decide which solution you get.
```

```python
iters = 60
lr = 0.01  # Himmelblau can be sensitive; keep it small
starts = [(-3.5, -3.0), (3.5, 0.5)]

for s in starts:
    traj = gd_2d(start_xy=s, lr=lr, iters=iters, grad=grad_himmelblau)
    contour_with_trajectory(
        f=lambda X, Y: himmelblau(X, Y),
        traj=traj,
        xlim=(-5, 5), ylim=(-5, 5),
        levels=40,
        title=f"Himmelblau | start={s} | lr={lr}"
    )
```

---

## Cell 22 — Module 3 discussion

```markdown
### MODULE 3 — Discussion

**What do you observe?**
- Ill-conditioned quadratic: what “shape” causes zig-zag? (look at contours!)
- Himmelblau: did different starts land in different minima?
- How does lr influence whether you overshoot a basin boundary?

**Expected takeaway:**
Contours are a geometric explanation of optimization. Ill-conditioning creates narrow valleys and zig-zag;
non-convexity creates multiple basins, so initialization matters.
```

---

## Cell 23 — Module 4: Micro reflection + mini checklist (5 min)

```markdown
## MODULE 4 — Micro reflection + mini checklist (5 min)

**Reflection questions (answer in 1–2 sentences each):**
1) Why did Armijo help compared to a fixed learning rate?  
2) What does zig-zag motion tell you about the function’s curvature/conditioning?  
3) In non-convex objectives, why can two starts converge to different outcomes?

**Mini checklist (diagnosing unstable GD):**
- [ ] Try a learning-rate sweep (10× smaller and 10× larger).
- [ ] Plot f(x_k) vs k (is it decreasing? exploding?).
- [ ] Plot x_k vs k (oscillation? stagnation?).
- [ ] Check gradient magnitudes (are they huge?).
- [ ] If unstable: reduce lr or use line search.
```

---

# Wrap-up (10 min)

---

## Cell 24 — Key takeaways (6–8 bullets)

```markdown
## Key takeaways
- GD uses the local gradient direction: \(x_{k+1}=x_k-\eta\nabla f(x_k)\).
- Learning rate η controls a stability–speed tradeoff; too large can oscillate/diverge.
- Convex quadratics provide “clean” intuition: stable η decreases f smoothly.
- Non-convex objectives can have multiple basins; initialization matters.
- Armijo backtracking chooses η adaptively to ensure sufficient decrease, improving robustness.
- In 2D, contour geometry explains behavior: narrow valleys (ill-conditioning) cause zig-zag.
- Visualization is a primary debugging tool: always plot iterates + f(x_k) and (if using Armijo) η_k.
- Gradient checking with finite differences helps prevent subtle implementation bugs.
```

---

## Cell 25 — GD debugging checklist (≥10 bullets)

```markdown
## GD debugging checklist (practical)
Use this when your optimization “looks wrong”:

1. Do a learning-rate sweep: try {η/10, η, 10η}.  
2. Plot `f(x_k)` vs k: should decrease most iterations (small bumps can happen in non-convex).  
3. Plot `x_k` (or ‖x_k‖) vs k: look for oscillation or divergence.  
4. Plot gradient norms ‖∇f(x_k)‖: exploding gradients often imply too-large η.  
5. Check your gradient implementation (finite differences at random points).  
6. If non-convex: try multiple starts; compare which basin/minimum you reach.  
7. Reduce η if you see sign-flipping oscillations across a minimum.  
8. If ill-conditioned: expect zig-zag; try smaller η (or consider scaling/preconditioning).  
9. Consider Armijo line search to avoid catastrophic step sizes.  
10. Track step sizes η_k (if adaptive): do they collapse to ~0? (maybe c too strict or beta too small).  
11. Check for numerical overflow (polynomials can explode): clamp domain or reduce η.  
12. Confirm your stopping logic (iters, tolerances) isn’t ending too early.
```

---

## Cell 26 — Concept questions (4–6)

```markdown
## Concept questions (answer without code)
1. Convex vs non-convex: why is GD typically easier to reason about on convex functions?
2. Intuitively, why does too-large learning rate cause divergence or oscillation?
3. What does the Armijo condition try to guarantee (informally), and how does backtracking enforce it?
4. Why does ill-conditioning (narrow valley) produce zig-zag trajectories in 2D?
5. In non-convex optimization, how and why does initialization affect the final solution?
6. Why is plotting `f(x_k)` often more informative than plotting `x_k` alone?
```

---

## Cell 27 — End-of-lab checklist (student-facing)

```markdown
## End-of-lab checklist
- [ ] I can implement 1D GD and return a `history` of iterates.
- [ ] I can explain (with plots) how learning rate affects stability and speed.
- [ ] I can compare convex and non-convex behavior and describe sensitivity to initialization.
- [ ] I can implement Armijo backtracking and plot η_k over iterations.
- [ ] I can implement 2D GD and plot trajectories on contour plots.
- [ ] I can recognize zig-zag as a sign of ill-conditioning.
```
