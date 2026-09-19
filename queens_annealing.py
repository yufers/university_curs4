# -*- coding: utf-8 -*-
"""
Задача N ферзей методом имитации отжига.
GUI: tkinter + matplotlib.   Запуск:  python queens_annealing.py
Зависимости: pip install matplotlib numpy
"""
import csv
import math
import queue
import random
import threading
import time
import tkinter as tk
from collections import namedtuple
from dataclasses import dataclass
from tkinter import filedialog, messagebox, ttk

import matplotlib
import numpy as np

matplotlib.use("TkAgg")
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.colors import ListedColormap
from matplotlib.figure import Figure
from matplotlib.ticker import MaxNLocator


# ============================================================
#                    ЯДРО АЛГОРИТМА ОТЖИГА
# ============================================================
@dataclass
class Params:
    n: int = 30              # количество ферзей
    t_max: float = 10.0      # максимальная (начальная) температура
    t_min: float = 0.1       # минимальная (конечная) температура
    alpha: float = 0.95      # коэффициент понижения температуры
    steps: int = 100         # шагов при постоянной температуре
    stop_on_zero: bool = True  # остановиться, как только найдено решение


# Снимок состояния после завершения одного температурного уровня
Snapshot = namedtuple(
    "Snapshot", "level temp accepted_bad best_energy best_solution evaluations"
)


def energy(sol):
    """Энергия = число пар ферзей, бьющих друг друга по диагонали.
    Кодировка (перестановка) уже исключает конфликты по строкам и столбцам."""
    n = len(sol)
    d1 = [0] * (2 * n)
    d2 = [0] * (2 * n)
    for col, row in enumerate(sol):
        d1[col + row] += 1
        d2[col - row + n] += 1
    return sum(c * (c - 1) // 2 for c in d1) + sum(c * (c - 1) // 2 for c in d2)


def conflicted_queens(sol):
    """Номера столбцов ферзей, которые находятся под боем (для подсветки)."""
    n = len(sol)
    d1, d2 = {}, {}
    for col, row in enumerate(sol):
        d1[col + row] = d1.get(col + row, 0) + 1
        d2[col - row] = d2.get(col - row, 0) + 1
    return [c for c, r in enumerate(sol) if d1[c + r] > 1 or d2[c - r] > 1]


def anneal(p: Params, rng=random):
    """Генератор: после каждого температурного уровня отдаёт Snapshot."""
    n = p.n
    current = list(range(n))
    rng.shuffle(current)                 # случайное начальное решение
    cur_e = energy(current)
    best, best_e = current[:], cur_e
    temp = p.t_max
    level = 0
    evals = 0

    while temp > p.t_min:
        accepted_bad = 0
        for _ in range(p.steps):
            # «шевеление»: меняем местами двух случайных ферзей
            i = rng.randrange(n)
            j = rng.randrange(n - 1)
            if j >= i:
                j += 1
            current[i], current[j] = current[j], current[i]
            new_e = energy(current)
            evals += 1
            delta = new_e - cur_e

            if delta <= 0:                                   # не хуже -> берём
                cur_e = new_e
            elif rng.random() < math.exp(-delta / temp):     # хуже -> критерий допуска
                cur_e = new_e
                accepted_bad += 1
            else:                                            # отвергаем, откат
                current[i], current[j] = current[j], current[i]

            if cur_e < best_e:
                best, best_e = current[:], cur_e
                if best_e == 0 and p.stop_on_zero:
                    break

        level += 1
        yield Snapshot(level, temp, accepted_bad, best_e, best[:], evals)
        if best_e == 0 and p.stop_on_zero:
            return
        temp *= p.alpha


def run_once(p: Params):
    """Полный прогон без GUI (для экспериментов). Возвращает (энергия, оценок, сек)."""
    t0 = time.perf_counter()
    last = None
    for last in anneal(p):
        pass
    return last.best_energy, last.evaluations, time.perf_counter() - t0


# ============================================================
#                       ЭКСПЕРИМЕНТЫ
# ============================================================
LEVELS_PER_FRAME = 5   # сколько температурных уровней считать между перерисовками

BASE = dict(n=30, t_max=10.0, t_min=0.1, alpha=0.95, steps=100)


def build_experiments():
    """22 комбинации параметров: базовая + изменение по одному параметру + комбинации."""
    def e(**kw):
        d = dict(BASE)
        d.update(kw)
        return Params(**d)

    exps = [e()]                                             # базовый
    exps += [e(alpha=a) for a in (0.80, 0.90, 0.98, 0.99)]   # коэффициент охлаждения
    exps += [e(steps=s) for s in (10, 30, 300, 1000)]        # шагов на температуру
    exps += [e(t_max=t) for t in (0.5, 2.0, 30.0, 100.0)]    # начальная температура
    exps += [e(t_min=t) for t in (0.01, 0.5, 1.0)]           # конечная температура
    exps += [e(n=n) for n in (21, 40, 60)]                   # размер доски
    exps += [e(t_max=30.0, t_min=0.5, alpha=0.98, steps=100)]  # параметры из методички
    exps += [e(alpha=0.80, steps=1000)]                      # быстро остывает, много шагов
    exps += [e(alpha=0.99, steps=10)]                        # медленно остывает, мало шагов
    return exps


# ============================================================
#                            GUI
# ============================================================
class App(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title("Задача N ферзей — метод отжига")
        self.geometry("1350x780")

        self.gen = None
        self.running = False
        self.exp_thread = None
        self.exp_queue = queue.Queue()
        self.exp_stop = threading.Event()

        self._build_left_panel()
        self._build_right_panel()
        self._reset_plots()

    # ---------------- левая панель ----------------
    def _build_left_panel(self):
        left = ttk.Frame(self, padding=8)
        left.pack(side=tk.LEFT, fill=tk.Y)

        box = ttk.LabelFrame(left, text="Параметры алгоритма", padding=8)
        box.pack(fill=tk.X)

        self.vars = {}
        fields = [
            ("n", "Количество ферзей (N)", 30),
            ("t_max", "Максимальная температура", 10),
            ("t_min", "Минимальная температура", 0.1),
            ("alpha", "Коэффициент понижения (α)", 0.95),
            ("steps", "Шагов при постоянной T", 100),
        ]
        for r, (key, label, default) in enumerate(fields):
            ttk.Label(box, text=label).grid(row=r, column=0, sticky="w", pady=3)
            v = tk.StringVar(value=str(default))
            ttk.Entry(box, textvariable=v, width=9).grid(row=r, column=1, padx=(8, 0))
            self.vars[key] = v

        self.stop_on_zero = tk.BooleanVar(value=True)
        ttk.Checkbutton(
            box, text="Остановиться при энергии 0", variable=self.stop_on_zero
        ).grid(row=len(fields), column=0, columnspan=2, sticky="w", pady=(6, 0))

        btns = ttk.Frame(left)
        btns.pack(fill=tk.X, pady=8)
        self.btn_start = ttk.Button(btns, text="▶ Запустить", command=self.start)
        self.btn_start.pack(side=tk.LEFT, expand=True, fill=tk.X)
        self.btn_stop = ttk.Button(btns, text="■ Стоп", command=self.stop, state=tk.DISABLED)
        self.btn_stop.pack(side=tk.LEFT, expand=True, fill=tk.X, padx=(6, 0))

        info = ttk.LabelFrame(left, text="Текущее состояние", padding=8)
        info.pack(fill=tk.X)
        self.status = tk.StringVar(value="Задайте параметры и нажмите «Запустить»")
        ttk.Label(info, textvariable=self.status, wraplength=250, justify="left").pack(anchor="w")

    # ---------------- правая панель ----------------
    def _build_right_panel(self):
        nb = ttk.Notebook(self)
        nb.pack(side=tk.RIGHT, fill=tk.BOTH, expand=True, padx=8, pady=8)

        # --- вкладка 1: решение и графики
        tab1 = ttk.Frame(nb)
        nb.add(tab1, text="Решение и графики")
        self.fig = Figure(figsize=(10, 6.5), constrained_layout=True)
        gs = self.fig.add_gridspec(3, 2, width_ratios=[1, 1.3])
        self.ax_board = self.fig.add_subplot(gs[:, 0])
        self.ax_acc = self.fig.add_subplot(gs[0, 1])
        self.ax_e = self.fig.add_subplot(gs[1, 1], sharex=self.ax_acc)
        self.ax_t = self.fig.add_subplot(gs[2, 1], sharex=self.ax_acc)
        self.canvas = FigureCanvasTkAgg(self.fig, master=tab1)
        self.canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)

        # --- вкладка 2: эксперименты
        tab2 = ttk.Frame(nb, padding=8)
        nb.add(tab2, text="Эксперименты")

        top = ttk.Frame(tab2)
        top.pack(fill=tk.X)
        ttk.Label(top, text="Запусков на эксперимент:").pack(side=tk.LEFT)
        self.runs_var = tk.StringVar(value="10")
        ttk.Entry(top, textvariable=self.runs_var, width=4).pack(side=tk.LEFT, padx=(4, 12))
        self.btn_exp = ttk.Button(top, text="▶ Запустить серию (22 эксперимента)", command=self.start_experiments)
        self.btn_exp.pack(side=tk.LEFT)
        self.btn_exp_stop = ttk.Button(top, text="■ Стоп", command=self.exp_stop.set, state=tk.DISABLED)
        self.btn_exp_stop.pack(side=tk.LEFT, padx=6)
        ttk.Button(top, text="Сохранить в CSV", command=self.save_csv).pack(side=tk.LEFT)

        self.progress = ttk.Progressbar(tab2, mode="determinate")
        self.progress.pack(fill=tk.X, pady=6)
        self.exp_status = tk.StringVar(value="")
        ttk.Label(tab2, textvariable=self.exp_status).pack(anchor="w")

        cols = ("no", "n", "t_max", "t_min", "alpha", "steps", "ok", "avg_e", "min_e", "evals", "time")
        heads = ("№", "N", "T max", "T min", "α", "Шагов", "Решено", "Ср. энергия",
                 "Мин. энергия", "Ср. число оценок", "Ср. время, с")
        widths = (35, 40, 60, 60, 55, 60, 75, 90, 90, 115, 95)
        self.tree = ttk.Treeview(tab2, columns=cols, show="headings", height=22)
        for c, h, w in zip(cols, heads, widths):
            self.tree.heading(c, text=h)
            self.tree.column(c, width=w, anchor="center")
        self.tree.tag_configure("solved", background="#d9f2d9")
        self.tree.tag_configure("part", background="#fff4cc")
        self.tree.tag_configure("fail", background="#f8d7d7")
        self.tree.pack(fill=tk.BOTH, expand=True, pady=6)
        ttk.Label(
            tab2,
            text="Зелёный — решено во всех запусках, жёлтый — в части запусков, красный — ни разу.",
        ).pack(anchor="w")
        self.exp_rows = []

    # ---------------- графики ----------------
    def _reset_plots(self):
        self.h_level, self.h_acc, self.h_best, self.h_temp = [], [], [], []
        for ax, title, color in (
            (self.ax_acc, "Принятые плохие решения (за уровень)", "tab:orange"),
            (self.ax_e, "Энергия лучшего решения", "tab:green"),
            (self.ax_t, "Температура", "tab:red"),
        ):
            ax.clear()
            ax.set_title(title, fontsize=10)
            ax.grid(alpha=0.3)
        self.ax_t.set_xlabel("Номер температурного уровня")
        (self.line_acc,) = self.ax_acc.plot([], [], color="tab:orange")
        (self.line_e,) = self.ax_e.plot([], [], color="tab:green", drawstyle="steps-post")
        (self.line_t,) = self.ax_t.plot([], [], color="tab:red")
        self._draw_board(None, 0, None)
        self._drawn_energy = None
        self.canvas.draw_idle()

    def _draw_board(self, sol, n, best_e):
        ax = self.ax_board
        ax.clear()
        ax.set_xticks([])
        ax.set_yticks([])
        if sol is None:
            ax.set_title("Лучшее решение")
            return
        board = (np.indices((n, n)).sum(axis=0)) % 2
        ax.imshow(board, cmap=ListedColormap(["#f0d9b5", "#b58863"]), interpolation="nearest")
        bad = set(conflicted_queens(sol))
        fs = max(5, min(30, 320 / n))
        for col, row in enumerate(sol):
            ax.text(col, row, "♛", ha="center", va="center", fontsize=fs,
                    color="#c0392b" if col in bad else "black")
        ax.set_title(f"Лучшее решение (энергия = {best_e})\n"
                     + ("✔ конфликтов нет" if best_e == 0 else "красным — ферзи под боем"),
                     fontsize=10)

    def _update_plots(self, snap, n):
        self.h_level.append(snap.level)
        self.h_acc.append(snap.accepted_bad)
        self.h_best.append(snap.best_energy)
        self.h_temp.append(snap.temp)
        for line, data in ((self.line_acc, self.h_acc), (self.line_e, self.h_best), (self.line_t, self.h_temp)):
            line.set_data(self.h_level, data)
        for ax in (self.ax_acc, self.ax_e, self.ax_t):
            ax.relim()
            ax.autoscale_view()
        self.ax_e.set_ylim(bottom=-0.3)
        self.ax_e.yaxis.set_major_locator(MaxNLocator(integer=True))
        # доска меняется только при улучшении лучшего решения — тогда и перерисовываем
        if snap.best_energy != self._drawn_energy:
            self._draw_board(snap.best_solution, n, snap.best_energy)
            self._drawn_energy = snap.best_energy
        self.canvas.draw_idle()

    # ---------------- запуск одиночного прогона ----------------
    def _read_params(self):
        try:
            p = Params(
                n=int(self.vars["n"].get()),
                t_max=float(self.vars["t_max"].get().replace(",", ".")),
                t_min=float(self.vars["t_min"].get().replace(",", ".")),
                alpha=float(self.vars["alpha"].get().replace(",", ".")),
                steps=int(self.vars["steps"].get()),
                stop_on_zero=self.stop_on_zero.get(),
            )
        except ValueError:
            messagebox.showerror("Ошибка", "Проверьте, что все параметры — числа.")
            return None
        if p.n < 4:
            messagebox.showerror("Ошибка", "Количество ферзей должно быть не меньше 4.")
        elif not (0 < p.t_min < p.t_max):
            messagebox.showerror("Ошибка", "Нужно: 0 < T min < T max.")
        elif not (0 < p.alpha < 1):
            messagebox.showerror("Ошибка", "Коэффициент α должен быть в интервале (0; 1).")
        elif p.steps < 1:
            messagebox.showerror("Ошибка", "Число шагов должно быть ≥ 1.")
        else:
            return p
        return None

    def start(self):
        p = self._read_params()
        if p is None:
            return
        self.params = p
        self._reset_plots()
        self.gen = anneal(p)
        self.t_start = time.perf_counter()
        self.running = True
        self.btn_start.config(state=tk.DISABLED)
        self.btn_stop.config(state=tk.NORMAL)
        self.after(1, self._tick)

    def stop(self):
        self.running = False
        self._finish("Остановлено пользователем.")

    def _tick(self):
        if not self.running:
            return
        finished = False
        for _ in range(LEVELS_PER_FRAME):        # несколько уровней за кадр — быстрее анимация
            try:
                snap = next(self.gen)
            except StopIteration:
                finished = True
                break
            self.last = snap
            self._update_plots(snap, self.params.n)
        if finished:
            self._finish("Завершено.")
            return
        self.status.set(
            f"Уровень: {snap.level}\nТемпература: {snap.temp:.4f}\n"
            f"Лучшая энергия: {snap.best_energy}\nПринято плохих на уровне: {snap.accepted_bad}\n"
            f"Оценок решений: {snap.evaluations}"
        )
        self.after(1, self._tick)

    def _finish(self, msg):
        self.running = False
        self.btn_start.config(state=tk.NORMAL)
        self.btn_stop.config(state=tk.DISABLED)
        last = getattr(self, "last", None)
        if last is not None and self.gen is not None:
            res = "РЕШЕНИЕ НАЙДЕНО" if last.best_energy == 0 else f"решение не найдено (энергия {last.best_energy})"
            self.status.set(f"{msg}\n{res}\nВремя: {time.perf_counter() - self.t_start:.2f} с\n"
                            f"Уровней: {last.level}, оценок: {last.evaluations}")
        self.gen = None

    # ---------------- серия экспериментов ----------------
    def start_experiments(self):
        try:
            runs = int(self.runs_var.get())
            assert runs >= 1
        except Exception:
            messagebox.showerror("Ошибка", "Число запусков должно быть целым числом ≥ 1.")
            return
        self.tree.delete(*self.tree.get_children())
        self.exp_rows = []
        self.exp_stop.clear()
        self.btn_exp.config(state=tk.DISABLED)
        self.btn_exp_stop.config(state=tk.NORMAL)
        exps = build_experiments()
        self.progress.config(maximum=len(exps), value=0)
        self.exp_thread = threading.Thread(target=self._exp_worker, args=(exps, runs), daemon=True)
        self.exp_thread.start()
        self.after(150, self._poll_experiments)

    def _exp_worker(self, exps, runs):
        for idx, p in enumerate(exps, 1):
            if self.exp_stop.is_set():
                break
            energies, evals, times = [], [], []
            for _ in range(runs):
                if self.exp_stop.is_set():
                    break
                e, ev, t = run_once(p)
                energies.append(e)
                evals.append(ev)
                times.append(t)
            if energies:
                self.exp_queue.put(("row", idx, p, runs, energies, evals, times))
        self.exp_queue.put(("done",))

    def _poll_experiments(self):
        try:
            while True:
                msg = self.exp_queue.get_nowait()
                if msg[0] == "done":
                    self.btn_exp.config(state=tk.NORMAL)
                    self.btn_exp_stop.config(state=tk.DISABLED)
                    self.exp_status.set("Серия завершена. Результаты можно сохранить в CSV.")
                    return
                _, idx, p, runs, energies, evals, times = msg
                solved = sum(1 for e in energies if e == 0)
                row = (idx, p.n, p.t_max, p.t_min, p.alpha, p.steps,
                       f"{solved}/{len(energies)}",
                       round(sum(energies) / len(energies), 2), min(energies),
                       int(sum(evals) / len(evals)), round(sum(times) / len(times), 2))
                tag = "solved" if solved == len(energies) else ("part" if solved else "fail")
                self.tree.insert("", tk.END, values=row, tags=(tag,))
                self.exp_rows.append(row)
                self.progress.config(value=idx)
                self.exp_status.set(f"Выполнено экспериментов: {idx}")
        except queue.Empty:
            pass
        self.after(150, self._poll_experiments)

    def save_csv(self):
        if not self.exp_rows:
            messagebox.showinfo("Нет данных", "Сначала запустите серию экспериментов.")
            return
        path = filedialog.asksaveasfilename(defaultextension=".csv", filetypes=[("CSV", "*.csv")],
                                            initialfile="experiments.csv")
        if not path:
            return
        with open(path, "w", newline="", encoding="utf-8-sig") as f:
            w = csv.writer(f, delimiter=";")
            w.writerow(["№", "N", "T max", "T min", "alpha", "steps", "Решено", "Ср. энергия",
                        "Мин. энергия", "Ср. число оценок", "Ср. время, с"])
            w.writerows(self.exp_rows)
        messagebox.showinfo("Готово", f"Сохранено: {path}")


if __name__ == "__main__":
    App().mainloop()