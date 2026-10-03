"""Interactive RGB curve fitting. Run with: python gui_rgb_fit.py."""
from datetime import datetime
from pathlib import Path
import warnings

import numpy as np
from numpy.polynomial import Polynomial
import matplotlib.pyplot as plt
from matplotlib import font_manager
from matplotlib.widgets import Button
from scipy.interpolate import PPoly, PchipInterpolator

MAX_POLY_DEG = 5
IMG_W, IMG_H = 1000, 600
DUPLICATE_X_TOL = 1e-8


def configure_font():
    """Use an installed Chinese font, or readable English labels."""
    installed = {font.name for font in font_manager.fontManager.ttflist}
    for name in ("Microsoft YaHei", "SimHei", "Noto Sans CJK SC",
                 "Source Han Sans SC", "WenQuanYi Micro Hei", "PingFang SC"):
        if name in installed:
            plt.rcParams["font.sans-serif"] = [name, "DejaVu Sans"]
            plt.rcParams["axes.unicode_minus"] = False
            return True
    return False


def generate_rgb_image(w=IMG_W, h=IMG_H):
    xv, yv = np.meshgrid(np.linspace(0, 1, w), np.linspace(1, 0, h))
    return np.stack((xv, yv, 1 - xv * yv), axis=-1)


def _merge_duplicate_x(points):
    """Keep the last point when x coordinates are equal within UI tolerance."""
    merged = []
    for x, y in points:
        matches = [i for i, (old_x, _) in enumerate(merged)
                   if abs(old_x - x) <= DUPLICATE_X_TOL]
        if matches:
            first = matches[0]
            merged[first] = [x, y]
            for i in reversed(matches[1:]):
                merged.pop(i)
        else:
            merged.append([x, y])
    return np.asarray(merged, dtype=float)


def fit_curve(points, mode="spline"):
    """Return a piecewise polynomial for y(x); input order is irrelevant."""
    pts = np.asarray(points, dtype=float)
    if pts.ndim != 2 or pts.shape[1] != 2 or len(pts) < 2:
        raise ValueError("At least two points are required.")
    if not np.isfinite(pts).all() or np.any((pts < 0) | (pts > 1)):
        raise ValueError("Points must be finite and within [0, 1].")
    pts = _merge_duplicate_x(pts)
    if len(pts) < 2:
        raise ValueError("At least two distinct x coordinates are required.")
    pts = pts[np.argsort(pts[:, 0])]
    x, y = pts.T
    if mode == "spline":
        return PchipInterpolator(x, y, extrapolate=False)
    if mode != "poly":
        raise ValueError("Unknown fit mode.")
    # Scaling the domain avoids fitting directly to badly scaled powers of x.
    with warnings.catch_warnings():
        warnings.simplefilter("error", np.exceptions.RankWarning)
        p = Polynomial.fit(x, y, min(MAX_POLY_DEG, len(x) - 1)).convert()
    return PPoly(p.coef[::-1, None], [0.0, 1.0], extrapolate=True)


def evaluate_y(curve, x):
    """Evaluate upward-increasing y with endpoint hold and a domain guard."""
    x = np.asarray(x, dtype=float)
    held_x = np.clip(x, float(curve.x[0]), float(curve.x[-1]))
    return np.clip(curve(held_x), 0, 1)


def curve_rgb(curve, x):
    """Exact RGB of the analytical gradient from the displayed curve y."""
    red = np.clip(np.asarray(x, dtype=float), 0, 1)
    green = evaluate_y(curve, red)
    blue = 1 - red * green
    return np.stack((red, green, blue), axis=-1)


def format_number(value):
    """Use the shortest decimal that round-trips to the same float."""
    value = float(value)
    if value == 0:
        return "0"
    return str(int(value)) if value.is_integer() else repr(value)


def format_table(headers, rows):
    """Return a compact aligned plain-text table."""
    text_rows = [[str(value) for value in row] for row in rows]
    widths = [len(header) for header in headers]
    for row in text_rows:
        widths = [max(width, len(value)) for width, value in zip(widths, row)]

    def render(row):
        return " | ".join(value.rjust(width) for value, width in zip(row, widths))

    return [render(headers), "-+-".join("-" * width for width in widths),
            *(render(row) for row in text_rows)]


def expression_text(curve, mode, points):
    display_mode = "PCHIP" if mode == "spline" else mode
    boundary_policy = ("Hold y at the nearest sampled endpoint"
                       if mode == "spline" else "Evaluate the polynomial on 0 <= x <= 1")
    lines = [
        "RGB CURVE FIT EXPORT / RGB 曲线拟合导出",
        "=" * 72,
        "",
        "[SUMMARY / 摘要]",
        f"Mode: {display_mode}",
        "Domain / 定义域          : 0 <= x <= 1",
        "Coordinates / 坐标: x increases rightward; y increases upward.",
        f"Boundary / 边界         : {boundary_policy}",
        "",
        "[RGB MAPPING / RGB 映射]",
        "Normalized / 归一化 (0-1):",
        "  R(x) = x",
        "  G(x) = y(x)",
        "  B(x) = 1 - x*y(x)",
        "",
        "8-bit channels / 8 位通道 (0-255):",
        "  R_255(x) = 255*x",
        "  G_255(x) = 255*y(x)",
        "  B_255(x) = 255*(1 - x*y(x))",
        "  Round the result if integer channels are needed / 需要整数时四舍五入。",
        "",
        "[CURVE DEFINITION / 曲线定义]",
        "x is the horizontal coordinate and red channel R; y(x) is the fitted curve",
        "height and green channel G. Each x therefore selects one RGB color.",
        "x 是横坐标，也是红色通道 R；y(x) 是拟合曲线的高度，也是绿色通道 G。",
        "给定一个 x，即可得到曲线上的一个 RGB 颜色。",
        "",
        "How to calculate y(x) / 如何计算 y(x):",
        "1. Find the coefficient-table row containing x; x0 and x1 are its bounds.",
        "   在下方系数表中找到包含 x 的一行；x0、x1 是该段的左右端点。",
        "2. Set t = (x - x0)/(x1 - x0), which maps that segment to 0 <= t <= 1.",
        "   用 t 把这一段映射到 0～1；a0、a1、... 是同一行的系数。",
        "3. Substitute t and the row's coefficients into raw_y(x) below.",
        "   raw_y(x) 是裁剪前的拟合结果，不是 RGB 通道值。",
        "4. Apply clip to obtain the displayed y(x), then use G=y(x) and B=1-x*y(x).",
        "   clip 把结果限制在 0～1；最终用 y(x) 计算 G 和 B。",
        "",
        "clip(v, 0, 1) = min(1, max(0, v))",
        "t = (x - x0)/(x1 - x0)",
    ]

    degree = len(curve.c) - 1
    powers = np.arange(degree, -1, -1)
    template_terms = [f"a{power}" if power == 0 else
                      f"a{power}*t" if power == 1 else f"a{power}*t**{power}"
                      for power in powers]
    lines.append("raw_y(x) = " + " + ".join(template_terms))
    lines.append("y(x) = clip(raw_y(x), 0, 1)")

    if mode == "spline":
        first_x, last_x = float(curve.x[0]), float(curve.x[-1])
        first_y, last_y = float(curve(first_x)), float(curve(last_x))
        lines += ["", "Endpoint hold / 端点保持:"]
        if first_x > 0:
            lines.append(f"  0 <= x < {format_number(first_x)}: "
                         f"y(x) = {format_number(first_y)}")
        if last_x < 1:
            lines.append(f"  {format_number(last_x)} < x <= 1: "
                         f"y(x) = {format_number(last_y)}")

    rows = []
    for i in range(len(curve.x) - 1):
        start, end = float(curve.x[i]), float(curve.x[i + 1])
        width = end - start
        normalized = curve.c[:, i] * width ** powers
        rows.append([str(i + 1), format_number(start), format_number(end),
                     *(format_number(value) for value in normalized)])
    section = "PCHIP SEGMENTS" if mode == "spline" else "POLYNOMIAL SEGMENTS"
    coefficient_headers = [f"a{power}" for power in powers]
    lines += ["", f"[{section} / 分段系数]",
              "Use the row whose x0 <= x <= x1 / 使用包含 x 的对应行："]
    lines += format_table(["#", "x0", "x1", *coefficient_headers], rows)

    effective_points = _merge_duplicate_x(np.asarray(points, dtype=float))
    effective_points = effective_points[np.argsort(effective_points[:, 0])]
    fitted_y = evaluate_y(curve, effective_points[:, 0])
    point_rgb = curve_rgb(curve, effective_points[:, 0])
    if np.allclose(effective_points[:, 1], fitted_y, rtol=0, atol=1e-12):
        point_rows = [[str(i + 1), format_number(x), format_number(y_fit),
                       *(format_number(value) for value in rgb)]
                      for i, ((x, _), y_fit, rgb) in
                      enumerate(zip(effective_points, fitted_y, point_rgb))]
        lines += ["", "[SELECTED POINTS / 取点]",
                  "The curve passes through each point: y = G.",
                  "曲线经过每个取点：y = G。"]
        lines += format_table(["#", "x", "y", "R", "G", "B"], point_rows)
    else:
        point_rows = [[str(i + 1), format_number(x), format_number(y),
                       format_number(y_fit),
                       *(format_number(value) for value in rgb)]
                      for i, ((x, y), y_fit, rgb) in
                      enumerate(zip(effective_points, fitted_y, point_rgb))]
        lines += ["", "[SELECTED POINTS / 取点]",
                  "The polynomial approximates these points; RGB uses fitted y_curve.",
                  "多项式近似这些取点；RGB 使用拟合后的 y_curve。"]
        lines += format_table(["#", "x", "y_point", "y_curve", "R", "G", "B"],
                              point_rows)
    return "\n".join(lines) + "\n"


class RGBFitApp:
    def __init__(self, output_dir=None):
        self.chinese = configure_font()
        self.output_dir = Path(output_dir) if output_dir else Path(__file__).resolve().parent / "output"
        self.points = []
        self.fit_mode = "spline"
        self.curve = None
        self.fig, self.ax = plt.subplots(figsize=(10, 6))
        self.fig.subplots_adjust(left=0.08, right=0.96, top=0.89, bottom=0.25)
        self.fig.canvas.manager.set_window_title("RGB Curve Fit")
        # Use image pixels' aspect ratio, rather than forcing a square plot.
        self.ax.imshow(generate_rgb_image(), extent=[0, 1, 0, 1],
                       origin="upper", aspect="auto")
        self.ax.set(xlim=(0, 1), ylim=(0, 1), xlabel="x", ylabel="y")
        self.ax.set_title("左键添加点 | 右键删除最近点" if self.chinese
                          else "Left click: add point | Right click: remove nearest point")
        self.scatter = self.ax.scatter([], [], s=50, color=(0.3, 0.6, 1),
                                       edgecolors="black", zorder=4)
        self.line, = self.ax.plot([], [], color="lime", lw=2, zorder=3)
        self.status = self.fig.text(0.08, 0.17, "", fontsize=9)
        self.buttons = []  # Keep widget references alive for the window lifetime.
        for left, label, callback in (
            (0.08, "Clear", self.on_clear), (0.30, "Save PNG", self.on_save),
            (0.52, "Fit: Spline", self.on_toggle_fit), (0.74, "Export TXT", self.on_export),
        ):
            button = Button(self.fig.add_axes([left, 0.055, 0.18, 0.07]), label)
            button.on_clicked(callback)
            self.buttons.append(button)
        self.fig.canvas.mpl_connect("button_press_event", self.onclick)
        self.update_fit()

    def set_status(self, message):
        self.status.set_text(message)
        self.fig.canvas.draw_idle()

    def update_fit(self):
        self.scatter.set_offsets(np.asarray(self.points).reshape(-1, 2))
        self.curve = None
        self.line.set_data([], [])
        if len(self.points) < 2:
            self.set_status(f"Points: {len(self.points)} | Add at least two points to fit.")
            return
        try:
            self.curve = fit_curve(self.points, self.fit_mode)
            xx = np.linspace(0, 1, 1500)
            self.line.set_data(xx, evaluate_y(self.curve, xx))
            mode = "PCHIP" if self.fit_mode == "spline" else self.fit_mode
            self.set_status(f"Points: {len(self.points)} | Mode: {mode} | x, y: 0-1")
        except (ValueError, ArithmeticError, np.linalg.LinAlgError, np.exceptions.RankWarning) as exc:
            self.curve = None
            self.set_status(f"Fit failed: {exc}")

    def onclick(self, event):
        if event.inaxes is not self.ax or event.xdata is None or event.ydata is None:
            return
        toolbar = getattr(self.fig.canvas, "toolbar", None)
        if toolbar is not None and toolbar.mode:
            return
        x, y = float(event.xdata), float(event.ydata)
        if not (0 <= x <= 1 and 0 <= y <= 1):
            return
        if event.button == 1:
            # y(x) cannot have two values at the same x: update that point.
            matches = [i for i, (px, _) in enumerate(self.points)
                       if abs(px - x) <= DUPLICATE_X_TOL]
            if matches:
                self.points[matches[0]] = [x, y]
                for i in reversed(matches[1:]):
                    self.points.pop(i)
            else:
                self.points.append([x, y])
        elif event.button == 3 and self.points:
            pixels = self.ax.transData.transform(self.points)
            click = self.ax.transData.transform([[x, y]])[0]
            self.points.pop(int(np.argmin(np.linalg.norm(pixels - click, axis=1))))
        else:
            return
        self.update_fit()

    def on_clear(self, event=None):
        self.points.clear()
        self.update_fit()

    def on_toggle_fit(self, event=None):
        self.fit_mode = "poly" if self.fit_mode == "spline" else "spline"
        self.buttons[2].label.set_text("Fit: Poly" if self.fit_mode == "poly" else "Fit: Spline")
        self.update_fit()

    def output_path(self, prefix, suffix):
        self.output_dir.mkdir(parents=True, exist_ok=True)
        return self.output_dir / f"{prefix}_{datetime.now():%Y%m%d_%H%M%S_%f}.{suffix}"

    def on_save(self, event=None):
        try:
            path = self.output_path("rgb_fit", "png")
            self.fig.savefig(path, dpi=150)
            self.set_status(f"Saved: {path.name}")
            print(f"Saved: {path}")
            return path
        except OSError as exc:
            self.set_status("Save failed; see terminal for details.")
            print(f"Save failed: {exc}")

    def on_export(self, event=None):
        if self.curve is None:
            self.set_status("Add at least two distinct x points and obtain a valid fit first.")
            return
        try:
            path = self.output_path("rgb_fit_expr", "txt")
            path.write_text(expression_text(self.curve, self.fit_mode, self.points), encoding="utf-8")
            self.set_status(f"Exported: {path.name}")
            print(f"Exported: {path}")
            return path
        except OSError as exc:
            self.set_status("Export failed; see terminal for details.")
            print(f"Export failed: {exc}")


def build_gui():
    app = RGBFitApp()
    plt.show()
    return app


if __name__ == "__main__":
    build_gui()
