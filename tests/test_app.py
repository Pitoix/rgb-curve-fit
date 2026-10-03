import os
os.environ['MPLBACKEND'] = 'Agg'
import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import numpy as np
import matplotlib.pyplot as plt
from gui_rgb_fit import (RGBFitApp, curve_rgb, evaluate_y, expression_text,
                         fit_curve, generate_rgb_image)


class FitTests(unittest.TestCase):
    def test_unsorted_points_and_rgb(self):
        pts = [[0.8, 0.7], [0.1, 0.2], [0.5, 0.4]]
        for mode in ('spline', 'poly'):
            curve = fit_curve(pts, mode)
            x, y = np.array(pts).T
            np.testing.assert_allclose(curve(x), y, atol=1e-12)
            expected = np.column_stack((x, y, 1-x*y))
            np.testing.assert_allclose(curve_rgb(curve, x), expected, atol=1e-12)
            xx = np.linspace(0, 1, 100)
            np.testing.assert_allclose(evaluate_y(curve, xx),
                                       evaluate_y(fit_curve(pts[::-1], mode), xx))

    def test_gradient_uses_upward_green_axis(self):
        image = generate_rgb_image(3, 3)
        np.testing.assert_allclose(image[0, :, 1], 1)
        np.testing.assert_allclose(image[-1, :, 1], 0)
        np.testing.assert_allclose(image[..., 2],
                                   1 - image[..., 0] * image[..., 1])

    def test_pchip_shape_and_endpoint_hold(self):
        pts = np.array([[.1, .1], [.4, .9], [.41, .05], [.7, .8], [.9, .2]])
        curve = fit_curve(pts, 'spline')
        inside_x = np.linspace(pts[:, 0].min(), pts[:, 0].max(), 10001)
        inside_y = curve(inside_x)
        self.assertGreaterEqual(float(inside_y.min()), 0)
        self.assertLessEqual(float(inside_y.max()), 1)
        self.assertTrue(np.isnan(curve(0)))
        self.assertTrue(np.isnan(curve(1)))
        np.testing.assert_allclose(evaluate_y(curve, [0, .1, .9, 1]),
                                   [.1, .1, .2, .2])
        np.testing.assert_allclose(curve(pts[:, 0]), pts[:, 1], atol=1e-12)

    def test_duplicate_x_keeps_last_point(self):
        curve = fit_curve([[.2, .1], [.8, .7], [.2 + 5e-9, .9]], 'spline')
        self.assertAlmostEqual(float(curve(.2 + 5e-9)), .9)
        self.assertAlmostEqual(float(curve(.8)), .7)
        with self.assertRaises(ValueError):
            fit_curve([[.2, .1], [.2 + 5e-9, .9]], 'spline')

    def test_invalid_points(self):
        for pts in ([], [[0, 0]], [[0, 0], [np.nan, 1]], [[0, 0], [1, 2]]):
            with self.assertRaises(ValueError):
                fit_curve(pts)

    def test_export_equations_match_curve(self):
        pts = [[0.15, 0.8], [0.3, 0.1], [0.7, 0.9], [0.85, 0.3]]
        for mode in ('spline', 'poly'):
            curve = fit_curve(pts, mode)
            content = expression_text(curve, mode, pts)
            lines = content.splitlines()
            section = next(i for i, line in enumerate(lines)
                           if 'SEGMENTS / 分段系数]' in line)
            header = next(i for i in range(section, len(lines))
                          if lines[i].lstrip().startswith('#') and
                          'x0' in lines[i] and 'x1' in lines[i])
            segment_rows = []
            for line in lines[header + 2:]:
                if not line.strip():
                    break
                segment_rows.append([value.strip() for value in line.split('|')])
            self.assertEqual(len(segment_rows), len(curve.x) - 1)
            for row in segment_rows:
                start, end = map(float, row[1:3])
                coefficients = np.array(row[3:], dtype=float)
                x = np.linspace(start, end, 31)
                t = (x - start) / (end - start)
                actual = np.polyval(coefficients, t)
                np.testing.assert_allclose(np.clip(actual, 0, 1),
                                           evaluate_y(curve, x), atol=1e-10)
            rgb = curve_rgb(curve, np.linspace(0, 1, 101))
            self.assertTrue(np.all((rgb >= 0) & (rgb <= 1)))
            self.assertTrue(np.all((255 * rgb >= 0) & (255 * rgb <= 255)))
            self.assertIn('G(x) = y(x)', content)
            self.assertIn('B(x) = 1 - x*y(x)', content)
        self.assertIn('Mode: PCHIP', expression_text(fit_curve(pts), 'spline', pts))
        self.assertIn('Hold y at the nearest sampled endpoint',
                      expression_text(fit_curve(pts), 'spline', pts))

    def test_interpolating_export_uses_one_y_column(self):
        pts = [[.15, .8], [.3, .1], [.7, .9], [.85, .3]]
        for mode in ('spline', 'poly'):
            lines = expression_text(fit_curve(pts, mode), mode, pts).splitlines()
            header = next(line for line in lines
                          if line.lstrip().startswith('#') and
                          [value.strip() for value in line.split('|')][:3] ==
                          ['#', 'x', 'y'])
            self.assertEqual([value.strip() for value in header.split('|')],
                             ['#', 'x', 'y', 'R', 'G', 'B'])
            self.assertIn('The curve passes through each point: y = G.', lines)

    def test_poly_point_table_distinguishes_clicked_and_fitted_y(self):
        pts = [[0, .1], [.15, .8], [.3, .2], [.45, .9],
               [.6, .1], [.75, .8], [.9, .2], [1, .9]]
        curve = fit_curve(pts, 'poly')
        lines = expression_text(curve, 'poly', pts).splitlines()
        header = next(i for i, line in enumerate(lines)
                      if 'y_point' in line and 'y_curve' in line and '|' in line)
        rows = [[float(value.strip()) for value in line.split('|')]
                for line in lines[header + 2:] if line.strip()]
        self.assertEqual(len(rows), len(pts))
        self.assertTrue(any(abs(row[2] - row[3]) > .1 for row in rows))
        for _, x, y_point, y_curve, red, green, blue in rows:
            self.assertAlmostEqual(y_point, dict(pts)[x])
            self.assertAlmostEqual(y_curve, float(evaluate_y(curve, x)))
            self.assertAlmostEqual(red, x)
            self.assertAlmostEqual(green, y_curve)
            self.assertAlmostEqual(blue, 1 - red * green)

    def test_dense_segment_export_preserves_endpoints(self):
        pts = [[.12345678901234566, .1], [.12345680001234566, .9],
               [.12345681101234566, .2], [.8, .7]]
        curve = fit_curve(pts)
        lines = expression_text(curve, 'spline', pts).splitlines()
        header = next(i for i, line in enumerate(lines)
                      if line.lstrip().startswith('#') and 'x0' in line and 'x1' in line)
        row = [float(value.strip()) for value in lines[header + 2].split('|')[1:]]
        start, end = row[:2]
        x = np.linspace(pts[0][0], pts[1][0], 101)
        exported = np.polyval(row[2:], (x - start) / (end - start))
        np.testing.assert_allclose(exported, evaluate_y(curve, x),
                                   rtol=0, atol=1e-10)


class GUITests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.app = RGBFitApp(self.tmp.name)

    def tearDown(self):
        plt.close(self.app.fig)
        self.tmp.cleanup()

    def click(self, x, y, button=1, ax=None):
        self.app.onclick(SimpleNamespace(inaxes=self.app.ax if ax is None else ax,
                                        xdata=x, ydata=y, button=button))

    def test_buttons_do_not_add_points(self):
        for button in self.app.buttons:
            self.click(.5, .5, ax=button.ax)
        self.assertEqual(self.app.points, [])

    def test_ui_vertical_axis_shows_y_upward(self):
        self.assertEqual(self.app.ax.get_ylabel(), 'y')
        self.assertEqual(tuple(self.app.ax.get_ylim()), (0, 1))
        self.assertEqual(tuple(self.app.ax.images[0].get_extent()), (0, 1, 0, 1))
        self.click(.2, 1)
        self.click(.8, 0)
        np.testing.assert_allclose(self.app.points, [[.2, 1], [.8, 0]])
        np.testing.assert_allclose(self.app.scatter.get_offsets(), [[.2, 1], [.8, 0]])
        x, y = self.app.line.get_data()
        np.testing.assert_allclose(y, evaluate_y(self.app.curve, x))
        self.click(.2, 1, 3)
        np.testing.assert_allclose(self.app.points, [[.8, 0]])

    def test_add_replace_toggle_delete_clear(self):
        self.click(.8, .7)
        self.click(.2, .3)
        self.click(.2, .4)
        self.assertEqual(len(self.app.points), 2)
        self.assertAlmostEqual(float(self.app.curve(.2)), .4)
        self.app.on_toggle_fit()
        self.assertEqual(self.app.fit_mode, 'poly')
        self.assertIs(self.app.line.axes, self.app.ax)
        self.assertEqual(len(self.app.buttons[2].ax.lines), 0)
        self.click(.2, .4, 3)
        self.assertIsNone(self.app.curve)
        self.assertEqual(len(self.app.line.get_xdata()), 0)
        self.app.on_clear()
        self.assertEqual(self.app.scatter.get_offsets().shape, (0, 2))

    def test_toolbar_and_middle_button(self):
        self.click(.5, .5, 2)
        self.assertFalse(self.app.points)
        old = self.app.fig.canvas.toolbar
        self.app.fig.canvas.toolbar = SimpleNamespace(mode='pan/zoom')
        try:
            self.click(.5, .5)
            self.assertFalse(self.app.points)
        finally:
            self.app.fig.canvas.toolbar = old

    def test_save_export(self):
        self.assertIsNone(self.app.on_export())
        self.click(.8, .3)
        self.click(.2, .7)
        image = self.app.on_save()
        self.assertTrue(image.read_bytes().startswith(b'\x89PNG'))
        for mode in ('spline', 'poly'):
            path = self.app.on_export()
            expected_mode = 'PCHIP' if mode == 'spline' else mode
            self.assertIn(f'Mode: {expected_mode}', path.read_text(encoding='utf-8'))
            self.app.on_toggle_fit()
        self.assertEqual(len(list(Path(self.tmp.name).glob('*.txt'))), 2)

    def test_save_error_is_reported(self):
        target = Path(self.tmp.name) / 'file'
        target.write_text('not a directory')
        self.app.output_dir = target
        self.assertIsNone(self.app.on_save())
        self.assertIn('failed', self.app.status.get_text())


if __name__ == '__main__':
    unittest.main()
