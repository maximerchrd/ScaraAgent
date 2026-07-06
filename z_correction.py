import numpy as np
import csv

# ----------------------------------------------------------------------
# 1. Load data from CSV
# ----------------------------------------------------------------------
csv_path = "z_calibration.csv"

X, Y, Z = [], [], []
with open(csv_path, 'r') as f:
    reader = csv.reader(f)
    header = next(reader)  # skip header row
    for row in reader:
        if len(row) >= 3:
            X.append(float(row[0]))
            Y.append(float(row[1]))
            Z.append(float(row[2]))

X = np.array(X)
Y = np.array(Y)
Z = np.array(Z)

print(f"Loaded {len(X)} points from {csv_path}")

# ----------------------------------------------------------------------
# 2. Fit a 2nd-order polynomial: Z = a + b*X + c*Y + d*X^2 + e*Y^2 + f*X*Y
# ----------------------------------------------------------------------
A = np.column_stack([np.ones_like(X), X, Y, X**2, Y**2, X*Y])
coeffs, residuals, rank, s = np.linalg.lstsq(A, Z, rcond=None)

a, b, c, d, e, f = coeffs

print("\n=== Fitted coefficients (2nd order) ===")
print(f"a = {a:.6f}")
print(f"b = {b:.6f}")
print(f"c = {c:.6f}")
print(f"d = {d:.6f}")
print(f"e = {e:.6f}")
print(f"f = {f:.6f}")

# ----------------------------------------------------------------------
# 3. Check the fit quality
# ----------------------------------------------------------------------
Z_fit = A @ coeffs
errors = Z - Z_fit

print("\n=== Residuals (actual - fitted) ===")
for i, (x, y, z, zf, err) in enumerate(zip(X, Y, Z, Z_fit, errors)):
    flag = " ⚠️  LARGE ERROR" if abs(err) > 1.0 else ""
    print(f"Point {i:2d}:  X={x:6.1f}  Y={y:6.1f}  Z_meas={z:6.2f}  Z_fit={zf:6.2f}  error={err:+.3f} mm{flag}")

print(f"\nMax absolute error: {np.max(np.abs(errors)):.3f} mm")
print(f"Mean absolute error: {np.mean(np.abs(errors)):.3f} mm")

# ----------------------------------------------------------------------
# 4. Optional: refit after removing outliers
# ----------------------------------------------------------------------
outlier_threshold = 1.5  # mm — adjust as needed
mask = np.abs(errors) < outlier_threshold
num_outliers = np.sum(~mask)

if num_outliers > 0:
    print(f"\n=== Found {num_outliers} outlier(s) with error > {outlier_threshold} mm ===")
    for i in np.where(~mask)[0]:
        print(f"  Point {i}: X={X[i]:.1f}  Y={Y[i]:.1f}  Z={Z[i]:.2f}  error={errors[i]:+.3f} mm")

    X2, Y2, Z2 = X[mask], Y[mask], Z[mask]
    A2 = np.column_stack([np.ones_like(X2), X2, Y2, X2**2, Y2**2, X2*Y2])
    coeffs2, _, _, _ = np.linalg.lstsq(A2, Z2, rcond=None)
    a2, b2, c2, d2, e2, f2 = coeffs2

    Z_fit2 = A2 @ coeffs2
    errors2 = Z2 - Z_fit2

    print(f"\n=== Refitted coefficients (outliers removed) ===")
    print(f"a = {a2:.6f}")
    print(f"b = {b2:.6f}")
    print(f"c = {c2:.6f}")
    print(f"d = {d2:.6f}")
    print(f"e = {e2:.6f}")
    print(f"f = {f2:.6f}")
    print(f"Max abs error (clean points): {np.max(np.abs(errors2)):.3f} mm")
    print(f"Mean abs error (clean points): {np.mean(np.abs(errors2)):.3f} mm")

    # Use cleaned coefficients as default
    a, b, c, d, e, f = a2, b2, c2, d2, e2, f2
    print("\n⚠️  Using REFITTED coefficients (outliers excluded)")
else:
    print("\n✅ No outliers found — using original fit")

# ----------------------------------------------------------------------
# 5. Print the function for robot/controller.py
# ----------------------------------------------------------------------
print("\n=== Copy this into robot/controller.py ===")
print(f"""
    # Z sag correction coefficients (2nd order polynomial)
    # Fitted from {len(X)} calibration points
    _z_coeffs = [{a:.6f}, {b:.6f}, {c:.6f}, {d:.6f}, {e:.6f}, {f:.6f}]

    def get_table_z(self, x_mm, y_mm):
        \"\"\"Return Z (work coords) to touch the table at (x_mm, y_mm).\"\"\"
        c = self._z_coeffs
        return c[0] + c[1]*x_mm + c[2]*y_mm + c[3]*x_mm**2 + c[4]*y_mm**2 + c[5]*x_mm*y_mm
""")