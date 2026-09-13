use super::{FixedAngleConstraintShared, PointConstraintShared};
use crate::{
    dynamics::solver::{
        solver_body::{SolverBody, SolverBodyInertia},
        xpbd::*,
    },
    prelude::*,
};
use bevy::prelude::*;

/// Solver data for the [`FixedJoint`].
#[derive(Component, Clone, Copy, Debug, Default, PartialEq, Reflect)]
#[cfg_attr(feature = "serialize", derive(serde::Serialize, serde::Deserialize))]
#[cfg_attr(feature = "serialize", reflect(Serialize, Deserialize))]
#[reflect(Component, Debug, PartialEq)]
pub struct FixedJointSolverData {
    pub(super) point_constraint: PointConstraintShared,
    pub(super) angle_constraint: FixedAngleConstraintShared,
}

impl XpbdConstraintSolverData for FixedJointSolverData {
    fn clear_lagrange_multipliers(&mut self) {
        self.point_constraint.clear_lagrange_multipliers();
        self.angle_constraint.clear_lagrange_multipliers();
    }

    fn total_position_lagrange(&self) -> Vector {
        self.point_constraint.total_position_lagrange()
    }

    fn total_rotation_lagrange(&self) -> AngularVector {
        self.angle_constraint.total_rotation_lagrange()
    }
}

impl XpbdConstraint<2> for FixedJoint {
    type SolverData = FixedJointSolverData;

    fn prepare(
        &mut self,
        bodies: [&RigidBodyQueryReadOnlyItem; 2],
        solver_data: &mut FixedJointSolverData,
    ) {
        let [body1, body2] = bodies;

        let Some(local_anchor1) = self.local_anchor1() else {
            return;
        };
        let Some(local_anchor2) = self.local_anchor2() else {
            return;
        };
        let Some(local_basis1) = self.local_basis1() else {
            return;
        };
        let Some(local_basis2) = self.local_basis2() else {
            return;
        };

        // Prepare the point-to-point constraint.
        solver_data
            .point_constraint
            .prepare(bodies, local_anchor1, local_anchor2);

        // Prepare the angular constraint.
        solver_data.angle_constraint.prepare(
            body1.rotation,
            body2.rotation,
            local_basis1,
            local_basis2,
        );
    }

    fn solve(
        &mut self,
        bodies: [&mut SolverBody; 2],
        inertias: [&SolverBodyInertia; 2],
        solver_data: &mut FixedJointSolverData,
        dt: Scalar,
    ) {
        let [body1, body2] = bodies;

        #[cfg(feature = "3d")]
        if solve_fixed_block(
            self,
            [&mut *body1, &mut *body2],
            inertias,
            solver_data,
            self.point_compliance,
            self.angle_compliance,
            dt,
        ) {
            return;
        }

        // Solve the angular constraint.
        solver_data
            .angle_constraint
            .solve([body1, body2], inertias, self.angle_compliance, dt);

        // Solve the point-to-point constraint.
        solver_data
            .point_constraint
            .solve([body1, body2], inertias, self.point_compliance, dt);
    }
}

/// Solves all six degrees of freedom of a fixed joint **together**, as one 6x6 system.
///
/// Solving the angle and the point one after the other is exact for each and wrong for the joint.
/// A point anchor that sits off a body's axis is moved most cheaply by *turning* the body, so the
/// point solve fixes a sideways separation by twisting, and the angle solve that follows untwists
/// it and moves the anchor back. For a slender body — a rod, thousands of times easier to spin about
/// its length than end over end — the two ping-pong every substep, and more substeps only make them
/// ping-pong faster.
///
/// With `S = [r]×` and `M = I⁻¹` (world space) for each body, a positional impulse `p` at the anchors
/// and an angular impulse `τ` change the separation `s` and rotation error `θ` by
///
/// ```text
/// Δs = −(K_pp·p + K_pτ·τ)      K_pp = (m₁⁻¹ + m₂⁻¹) − S₁M₁S₁ − S₂M₂S₂      K_pτ = −(S₁M₁ + S₂M₂)
/// Δθ = −(K_τp·p + K_ττ·τ)      K_τp = M₁S₁ + M₂S₂ = K_pτᵀ                    K_ττ = M₁ + M₂
/// ```
///
/// so `(K + α̃)·[p; τ] = [s; θ]` drives both to what their compliances allow at once. `K` is
/// symmetric, which is the check that the signs above agree with the two scalar solves they
/// replace. Returns `false` without touching anything when `K` is singular, so the caller can fall
/// back to solving the parts separately.
#[cfg(feature = "3d")]
fn solve_fixed_block(
    joint: &FixedJoint,
    bodies: [&mut SolverBody; 2],
    inertias: [&SolverBodyInertia; 2],
    solver_data: &mut FixedJointSolverData,
    point_compliance: Scalar,
    angle_compliance: Scalar,
    dt: Scalar,
) -> bool {
    let [body1, body2] = bodies;
    let [inertia1, inertia2] = inertias;
    let point = &solver_data.point_constraint;
    let angle = &solver_data.angle_constraint;

    let inv_mass = inertia1.effective_inv_mass() + inertia2.effective_inv_mass();
    let m1 = inertia1.effective_inv_angular_inertia().to_mat3();
    let m2 = inertia2.effective_inv_angular_inertia().to_mat3();

    // The same errors the two scalar solves would compute, from the same state.
    let r1 = body1.delta_rotation * point.world_r1;
    let r2 = body2.delta_rotation * point.world_r2;
    let separation =
        (body2.delta_position - body1.delta_position) + (r2 - r1) + point.center_difference;
    let theta = -2.0
        * (angle.rotation_difference * body1.delta_rotation.0 * body2.delta_rotation.0.inverse())
            .xyz();

    let cross = |r: Vector| {
        Matrix::from_cols(
            Vector::new(0.0, r.z, -r.y),
            Vector::new(-r.z, 0.0, r.x),
            Vector::new(r.y, -r.x, 0.0),
        )
    };
    let (s1, s2) = (cross(r1), cross(r2));
    let h2 = dt * dt;
    let k_pp = Matrix::from_diagonal(inv_mass + Vector::splat(point_compliance / h2))
        - s1 * m1 * s1
        - s2 * m2 * s2;
    let k_pt = -(s1 * m1 + s2 * m2);
    let k_tp = m1 * s1 + m2 * s2;
    let k_tt = m1 + m2 + Matrix::from_diagonal(Vector::splat(angle_compliance / h2));

    let mut a = [[0.0 as Scalar; 7]; 6];
    for i in 0..3 {
        for j in 0..3 {
            a[i][j] = k_pp.col(j)[i];
            a[i][j + 3] = k_pt.col(j)[i];
            a[i + 3][j] = k_tp.col(j)[i];
            a[i + 3][j + 3] = k_tt.col(j)[i];
        }
        a[i][6] = separation[i];
        a[i + 3][6] = theta[i];
    }
    let scale = (0..6).map(|i| a[i][i].abs()).fold(0.0, Scalar::max);
    if !(scale > 0.0) || !scale.is_finite() {
        return false;
    }

    // Gaussian elimination with partial pivoting. Six unknowns; nothing smarter is worth it.
    for col in 0..6 {
        let pivot = (col..6)
            .max_by(|&x, &y| {
                a[x][col]
                    .abs()
                    .partial_cmp(&a[y][col].abs())
                    .unwrap_or(core::cmp::Ordering::Equal)
            })
            .unwrap_or(col);
        if !(a[pivot][col].abs() > scale * 1.0e-7) {
            return false;
        }
        a.swap(col, pivot);
        for row in (col + 1)..6 {
            let f = a[row][col] / a[col][col];
            if f != 0.0 {
                for k in col..7 {
                    a[row][k] -= f * a[col][k];
                }
            }
        }
    }
    let mut x = [0.0 as Scalar; 6];
    for row in (0..6).rev() {
        let tail: Scalar = ((row + 1)..6).map(|k| a[row][k] * x[k]).sum();
        x[row] = (a[row][6] - tail) / a[row][row];
    }
    if x.iter().any(|v| !v.is_finite()) {
        return false;
    }

    let p = Vector::new(x[0], x[1], x[2]);
    let tau = Vector::new(x[3], x[4], x[5]);
    let (inv1, inv2) = (
        inertia1.effective_inv_angular_inertia(),
        inertia2.effective_inv_angular_inertia(),
    );
    joint.apply_positional_impulse(body1, body2, inertia1, inertia2, p, r1, r2);
    joint.apply_angular_impulse(body1, body2, inv1, inv2, tau);
    solver_data.point_constraint.total_lagrange += p;
    // The angle solve reports `Δλ·n`, which is minus the impulse it applied.
    solver_data.angle_constraint.total_lagrange -= tau;
    true
}

impl PositionConstraint for FixedJoint {}

impl AngularConstraint for FixedJoint {}
