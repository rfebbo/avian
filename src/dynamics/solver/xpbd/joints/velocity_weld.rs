//! **A fixed joint solved with the contacts** (place-99).
//!
//! The XPBD joints run after the contacts have been solved and relaxed, and correct *positions*. For a light body a
//! joint holds to a heavy one under something resting on it — a 3 mm plate (1.4 kg) welded to a post, a 12 kg block on
//! it — the contact's impulse mostly moves the light body, the joint then puts it back every substep without the contact
//! seeing it, and the heavy body sinks until the soft contact alone holds it: the block rested 14 mm into the plate.
//!
//! A [`FixedJoint`] with [`SolveWithContacts`] leaves the XPBD solver and is solved the way Box2D v3 solves its weld
//! joint: as a velocity constraint, in the contacts' own stages — warm started with them, solved with a position bias
//! before them, and relaxed before them — so each sees what the other did within the substep. All six degrees of
//! freedom together (the 6×6 block of `fixed.rs`, here at the velocity level). Its angle compliance is a spring in the
//! constraint-force-mixing form (`γ = compliance / h²` on the angular diagonal: the same compliance), so a weld still
//! bends as it did. Its forces go to [`JointForces`] as the impulse over the substep, as `fixed.rs` reports them.

use super::FixedJointSolverData;
use crate::{
    dynamics::solver::{
        SolverConfig,
        softness_parameters::{SoftnessCoefficients, SoftnessParameters},
        solver_body::{SolverBody, SolverBodyInertia},
    },
    prelude::*,
};
use crate::dynamics::joints::EntityConstraint;
use bevy::prelude::*;

/// Solve this [`FixedJoint`] as a velocity constraint with the contacts, not with the XPBD joints (place-99; see the
/// module).
#[derive(Component, Clone, Copy, Debug, Default, PartialEq, Reflect)]
#[reflect(Component, Debug, Default, PartialEq)]
#[require(VelocityWeldData)]
pub struct SolveWithContacts;

/// What a [`SolveWithContacts`] joint carries between stages and steps.
#[derive(Component, Clone, Copy, Debug, Default, PartialEq, Reflect)]
#[reflect(Component, Debug, PartialEq)]
pub struct VelocityWeldData {
    /// The impulse applied each substep (warm started into the next), linear and angular.
    impulse: Vector,
    angular_impulse: Vector,
    /// The rigid parts' softness, and the angular spring's `γ` (zero: rigid).
    bias: Scalar,
    mass_scale: Scalar,
    impulse_scale: Scalar,
    gamma: Scalar,
    h: Scalar,
}

/// Before the substeps: each joint's frames from where its bodies are, as `fixed.rs` prepares them, and its softness.
pub(crate) fn prepare_velocity_welds(
    bodies: Query<RigidBodyQueryReadOnly, Without<RigidBodyDisabled>>,
    mut joints: Query<
        (&mut FixedJoint, &mut FixedJointSolverData, &mut VelocityWeldData),
        (With<SolveWithContacts>, Without<RigidBody>, Without<JointDisabled>),
    >,
    substep_time: Res<Time<Substeps>>,
    config: Res<SolverConfig>,
) {
    let h = substep_time.delta_secs_f64() as Scalar;
    if h <= 0.0 {
        return;
    }
    // Rigid: Box2D's joint softness — a quarter of the substep rate, twice-critical damping.
    let rigid: SoftnessCoefficients =
        SoftnessParameters::new(config.weld_damping_ratio, config.weld_frequency_factor * 0.25 / h)
            .compute_coefficients(h);
    for (mut joint, mut solver_data, mut weld) in &mut joints {
        if let Ok([body1, body2]) = bodies.get_many(joint.entities()) {
            crate::dynamics::solver::xpbd::XpbdConstraint::prepare(&mut *joint, [&body1, &body2], &mut solver_data);
        }
        weld.bias = rigid.bias;
        weld.mass_scale = rigid.mass_scale;
        weld.impulse_scale = rigid.impulse_scale;
        weld.gamma = if joint.angle_compliance > 0.0 { joint.angle_compliance / (h * h) } else { 0.0 };
        weld.h = h;
    }
}

/// The two bodies of a joint, or stand-ins for a static one.
macro_rules! bodies_of {
    ($bodies:expr, $joint:expr, $d1:ident, $d2:ident) => {{
        let [entity1, entity2] = $joint.entities();
        let (mut body1, mut inertia1) = (&mut $d1, &SolverBodyInertia::DUMMY);
        let (mut body2, mut inertia2) = (&mut $d2, &SolverBodyInertia::DUMMY);
        if let Ok((body, inertia)) = unsafe { $bodies.get_unchecked(entity1) } {
            body1 = body.into_inner();
            inertia1 = inertia;
        }
        if let Ok((body, inertia)) = unsafe { $bodies.get_unchecked(entity2) } {
            body2 = body.into_inner();
            inertia2 = inertia;
        }
        match (inertia1.dominance() - inertia2.dominance()).cmp(&0) {
            core::cmp::Ordering::Greater => inertia1 = &SolverBodyInertia::DUMMY,
            core::cmp::Ordering::Less => inertia2 = &SolverBodyInertia::DUMMY,
            _ => {}
        }
        (body1, inertia1, body2, inertia2)
    }};
}

/// Applies a linear impulse `p` at the anchors (to the second body; its opposite to the first) and an angular one.
fn apply(body1: &mut SolverBody, body2: &mut SolverBody, inertia1: &SolverBodyInertia, inertia2: &SolverBodyInertia, r1: Vector, r2: Vector, p: Vector, tau: Vector) {
    body1.linear_velocity -= p * inertia1.effective_inv_mass();
    body1.angular_velocity -= inertia1.effective_inv_angular_inertia() * (r1.cross(p) + tau);
    body2.linear_velocity += p * inertia2.effective_inv_mass();
    body2.angular_velocity += inertia2.effective_inv_angular_inertia() * (r2.cross(p) + tau);
}

/// Warm start: last substep's impulses, applied again (the velocities were integrated afresh).
pub(crate) fn warm_start_velocity_welds(
    bodies: Query<(&mut SolverBody, &SolverBodyInertia), Without<RigidBodyDisabled>>,
    joints: Query<(&FixedJoint, &FixedJointSolverData, &VelocityWeldData), (With<SolveWithContacts>, Without<RigidBody>, Without<JointDisabled>)>,
    config: Res<SolverConfig>,
) {
    let (mut d1, mut d2) = (SolverBody::default(), SolverBody::default());
    for (joint, data, weld) in &joints {
        let (body1, inertia1, body2, inertia2) = bodies_of!(bodies, joint, d1, d2);
        let r1 = body1.delta_rotation * data.point_constraint.world_r1;
        let r2 = body2.delta_rotation * data.point_constraint.world_r2;
        let k = config.warm_start_coefficient;
        apply(body1, body2, inertia1, inertia2, r1, r2, weld.impulse * k, weld.angular_impulse * k);
    }
}

/// One velocity solve of every [`SolveWithContacts`] joint, with its position bias (`USE_BIAS`) or to relax.
pub(crate) fn solve_velocity_welds<const USE_BIAS: bool>(
    bodies: Query<(&mut SolverBody, &SolverBodyInertia), Without<RigidBodyDisabled>>,
    mut joints: Query<(&FixedJoint, &FixedJointSolverData, &mut VelocityWeldData), (With<SolveWithContacts>, Without<RigidBody>, Without<JointDisabled>)>,
) {
    let (mut d1, mut d2) = (SolverBody::default(), SolverBody::default());
    for (joint, data, mut weld) in &mut joints {
        let (body1, inertia1, body2, inertia2) = bodies_of!(bodies, joint, d1, d2);
        let point = &data.point_constraint;
        let angle = &data.angle_constraint;
        let r1 = body1.delta_rotation * point.world_r1;
        let r2 = body2.delta_rotation * point.world_r2;
        // The errors, as `fixed.rs` measures them: the anchors' separation, and the second body's turn from the first.
        let separation = (body2.delta_position - body1.delta_position) + (r2 - r1) + point.center_difference;
        let theta = -2.0 * (angle.rotation_difference * body1.delta_rotation.0 * body2.delta_rotation.0.inverse()).xyz();
        // How fast they grow.
        let cdot_p = (body2.linear_velocity + body2.angular_velocity.cross(r2)) - (body1.linear_velocity + body1.angular_velocity.cross(r1));
        let cdot_t = body2.angular_velocity - body1.angular_velocity;

        let inv_mass = inertia1.effective_inv_mass() + inertia2.effective_inv_mass();
        let m1 = inertia1.effective_inv_angular_inertia().to_mat3();
        let m2 = inertia2.effective_inv_angular_inertia().to_mat3();
        let cross = |r: Vector| Matrix::from_cols(Vector::new(0.0, r.z, -r.y), Vector::new(-r.z, 0.0, r.x), Vector::new(r.y, -r.x, 0.0));
        let (s1, s2) = (cross(r1), cross(r2));
        let k_pp = Matrix::from_diagonal(inv_mass) - s1 * m1 * s1 - s2 * m2 * s2;
        let k_pt = -(s1 * m1 + s2 * m2);
        let k_tp = m1 * s1 + m2 * s2;
        let k_tt = m1 + m2 + Matrix::from_diagonal(Vector::splat(weld.gamma));

        // The rigid parts: Box2D's soft step (bias only in the biased solve). The angular spring, when compliant, in
        // constraint-force-mixing form, in both: `C/h + γλ`.
        let (bias_p, mass_p, keep_p) = if USE_BIAS { (separation * weld.bias, weld.mass_scale, weld.impulse_scale) } else { (Vector::ZERO, 1.0, 0.0) };
        let spring = weld.gamma > 0.0;
        let (bias_t, mass_t, keep_t) = if spring {
            (theta / weld.h + weld.angular_impulse * weld.gamma, 1.0, 0.0)
        } else if USE_BIAS {
            (theta * weld.bias, weld.mass_scale, weld.impulse_scale)
        } else {
            (Vector::ZERO, 1.0, 0.0)
        };
        let rhs_p = cdot_p + bias_p;
        let rhs_t = cdot_t + bias_t;
        let Some((x_p, x_t)) = solve6(k_pp, k_pt, k_tp, k_tt, rhs_p, rhs_t) else { continue };
        let p = -x_p * mass_p - weld.impulse * keep_p;
        let tau = -x_t * mass_t - weld.angular_impulse * keep_t;
        if !p.is_finite() || !tau.is_finite() {
            continue;
        }
        weld.impulse += p;
        weld.angular_impulse += tau;
        apply(body1, body2, inertia1, inertia2, r1, r2, p, tau);
    }
}

/// After the substeps: the forces, as the last substep's impulse over the substep — the force on the first body, and
/// the torque with the sign `fixed.rs` reports.
pub(crate) fn write_velocity_weld_forces(mut joints: Query<(&VelocityWeldData, &mut JointForces), With<SolveWithContacts>>) {
    for (weld, mut forces) in &mut joints {
        if weld.h > 0.0 {
            forces.set_force(-weld.impulse / weld.h);
            // The angle solve's sign, which `fixed.rs` reports as it applies it (a plate hanging off a wall: +40 N·m about
            // +Z from XPBD, and so here).
            forces.set_torque(weld.angular_impulse / weld.h);
        }
    }
}

/// `[K_pp K_pt; K_tp K_tt] · [x_p; x_t] = [r_p; r_t]` by Gaussian elimination with partial pivoting, or `None` if
/// singular.
fn solve6(k_pp: Matrix, k_pt: Matrix, k_tp: Matrix, k_tt: Matrix, r_p: Vector, r_t: Vector) -> Option<(Vector, Vector)> {
    let mut a = [[0.0 as Scalar; 7]; 6];
    for i in 0..3 {
        for j in 0..3 {
            a[i][j] = k_pp.col(j)[i];
            a[i][j + 3] = k_pt.col(j)[i];
            a[i + 3][j] = k_tp.col(j)[i];
            a[i + 3][j + 3] = k_tt.col(j)[i];
        }
        a[i][6] = r_p[i];
        a[i + 3][6] = r_t[i];
    }
    let scale = (0..6).map(|i| a[i][i].abs()).fold(0.0, Scalar::max);
    if !(scale > 0.0) || !scale.is_finite() {
        return None;
    }
    for col in 0..6 {
        let pivot = (col..6).max_by(|&x, &y| a[x][col].abs().partial_cmp(&a[y][col].abs()).unwrap_or(core::cmp::Ordering::Equal)).unwrap_or(col);
        if !(a[pivot][col].abs() > scale * 1.0e-7) {
            return None;
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
    Some((Vector::new(x[0], x[1], x[2]), Vector::new(x[3], x[4], x[5])))
}
